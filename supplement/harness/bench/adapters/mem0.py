"""Mem0Adapter — Mem0 baseline for PWM-Bench T1.

Mem0 (Chhikara et al. 2025) is a vector + extracted-entity store over a
textified archive. For PWM-Bench T1 we treat each moment description as a
single "memory" associated with the persona, push them into Mem0 once at
adapter init, then query via Mem0's `search` + LLM completion path at
query time.

Mem0 does not natively expose moment-ID citations the way our grounded
retrieval does — Mem0 returns memory rows it considers relevant. We map
each Mem0 memory row back to its source moment_id via the `metadata.moment_id`
field we set at ingest. Citations therefore equal the moment IDs of the
memories Mem0 retrieved for the query.

This adapter requires the `mem0ai` SDK. We do NOT add it to the project's
default deps — it gates on import. Install with:
    uv add mem0ai
or:
    uv pip install mem0ai

Mem0 also requires an OpenAI / equivalent API key for its embedder + LLM
unless configured otherwise; we pass through GEMINI_API_KEY / OPENAI_API_KEY
from the environment.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ..substrate import SubstrateResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MEM0_INSTALL_HINT = (
    "Mem0Adapter requires the `mem0ai` SDK. Install with `uv add mem0ai` "
    "(or `uv pip install mem0ai`). Mem0 also needs an OpenAI API key by "
    "default — set OPENAI_API_KEY in your environment, or configure Mem0 "
    "to use a different provider per https://docs.mem0.ai/."
)


class Mem0Adapter:
    """Mem0 baseline substrate adapter for PWM-Bench T1.

    Ingests the persona's moment descriptions into Mem0 at first query
    (lazy init), then routes T1 queries through `mem0.Memory.search()` +
    a synthesis LLM call.
    """

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        top_k: int = 20,
        model: str | None = None,
    ):
        # PWMBENCH_SYNTH_MODEL pins the answering LLM across all systems so the
        # comparison isolates the memory layer, not the model. Explicit arg wins.
        model = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or "gemini-2.5-flash"
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(_MEM0_INSTALL_HINT) from e

        self._Memory = Memory
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._top_k = top_k
        self._model_name = model
        self._mem: Memory | None = None
        self._ingested = False
        self._llm = None

    def name(self) -> str:
        return "mem0"

    def _ensure_ingested(self) -> None:
        if self._ingested:
            return
        # Pin Mem0's LLM + embedder to Gemini so the baseline needs no OpenAI
        # key (the project already has a Gemini key). Mem0 is backend-agnostic;
        # we use Gemini Flash for extraction and gemini-embedding-001 (768-d MRL
        # truncation) for the vector store. Vector store stays Mem0's default
        # (embedded Qdrant), sized to the embedder's 768 dims.
        _key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
        )
        # mem0's Gemini embedder emits 1536-d for gemini-embedding-001; the
        # vector store must match (it ignores a smaller embedding_dims request).
        _dims = 1536
        _emb_cfg = {"model": "models/gemini-embedding-001", "embedding_dims": _dims}
        _llm_cfg = {"model": "gemini-2.5-flash", "temperature": 0.1, "max_tokens": 1024}
        if _key:
            _emb_cfg["api_key"] = _key
            _llm_cfg["api_key"] = _key
        self._mem = self._Memory.from_config({
            "llm": {"provider": "gemini", "config": _llm_cfg},
            "embedder": {"provider": "gemini", "config": _emb_cfg},
            "vector_store": {
                "provider": "qdrant",
                "config": {"embedding_model_dims": _dims},
            },
        })

        # Load all moment descriptions for the persona from Golgi's repo
        # (we don't share storage with Mem0; this is a one-shot textify
        # + ingest at adapter init, mimicking how a Mem0 deployment would
        # have been seeded from the archive).
        from ...interfaces.cli.shared import get_repos
        repos = get_repos(
            persona=self._persona, data_root=self._data_root, read_only=True
        )
        moments = repos.moment.find_all(limit=20000)
        logger.info("Mem0 ingesting %d moments for persona=%s", len(moments), self._persona)
        count = 0
        for m in moments:
            desc = (m.description or "").strip()
            if not desc:
                continue
            # Mem0's add() runs an LLM extraction; at scale Gemini rate-limits,
            # and a throttled extraction returns EMPTY results WITHOUT raising,
            # leaving the store unpopulated. Retry on both exceptions and empty
            # results with backoff so the full archive actually lands.
            for attempt in range(4):
                try:
                    r = self._mem.add(
                        messages=[{"role": "user", "content": desc}],
                        user_id=self._persona,
                        metadata={"moment_id": m.id},
                    )
                    added = r.get("results", r) if isinstance(r, dict) else r
                    if added:
                        count += 1
                        break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        logger.debug("mem0 add failed for moment %s: %s", m.id, e)
                time.sleep(1.0 * (attempt + 1))
        logger.info("Mem0 ingested %d / %d moments into store", count, len(moments))
        if count == 0:
            logger.warning("Mem0 store is EMPTY after ingest — results will be null")
        self._ingested = True

        # Synthesis LLM (shared with PWM adapter — Gemini Flash for cost).
        from ...infrastructure.config import Settings
        from ...infrastructure.llm import get_synth_service
        settings = Settings.from_env()
        # Routed by model name so the §4.8 matrix can pin the answer model
        # (Gemini / GPT-4o / on-device Ollama) across all systems.
        self._llm = get_synth_service(self._model_name, settings)

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"Mem0Adapter bound to persona={self._persona!r}, got {persona!r}"
            )
        self._ensure_ingested()
        assert self._mem is not None and self._llm is not None

        t0 = time.perf_counter()
        try:
            # mem0ai >= 2.x requires user_id in filters dict, not as top-level kwarg.
            results = self._mem.search(
                query=q,
                filters={"user_id": self._persona},
                limit=self._top_k,
            )
        except Exception as e:
            logger.exception("Mem0 search failed")
            return SubstrateResponse(
                answer=f"[Mem0 search error: {e}]",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Mem0 SDK returns either a dict {"results": [...]} or a list,
        # depending on version. Normalize.
        rows = results.get("results", results) if isinstance(results, dict) else results
        if not rows:
            return SubstrateResponse(
                answer="Mem0 found no relevant memories for this query.",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": 0},
            )

        cited: list[str] = []
        ctx_lines: list[str] = []
        for i, row in enumerate(rows, start=1):
            text = row.get("memory") or row.get("text") or ""
            meta = row.get("metadata") or {}
            mid = meta.get("moment_id")
            if mid and mid not in cited:
                cited.append(mid)
            ctx_lines.append(f"[{i}] {text}")

        ctx = "\n".join(ctx_lines)
        prompt = (
            f"Question: {q}\n\n"
            f"Relevant memories from Mem0:\n{ctx}\n\n"
            "Answer the question using only these memories. If they don't "
            "contain enough information, say so plainly."
        )
        try:
            from ..substrate import synth_config
            answer = self._llm.generate(prompt, config=synth_config(self._model_name))
        except Exception as e:
            logger.exception("Mem0 synthesis LLM failed")
            answer = f"[Mem0 synthesis error: {e}]"

        return SubstrateResponse(
            answer=answer.strip(),
            cited_moment_ids=cited,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"retrieved": len(rows), "model": self._model_name},
        )
