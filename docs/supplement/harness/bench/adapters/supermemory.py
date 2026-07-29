"""SupermemoryAdapter — Supermemory baseline for PWM-Bench T1.

Supermemory (https://supermemory.ai) is the hosted memory layer behind the
MemoryBench framework; it uses chunk-based semantic search over ingested
content. We include it as an extra baseline alongside the spec §4.6 locked
set (Mem0, Zep, Vanilla RAG) because it is one of the systems the PWM paper
references and the provider whose benchmark methodology (MemoryBench) we align
our harness with.

Ingestion: each moment description is added as a memory under a per-persona
`container_tag`, with `metadata.moment_id` set so retrieved memories map back
to a source moment for the source-grounding metric — exactly the mechanism the
Mem0 adapter uses.

Query: semantic search scoped to the persona's container, then a synthesis LLM
call over the retrieved chunks — same answering model and citation protocol as
the other text-store baselines.

Requires the `supermemory` SDK and a `SUPERMEMORY_API_KEY`. Neither is a
default project dependency; both gate at construction with an install/config
hint.
"""
from __future__ import annotations

import logging
import os
import time

from ..substrate import SubstrateResponse

logger = logging.getLogger(__name__)

_SM_INSTALL_HINT = (
    "SupermemoryAdapter requires the `supermemory` SDK and a "
    "SUPERMEMORY_API_KEY. Install with `uv add supermemory` (or "
    "`uv pip install supermemory`) and set SUPERMEMORY_API_KEY in your "
    "environment (get one at https://supermemory.ai/)."
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 20


class SupermemoryAdapter:
    """Supermemory chunk-based semantic-search baseline for PWM-Bench T1."""

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        model: str | None = None,
    ):
        # PWMBENCH_SYNTH_MODEL pins the answering LLM across systems; arg wins.
        model = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or DEFAULT_MODEL
        api_key = os.environ.get("SUPERMEMORY_API_KEY")
        if not api_key:
            raise RuntimeError(_SM_INSTALL_HINT)
        try:
            from supermemory import Supermemory  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(_SM_INSTALL_HINT) from e

        self._client = Supermemory(api_key=api_key)
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._top_k = top_k
        self._model_name = model
        self._container = f"pwmbench_{persona}"
        self._llm = None
        self._ingested = False

    def name(self) -> str:
        return "supermemory"

    def _ensure_ingested(self) -> None:
        if self._ingested:
            return
        # Idempotency: if this container already holds a processed corpus from a
        # prior run, skip re-ingestion (re-adding would duplicate documents).
        existing_done = self._count_done()
        if existing_done >= 600:
            logger.info("Supermemory container already has %d done docs; skipping ingest", existing_done)
            self._ingested = True
            self._setup_llm()
            return

        from ...interfaces.cli.shared import get_repos
        repos = get_repos(
            persona=self._persona, data_root=self._data_root, read_only=True
        )
        moments = repos.moment.find_all(limit=20000)
        logger.info(
            "Supermemory ingesting %d moments for persona=%s (container=%s)",
            len(moments), self._persona, self._container,
        )
        count = 0
        for m in moments:
            desc = (m.description or "").strip() or (m.scene_summary or "").strip()
            if not desc:
                continue
            try:
                # v3 SDK: top-level client.add (NOT memories.add, which is
                # forget/update only). container_tag (singular) is correct here.
                self._client.add(
                    content=desc,
                    container_tag=self._container,
                    metadata={"moment_id": m.id},
                )
                count += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("supermemory add failed for moment %s: %s", m.id, e)
        logger.info("Supermemory submitted %d/%d documents", count, len(moments))

        # Supermemory processes documents asynchronously; wait until most are
        # 'done' so queries hit a populated index (else search returns empty
        # and the system scores unfairly low).
        self._wait_for_processing(expected=count)
        self._ingested = True
        self._setup_llm()

    def _setup_llm(self) -> None:
        from ...infrastructure.config import Settings
        from ...infrastructure.llm import get_synth_service
        settings = Settings.from_env()
        self._llm = get_synth_service(self._model_name, settings)

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"SupermemoryAdapter bound to persona={self._persona!r}, got {persona!r}"
            )
        self._ensure_ingested()
        assert self._llm is not None

        t0 = time.perf_counter()
        try:
            # container_tags (PLURAL list) — the singular form silently returns
            # zero results in v3.47.
            results = self._client.search.execute(
                q=q,
                container_tags=[self._container],
                limit=self._top_k,
            )
        except Exception as e:
            logger.exception("Supermemory search failed")
            return SubstrateResponse(
                answer=f"[Supermemory search error: {e}]",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        chunks = _extract_chunks(results)
        if not chunks:
            return SubstrateResponse(
                answer="Supermemory found no relevant memories for this query.",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": 0},
            )

        cited: list[str] = []
        cited_texts: list[str] = []
        ctx_lines: list[str] = []
        for i, (text, mid) in enumerate(chunks, start=1):
            if mid and mid not in cited:
                cited.append(mid)
            cited_texts.append(text)
            ctx_lines.append(f"[{i}] {text}")

        prompt = (
            f"Question: {q}\n\n"
            f"Relevant memories from Supermemory:\n{chr(10).join(ctx_lines)}\n\n"
            "Answer the question using only these memories. If they don't "
            "contain enough information, say so plainly."
        )
        try:
            from ..substrate import synth_config
            answer = self._llm.generate(prompt, config=synth_config(self._model_name))
        except Exception as e:
            logger.exception("Supermemory synthesis LLM failed")
            answer = f"[Supermemory synthesis error: {e}]"

        return SubstrateResponse(
            answer=answer.strip(),
            cited_moment_ids=cited,
            cited_texts=cited_texts,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"retrieved": len(chunks), "model": self._model_name},
        )

    def _wait_for_processing(
        self, expected: int, timeout_s: float = 900.0, poll_s: float = 15.0
    ) -> None:
        """Poll until most documents in the container report 'done'.

        Supermemory indexes asynchronously; querying before indexing finishes
        returns empty results. We wait until ~95% of submitted docs are done
        (or the count stalls / times out), so the benchmark hits a populated
        index. Bounded so a stuck doc can't hang the run.
        """
        if expected <= 0:
            return
        target = max(1, int(expected * 0.95))
        deadline = time.perf_counter() + timeout_s
        last = -1
        stable = 0
        while time.perf_counter() < deadline:
            done = self._count_done()
            logger.info("Supermemory indexing: %d/%d done", done, expected)
            if done >= target:
                return
            stable = stable + 1 if done == last else 0
            if stable >= 6 and done > 0:
                logger.warning("Supermemory indexing stalled at %d/%d; proceeding", done, expected)
                return
            last = done
            time.sleep(poll_s)
        logger.warning("Supermemory indexing wait timed out at %d/%d", last, expected)

    def _count_done(self) -> int:
        """Count 'done' documents in this container across pagination."""
        done = 0
        page = 1
        try:
            while True:
                resp = self._client.documents.list(
                    container_tags=[self._container], page=page, limit=200
                )
                d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                docs = d.get("memories") or d.get("documents") or d.get("results") or []
                if not docs:
                    break
                done += sum(1 for x in docs if str(x.get("status")) == "done")
                pg = d.get("pagination") or {}
                total_pages = pg.get("totalPages") or pg.get("total_pages") or 1
                if page >= total_pages:
                    break
                page += 1
        except Exception as e:  # noqa: BLE001
            logger.debug("supermemory _count_done failed: %s", e)
        return done


def _extract_chunks(results) -> list[tuple[str, str | None]]:
    """Normalize Supermemory v3 search results to [(text, moment_id?), ...].

    Each result row carries: title, chunks[].content (the matched text;
    `content`/`summary` come back null), and metadata.moment_id we set at
    ingest. We build evidence text from the title + chunk contents.
    """
    d = results.model_dump() if hasattr(results, "model_dump") else results
    rows = d.get("results") if isinstance(d, dict) else None
    out: list[tuple[str, str | None]] = []
    for r in rows or []:
        title = r.get("title") or ""
        chunk_texts = [
            c.get("content", "") for c in (r.get("chunks") or []) if c.get("content")
        ]
        text = " ".join(filter(None, [title, *chunk_texts])).strip()
        if not text:
            text = r.get("content") or r.get("summary") or ""
        meta = r.get("metadata") or {}
        mid = meta.get("moment_id") if isinstance(meta, dict) else None
        if text:
            out.append((str(text), mid))
    return out
