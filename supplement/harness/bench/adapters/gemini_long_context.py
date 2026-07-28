"""GeminiLongContextAdapter — Gemini 2.5 Pro long-context baseline for T1.

This adapter dumps EVERY moment description in the persona's archive into
a single prompt prefix and asks Gemini 2.5 Pro to answer the question.
It is the canonical "no substrate, just a big context window" baseline:
it tests whether the substrate's structural commitment beats brute-force
long-context recall.

Each moment is prefixed with its moment_id so the model can cite specific
moments. We parse citations the same way as the PWM adapter (Cited moments:
trailing line + UUID regex fallback).

Cost note: at 686 moments × ~200 chars description ≈ 140K tokens of
context. Within Gemini 2.5 Pro's 1M context window but not free; expect
each query to cost a few cents.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING

from ..substrate import SubstrateResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Long-context prompts can be cached across queries on the same persona.
# Cache key: (persona, data_root, model). Value: the rendered moments
# block (str). Built once per adapter instance.
_DEFAULT_MODEL = "gemini-2.5-pro"

_CITED_RE = re.compile(r"Cited moments?:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class GeminiLongContextAdapter:
    """Gemini 2.5 Pro long-context baseline for PWM-Bench T1."""

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_moments: int | None = None,
    ):
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._model_name = model
        self._max_moments = max_moments
        self._moments_block: str | None = None
        self._valid_ids: set[str] = set()
        self._llm = None

    def name(self) -> str:
        return "gemini_1m"

    def _ensure_loaded(self) -> None:
        if self._moments_block is not None and self._llm is not None:
            return
        from ...infrastructure.config import Settings
        from ...infrastructure.gemini_client import create_gemini_client
        from ...interfaces.cli.shared import get_repos

        repos = get_repos(
            persona=self._persona, data_root=self._data_root, read_only=True
        )
        moments = repos.moment.find_all(limit=20000)
        if self._max_moments:
            moments = moments[: self._max_moments]

        # Sort by timestamp if available — gives the long-context model a
        # natural temporal ordering to reason over. Lookup timestamps via
        # the graph store's date index.
        graph = repos.graph
        moment_ids = [m.id for m in moments]
        try:
            ts_map = graph.get_dates_by_moments(moment_ids) if hasattr(graph, "get_dates_by_moments") else {}
        except Exception:
            ts_map = {}

        def _sort_key(m):
            ts = ts_map.get(m.id)
            return (ts is None, str(ts) if ts else "")

        moments_sorted = sorted(moments, key=_sort_key)

        lines: list[str] = []
        for m in moments_sorted:
            desc = (m.description or "").strip().replace("\n", " ")
            if not desc:
                desc = (m.scene_summary or "").strip().replace("\n", " ")
            if not desc:
                continue
            ts = ts_map.get(m.id, "unknown")
            lines.append(f"[{m.id}] ({ts}) {desc}")
            self._valid_ids.add(m.id)
        self._moments_block = "\n".join(lines)
        logger.info(
            "GeminiLongContext loaded %d moment descriptions (~%d chars) for persona=%s",
            len(lines), len(self._moments_block), self._persona,
        )

        # Use the raw google-genai client directly rather than
        # GeminiLLMService — the latter hardcodes thinking_budget=0,
        # which `gemini-2.5-pro` rejects (it requires thinking mode).
        settings = Settings.from_env()
        self._llm = create_gemini_client(settings)

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"GeminiLongContextAdapter bound to persona={self._persona!r}, got {persona!r}"
            )
        self._ensure_loaded()
        assert self._moments_block is not None and self._llm is not None

        prompt = (
            f"You have access to the full archive of {persona}'s personal "
            f"photo moments below. Each line is a single moment, prefixed "
            f"with its moment_id in square brackets and a timestamp.\n\n"
            f"=== ARCHIVE ({len(self._valid_ids)} moments) ===\n"
            f"{self._moments_block}\n"
            f"=== END ARCHIVE ===\n\n"
            f"Question: {q}\n\n"
            "Answer the question using ONLY information present in the "
            "archive above. After your answer, on a new line, write: "
            "'Cited moments: <comma-separated moment_ids you actually used>'."
        )

        t0 = time.perf_counter()
        try:
            response = self._llm.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            raw = response.text or ""
        except Exception as e:
            logger.exception("Gemini long-context generate failed")
            return SubstrateResponse(
                answer=f"[Gemini long-context error: {e}]",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        cited: list[str] = []
        m = _CITED_RE.search(raw)
        if m:
            for tok in _ID_RE.findall(m.group(1)):
                if tok in self._valid_ids and tok not in cited:
                    cited.append(tok)
            answer = raw[: m.start()].rstrip()
        else:
            answer = raw.strip()
            for tok in _ID_RE.findall(raw):
                if tok in self._valid_ids and tok not in cited:
                    cited.append(tok)

        return SubstrateResponse(
            answer=answer,
            cited_moment_ids=cited,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"corpus_moments": len(self._valid_ids), "model": self._model_name},
        )
