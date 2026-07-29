"""VanillaStrongAdapter — the FAIR flat-retrieval control for T1.

The plain VanillaRAG baseline retrieves on the raw question with no query
understanding, while PWM gets an LLM query-understander, top_k=30, and an
empty-retry. That conflates the bicameral GRAPH with generic LLM query
scaffolding. This adapter equalizes the scaffolding: it gives a flat SigLIP
cosine retriever the SAME budget (top_k=30, same synthesis model/protocol) AND
a generic one-shot LLM query-rewrite — so the ONLY thing it lacks vs PWM is the
knowledge graph (anchors, entity/FTS fusion). PWM's lift over THIS control is
attributable to structure, not to query understanding or recall breadth.

This is the apples-to-apples control the review panel flagged as must-have.
"""
from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ..substrate import SubstrateResponse

if TYPE_CHECKING:
    from ...domain.shared.value_objects.embedding import Embedding

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 30   # matched to PWM
DEFAULT_MODEL = "gemini-2.5-flash"

_REWRITE_PROMPT = (
    "Rewrite this question about a personal photo archive into a concise search "
    "query: keep the people, places, activities, objects, and time references; "
    "drop filler ('show me', 'find', 'my'). Return ONLY the rewritten query.\n\n"
    "Question: {q}\nSearch query:"
)

SYNTHESIS_PROMPT = """You are answering a question about a person's personal photo memory archive.

Question: {question}

Below are the {n} moments retrieved by dense similarity search. Each is tagged
with its moment_id.

Moments:
{moments_block}

Instructions:
1. Answer the question using ONLY information present in the moments above.
2. If the moments do not contain enough information to answer, say so plainly.
3. After your answer, on a new line, write: "Cited moments: <comma-separated moment_ids you actually used>".

Answer:"""


class VanillaStrongAdapter:
    """Flat SigLIP retrieval + LLM query-rewrite + matched budget (fair control)."""

    def __init__(self, persona, data_root=None, top_k=DEFAULT_TOP_K, model=None):
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._top_k = top_k
        self._model_name = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or DEFAULT_MODEL
        self._embedder = None
        self._llm = None
        self._ids: list[str] = []
        self._descs: list[str] = []
        self._vecs: list[Embedding] = []
        self._loaded = False

    def name(self) -> str:
        return "vanilla_strong"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ...infrastructure.config import Settings
        from ...infrastructure.embedding import SigLIPEmbedder
        from ...infrastructure.llm import get_synth_service
        from ...interfaces.cli.shared import get_repos

        repos = get_repos(persona=self._persona, data_root=self._data_root, read_only=True)
        ids, descs = [], []
        for m in repos.moment.find_all(limit=20000):
            desc = (m.description or "").strip() or (m.scene_summary or "").strip()
            if desc:
                ids.append(m.id)
                descs.append(desc.replace("\n", " "))
        self._embedder = SigLIPEmbedder()
        self._vecs = self._embedder.embed_texts_batch(descs)
        self._ids, self._descs = ids, descs
        self._llm = get_synth_service(self._model_name, Settings.from_env())
        self._loaded = True

    def _rewrite(self, q: str) -> str:
        try:
            from ..substrate import synth_config
            out = self._llm.generate(_REWRITE_PROMPT.format(q=q), config=synth_config(self._model_name))
            r = (out or "").strip().splitlines()[0].strip() if out else ""
            return f"{q} {r}".strip() if r else q
        except Exception:  # noqa: BLE001
            return q

    def _topk(self, text: str) -> list[int]:
        assert self._embedder is not None
        qv = self._embedder.embed_text(text)
        scored = sorted(((qv.cosine_similarity(v), i) for i, v in enumerate(self._vecs)),
                        key=lambda t: t[0], reverse=True)
        return [i for _, i in scored[: self._top_k]]

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(f"VanillaStrongAdapter bound to {self._persona!r}, got {persona!r}")
        self._ensure_loaded()
        assert self._llm is not None
        t0 = time.perf_counter()
        if not self._vecs:
            return SubstrateResponse(answer="No moments in the archive.", cited_moment_ids=[],
                                     elapsed_ms=(time.perf_counter() - t0) * 1000, debug={"retrieved": 0})
        idxs = self._topk(self._rewrite(q))
        cited = [self._ids[i] for i in idxs]
        block = "\n".join(f"[{r}] moment_id={self._ids[i]}\n    {self._descs[i]}" for r, i in enumerate(idxs, 1))
        try:
            from ..substrate import synth_config
            raw = self._llm.generate(SYNTHESIS_PROMPT.format(question=q, n=len(idxs), moments_block=block),
                                     config=synth_config(self._model_name))
        except Exception as e:  # noqa: BLE001
            logger.exception("VanillaStrong synthesis failed")
            return SubstrateResponse(answer=f"[VanillaStrong synthesis error: {e}]", cited_moment_ids=cited[:3],
                                     elapsed_ms=(time.perf_counter() - t0) * 1000, debug={"retrieved": len(idxs)})
        from .vanilla_rag import _split_citations
        answer, kept = _split_citations(raw, valid=set(cited))
        return SubstrateResponse(answer=answer, cited_moment_ids=kept or cited,
                                 elapsed_ms=(time.perf_counter() - t0) * 1000,
                                 debug={"retrieved": len(idxs), "model": self._model_name})
