"""VanillaRAGAdapter — the structure-free dense-retrieval baseline for T1.

Spec §4.6 baseline: "Vanilla RAG — dense retrieval over textified moments
without structural commitment. Implementation: SigLIP embedding + cosine
similarity top-k retrieval."

This is the control that isolates Golgi's *structural* contribution: it uses
the SAME SigLIP text encoder the PWM substrate retrieves with, the SAME
synthesis LLM, and the SAME citation protocol — but it throws away the
knowledge graph, anchors, recurrence, and query understanding. Any lift PWM
shows over Vanilla RAG is attributable to structure, not to a better encoder
or a better answering model.

Retrieval is fully local (SigLIP runs on-device); only the synthesis LLM
call leaves the machine. No external memory provider, no API key beyond the
one the synthesis model needs.
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

# Top-k textified moments to retrieve by cosine similarity. Matched to the
# Mem0 baseline (20) so retrieval breadth is held constant across the
# structure-free baselines.
DEFAULT_TOP_K = 20

# Synthesis model — Gemini Flash, identical to the Mem0 baseline so the
# answering LLM is held constant and only retrieval differs.
DEFAULT_MODEL = "gemini-2.5-flash"

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


class VanillaRAGAdapter:
    """SigLIP + cosine top-k dense-retrieval baseline for PWM-Bench T1."""

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        model: str | None = None,
    ):
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._top_k = top_k
        # PWMBENCH_SYNTH_MODEL pins the answering LLM across all systems so the
        # comparison isolates the memory layer, not the model. Explicit arg wins.
        self._model_name = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or DEFAULT_MODEL
        self._embedder = None  # SigLIPEmbedder
        self._llm = None
        # Parallel arrays: index i → (moment_id, description, Embedding).
        self._ids: list[str] = []
        self._descs: list[str] = []
        self._vecs: list[Embedding] = []
        self._loaded = False

    def name(self) -> str:
        return "vanilla_rag"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ...infrastructure.config import Settings
        from ...infrastructure.embedding import SigLIPEmbedder
        from ...infrastructure.llm import get_synth_service
        from ...interfaces.cli.shared import get_repos

        repos = get_repos(
            persona=self._persona, data_root=self._data_root, read_only=True
        )
        moments = repos.moment.find_all(limit=20000)

        ids: list[str] = []
        descs: list[str] = []
        for m in moments:
            desc = (m.description or "").strip()
            if not desc:
                desc = (m.scene_summary or "").strip()
            if not desc:
                continue
            ids.append(m.id)
            descs.append(desc.replace("\n", " "))

        self._embedder = SigLIPEmbedder()
        logger.info(
            "VanillaRAG embedding %d moment descriptions for persona=%s",
            len(descs), self._persona,
        )
        # SigLIP text embeddings are L2-normalized at source; batch for speed.
        self._vecs = self._embedder.embed_texts_batch(descs)
        self._ids = ids
        self._descs = descs

        settings = Settings.from_env()
        self._llm = get_synth_service(self._model_name, settings)
        self._loaded = True

    def _topk(self, q: str) -> list[int]:
        assert self._embedder is not None
        qvec = self._embedder.embed_text(q)
        scored = [
            (qvec.cosine_similarity(v), i) for i, v in enumerate(self._vecs)
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [i for _, i in scored[: self._top_k]]

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"VanillaRAGAdapter bound to persona={self._persona!r}, got {persona!r}"
            )
        self._ensure_loaded()
        assert self._llm is not None

        t0 = time.perf_counter()
        if not self._vecs:
            return SubstrateResponse(
                answer="No moments in the archive to retrieve from.",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": 0},
            )

        idxs = self._topk(q)
        cited = [self._ids[i] for i in idxs]
        ctx_lines = [
            f"[{rank}] moment_id={self._ids[i]}\n    {self._descs[i]}"
            for rank, i in enumerate(idxs, start=1)
        ]
        prompt = SYNTHESIS_PROMPT.format(
            question=q, n=len(idxs), moments_block="\n".join(ctx_lines)
        )
        try:
            from ..substrate import synth_config
            raw = self._llm.generate(prompt, config=synth_config(self._model_name))
        except Exception as e:
            logger.exception("VanillaRAG synthesis LLM failed")
            return SubstrateResponse(
                answer=f"[VanillaRAG synthesis error: {e}]",
                cited_moment_ids=cited[:3],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": len(idxs)},
            )

        # Keep only citations the model explicitly kept, falling back to the
        # full retrieved set if it emitted no "Cited moments:" line. We trust
        # retrieval order for grounding either way (IDs are all real).
        answer, kept = _split_citations(raw, valid=set(cited))
        return SubstrateResponse(
            answer=answer,
            cited_moment_ids=kept or cited,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"retrieved": len(idxs), "model": self._model_name},
        )


def _split_citations(raw: str, valid: set[str]) -> tuple[str, list[str]]:
    """Split a synthesis response into (answer, cited_ids ⊆ valid)."""
    import re

    cited_re = re.compile(r"Cited moments?:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    id_re = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    )
    cited: list[str] = []
    m = cited_re.search(raw)
    if m:
        for tok in id_re.findall(m.group(1)):
            if tok in valid and tok not in cited:
                cited.append(tok)
        return raw[: m.start()].rstrip(), cited
    for tok in id_re.findall(raw):
        if tok in valid and tok not in cited:
            cited.append(tok)
    return raw.strip(), cited
