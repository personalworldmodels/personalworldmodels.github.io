"""GraphRAGAdapter — Microsoft GraphRAG baseline for PWM-Bench T1.

The structure-committed-at-ingestion graph-RAG competitor reviewers expect. Runs
entirely on Gemini (LiteLLM `model_provider: gemini`) — no OpenAI. At init it
builds (once, cached under data/{persona}/graphrag/) a GraphRAG index over the
686 textified moment captions: per-caption chunks (group_by id=moment_id) so
citations stay at moment granularity. At query time it runs `local_search`
(entity-anchored, the right mode for personal-history QA) and maps the retrieved
source text-units back to moment_ids.

Validated end-to-end with Gemini (scripts/paper/graphrag_validate.py). Indexing
is expensive (~4-6 LLM calls/chunk); the index is cached and re-used.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from ..substrate import SubstrateResponse

logger = logging.getLogger(__name__)

_INSTALL_HINT = "GraphRAGAdapter requires `graphrag` (uv add graphrag) and GEMINI_API_KEY."


class GraphRAGAdapter:
    def __init__(self, persona, data_root=None, model=None):
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._model = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or "gemini-2.5-flash"
        self._key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._root = Path(self._data_root) / self._persona / "graphrag"
        self._cfg = None
        self._dfs = None  # (entities, communities, reports, text_units, relationships)
        self._tu2mid: dict = {}
        self._ready = False

    def name(self) -> str:
        return "graphrag"

    def _build_config(self):
        from graphrag.config.create_graphrag_config import create_graphrag_config
        # Gemini-pinned answer model = self._model so the synthesis LLM is held
        # constant with every other baseline; embeddings on gemini-embedding-001.
        models = {
            "default_chat_model": {"type": "chat", "model_provider": "gemini", "model": self._model,
                                   "api_key": self._key, "concurrent_requests": 4, "model_supports_json": True},
            "default_embedding_model": {"type": "embedding", "model_provider": "gemini",
                                        "model": "gemini-embedding-001", "api_key": self._key,
                                        "concurrent_requests": 4},
        }
        values = {
            "models": models,
            "input": {"file_type": "csv", "base_dir": "input", "text_column": "text"},
            "chunks": {"size": 80, "overlap": 0, "group_by_columns": ["id"]},
            "embed_text": {"model_id": "default_embedding_model"},
            "extract_graph": {"model_id": "default_chat_model"},
        }
        return create_graphrag_config(values=values, root_dir=str(self._root))

    def _prepare_input(self):
        import pandas as pd
        from ...interfaces.cli.shared import get_repos
        (self._root / "input").mkdir(parents=True, exist_ok=True)
        repos = get_repos(persona=self._persona, data_root=self._data_root, read_only=True)
        ids, texts = [], []
        for m in repos.moment.find_all(limit=20000):
            desc = (m.description or "").strip() or (m.scene_summary or "").strip()
            if desc:
                ids.append(m.id)
                texts.append(desc.replace("\n", " "))
        pd.DataFrame({"id": ids, "text": texts}).to_csv(self._root / "input" / "captions.csv", index=False)
        return len(ids)

    def _ensure_ready(self):
        if self._ready:
            return
        try:
            import graphrag.api  # noqa: F401
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e
        import pandas as pd
        out = self._root / "output"
        self._cfg = None
        if not (out / "text_units.parquet").exists():
            n = self._prepare_input()
            self._cfg = self._build_config()
            logger.info("GraphRAG indexing %d captions for %s (cached after)", n, self._persona)
            from graphrag.api import build_index
            asyncio.run(build_index(config=self._cfg))
        if self._cfg is None:
            self._prepare_input()
            self._cfg = self._build_config()

        def load(n):
            p = out / f"{n}.parquet"
            return pd.read_parquet(p) if p.exists() else pd.DataFrame()
        self._dfs = (load("entities"), load("communities"), load("community_reports"),
                     load("text_units"), load("relationships"))
        # map text_unit id -> moment_id via documents (group_by id => 1 caption/unit)
        docs = load("documents")
        tu = self._dfs[3]
        try:
            doc2mid = dict(zip(docs["id"], docs["id"])) if "id" in docs else {}
            for _, row in tu.iterrows():
                dids = row.get("document_ids") or []
                mid = (list(dids)[0] if len(dids) else None) if hasattr(dids, "__len__") else None
                self._tu2mid[row["id"]] = mid or row.get("id")
        except Exception:  # noqa: BLE001
            pass
        self._ready = True

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(f"GraphRAGAdapter bound to {self._persona!r}, got {persona!r}")
        self._ensure_ready()
        t0 = time.perf_counter()
        try:
            from graphrag.api import local_search
            ents, comms, reports, tus, rels = self._dfs
            resp, ctx = asyncio.run(local_search(
                config=self._cfg, entities=ents, communities=comms, community_reports=reports,
                text_units=tus, relationships=rels, covariates=None, community_level=2,
                response_type="concise", query=q))
        except Exception as e:  # noqa: BLE001
            logger.exception("GraphRAG local_search failed")
            return SubstrateResponse(answer=f"[GraphRAG error: {e}]", cited_moment_ids=[],
                                     elapsed_ms=(time.perf_counter() - t0) * 1000)
        cited_mids, cited_texts = [], []
        try:
            src = ctx.get("sources") if hasattr(ctx, "get") else None
            if src is not None and len(src):
                for _, row in src.iterrows():
                    txt = row.get("text") or row.get("content")
                    if txt:
                        cited_texts.append(str(txt)[:300])
                    mid = self._tu2mid.get(row.get("id"))
                    if mid:
                        cited_mids.append(mid)
        except Exception:  # noqa: BLE001
            pass
        return SubstrateResponse(answer=str(resp).strip(), cited_moment_ids=cited_mids,
                                 cited_texts=cited_texts or None,
                                 elapsed_ms=(time.perf_counter() - t0) * 1000,
                                 debug={"model": self._model})
