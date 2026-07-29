"""ZepAdapter — Zep temporal knowledge-graph baseline for PWM-Bench T1.

Zep (Rasmussen et al., arXiv:2501.13956) builds a temporal knowledge graph
(Graphiti) over ingested text and answers via graph-grounded retrieval. It is
spec §4.6's second locked baseline and the strongest publicly-defended LoCoMo
system as of mid-2026 — the natural "does a temporal KG beat our bicameral
substrate?" comparison.

Ingestion: each moment description is added to the persona's graph via the
Zep `graph.add` text API. We prefix every chunk with a `[moment_id=...]` token
so that, when Zep's extracted facts retain it, we can map a retrieved fact back
to its source moment for the source-grounding metric. Zep re-extracts facts
with its own LLM and may drop the token; when it does, citations come back
empty for that fact. This is the same native-citation limitation noted for the
Mem0 baseline and is reported honestly rather than worked around.

Query: `graph.search` over edges (facts) for the persona, then a synthesis LLM
call over the retrieved facts — same answering model and citation protocol as
the other text-store baselines so only the memory layer differs.

Requires the `zep-cloud` SDK and a `ZEP_API_KEY`. Neither is a default project
dependency; both gate at construction with an install/config hint.
"""
from __future__ import annotations

import logging
import os
import re
import time

from ..substrate import SubstrateResponse

logger = logging.getLogger(__name__)

_ZEP_INSTALL_HINT = (
    "ZepAdapter requires the `zep-cloud` SDK and a ZEP_API_KEY. Install with "
    "`uv add zep-cloud` (or `uv pip install zep-cloud`) and set ZEP_API_KEY in "
    "your environment (get one at https://www.getzep.com/)."
)

# moment_id token we embed in each ingested chunk so retrieved facts can be
# mapped back to a source moment when Zep preserves it.
_MID_TOKEN = re.compile(r"moment_id=([0-9a-f-]{6,})")
_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 20


class ZepAdapter:
    """Zep temporal knowledge-graph baseline for PWM-Bench T1."""

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        model: str | None = None,
    ):
        # PWMBENCH_SYNTH_MODEL pins the answering LLM across systems; arg wins.
        model = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or DEFAULT_MODEL
        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError(_ZEP_INSTALL_HINT)
        try:
            from zep_cloud.client import Zep  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(_ZEP_INSTALL_HINT) from e

        self._client = Zep(api_key=api_key)
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        self._top_k = top_k
        self._model_name = model
        # Zep graph ids must be conservative; namespace per persona.
        self._graph_user = f"pwmbench_{persona}"
        self._llm = None
        self._ingested = False

    def name(self) -> str:
        return "zep"

    def _ensure_ingested(self) -> None:
        if self._ingested:
            return
        # Create the user/graph owner (idempotent — ignore "already exists").
        try:
            self._client.user.add(user_id=self._graph_user)
        except Exception as e:  # noqa: BLE001 — Zep raises on duplicate
            logger.debug("zep user.add (likely exists): %s", e)

        # Idempotency: if this graph already holds a processed corpus from a
        # prior run, skip re-ingestion (it would only add duplicate episodes
        # and waste 15+ min). Threshold guards against a partial earlier run.
        try:
            existing = self._client.graph.episode.get_by_user_id(
                self._graph_user, lastn=2000
            )
            eps = getattr(existing, "episodes", None) or []
            done = sum(1 for e in eps if getattr(e, "processed", False))
            if done >= 600:
                logger.info("Zep graph already has %d processed episodes; skipping ingest", done)
                self._ingested = True
                self._setup_llm()
                return
        except Exception as e:  # noqa: BLE001
            logger.debug("zep existing-episode check failed: %s", e)

        from ...interfaces.cli.shared import get_repos
        repos = get_repos(
            persona=self._persona, data_root=self._data_root, read_only=True
        )
        moments = repos.moment.find_all(limit=20000)
        logger.info(
            "Zep ingesting %d moments for persona=%s (graph=%s)",
            len(moments), self._persona, self._graph_user,
        )
        # Build episode payloads (moment_id token embedded for citation recovery).
        from zep_cloud.types.episode_data import EpisodeData
        episodes = [
            EpisodeData(
                data=f"[moment_id={m.id}] {desc}".replace("\n", " "), type="text"
            )
            for m in moments
            if (desc := ((m.description or "").strip() or (m.scene_summary or "").strip()))
        ]

        # Batch ingest (chunks) rather than 686 sequential calls — far faster
        # and avoids free-tier rate limits. Fall back to per-episode on a
        # chunk failure so one bad payload doesn't drop the whole batch.
        count = 0
        BATCH = 20
        for i in range(0, len(episodes), BATCH):
            chunk = episodes[i : i + BATCH]
            try:
                self._client.graph.add_batch(
                    user_id=self._graph_user, episodes=chunk
                )
                count += len(chunk)
            except Exception as e:  # noqa: BLE001
                logger.warning("zep add_batch chunk @%d failed (%s); per-episode", i, e)
                for ep in chunk:
                    try:
                        self._client.graph.add(
                            user_id=self._graph_user, type=ep.type, data=ep.data
                        )
                        count += 1
                    except Exception as e2:  # noqa: BLE001
                        logger.debug("zep graph.add failed: %s", e2)
        logger.info("Zep submitted %d/%d episodes", count, len(moments))

        # Zep processes episodes into the temporal graph ASYNCHRONOUSLY. Querying
        # before processing finishes returns empty/partial facts and unfairly
        # tanks Zep's score, so wait until episodes report processed (bounded).
        self._wait_for_processing(expected=count)
        self._ingested = True
        self._setup_llm()

    def _setup_llm(self) -> None:
        from ...infrastructure.config import Settings
        from ...infrastructure.llm import get_synth_service
        settings = Settings.from_env()
        self._llm = get_synth_service(self._model_name, settings)

    def _wait_for_processing(
        self, expected: int, timeout_s: float = 900.0, poll_s: float = 15.0
    ) -> None:
        """Poll until ingested episodes report processed, or timeout.

        Zep builds the graph asynchronously; `episode.processed` flips True when
        a moment has been extracted. We wait for the processed count to reach
        `expected` (or stop growing) so the benchmark queries a fully-built
        graph. Bounded so a stuck episode can't hang the run forever.
        """
        if expected <= 0:
            return
        deadline = time.perf_counter() + timeout_s
        last_done = -1
        stable_rounds = 0
        while time.perf_counter() < deadline:
            try:
                resp = self._client.graph.episode.get_by_user_id(
                    self._graph_user, lastn=expected + 50
                )
                eps = getattr(resp, "episodes", None) or []
                done = sum(1 for e in eps if getattr(e, "processed", False))
            except Exception as e:  # noqa: BLE001
                logger.debug("zep processing poll failed: %s", e)
                done = last_done
            logger.info("Zep processing: %d/%d episodes done", done, expected)
            if done >= expected:
                return
            # Guard against episodes that never flip (e.g. dropped server-side):
            # if the count stops advancing for several rounds, proceed anyway.
            stable_rounds = stable_rounds + 1 if done == last_done else 0
            if stable_rounds >= 6 and done > 0:
                logger.warning(
                    "Zep processing stalled at %d/%d; proceeding", done, expected
                )
                return
            last_done = done
            time.sleep(poll_s)
        logger.warning("Zep processing wait timed out at %d/%d", last_done, expected)

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"ZepAdapter bound to persona={self._persona!r}, got {persona!r}"
            )
        self._ensure_ingested()
        assert self._llm is not None

        t0 = time.perf_counter()
        try:
            results = self._client.graph.search(
                user_id=self._graph_user, query=q, limit=self._top_k, scope="edges"
            )
        except Exception as e:
            logger.exception("Zep graph.search failed")
            return SubstrateResponse(
                answer=f"[Zep search error: {e}]",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        facts = _extract_facts(results)
        if not facts:
            return SubstrateResponse(
                answer="Zep found no relevant facts for this query.",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": 0},
            )

        cited: list[str] = []
        ctx_lines: list[str] = []
        for i, fact in enumerate(facts, start=1):
            for mid in _MID_TOKEN.findall(fact):
                if mid not in cited:
                    cited.append(mid)
            ctx_lines.append(f"[{i}] {fact}")

        prompt = (
            f"Question: {q}\n\n"
            f"Relevant facts from Zep's temporal knowledge graph:\n"
            f"{chr(10).join(ctx_lines)}\n\n"
            "Answer the question using only these facts. If they don't contain "
            "enough information, say so plainly."
        )
        try:
            from ..substrate import synth_config
            answer = self._llm.generate(prompt, config=synth_config(self._model_name))
        except Exception as e:
            logger.exception("Zep synthesis LLM failed")
            answer = f"[Zep synthesis error: {e}]"

        # Keep only well-formed UUIDs (the token regex is permissive).
        cited = [c for c in cited if _ID_RE.fullmatch(c)]
        # Zep's fact extraction usually strips our moment_id token, so cited IDs
        # are often empty. Hand the grounding judge Zep's retrieved FACTS as
        # evidence text instead — that's the provenance Zep actually exposes.
        return SubstrateResponse(
            answer=answer.strip(),
            cited_moment_ids=cited,
            cited_texts=facts,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"retrieved": len(facts), "model": self._model_name},
        )


def _extract_facts(results) -> list[str]:
    """Normalize Zep graph.search results to a list of fact strings.

    Zep SDK versions return either an object with `.edges` (each with a
    `.fact`) or a dict; be tolerant of both shapes.
    """
    edges = getattr(results, "edges", None)
    if edges is None and isinstance(results, dict):
        edges = results.get("edges")
    facts: list[str] = []
    for e in edges or []:
        fact = getattr(e, "fact", None)
        if fact is None and isinstance(e, dict):
            fact = e.get("fact")
        if fact:
            facts.append(str(fact))
    return facts
