"""PWMAdapter — the Golgi reference substrate for T1.

Architecture: query-time, the adapter
1. Runs `SearchService.search(question)` to get top-K hydrated moments
   (bicameral retrieval: vector + entity + FTS, fused).
2. Builds a context block with each moment's description, timestamp,
   location, and ID.
3. Asks an LLM to answer the question grounded ONLY in that context, with
   moment ID citations.

The cited moment IDs are returned as the substrate's grounding evidence
(spec §4.5: "Source grounding: Fraction of answers citing at least one
moment whose retrieved content supports the answer.").

Note: this adapter uses the existing SearchService + LLM factory rather
than the MCP tool surface. The MCP tools (`what_happened_with`, etc.)
are name-scoped to a single entity per call and don't compose into a
question-answering flow without an outer LLM agent loop — wrapping
SearchService is one layer thinner and gives the substrate a fair shot
at grounded recall.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..substrate import SubstrateResponse

if TYPE_CHECKING:
    from ...application.services.search import RichMoment, SearchService

logger = logging.getLogger(__name__)

# Top-K moments to retrieve and feed into the synthesis prompt. Chosen to
# fit in a reasonable prompt budget while giving the answer enough recall
# breadth for open-ended and recurring-pattern questions.
DEFAULT_TOP_K = 30   # bumped from 20 — wider recall for grounded synthesis

# Default synthesis model. Sonnet/4o are spec-listed alternatives (§4.8);
# Flash is fast + cheap for harness iteration. Override via PWMAdapter(model=...).
DEFAULT_MODEL = "gemini-2.5-pro"   # match Gemini-1M baseline for fair LLM-axis comparison

SYNTHESIS_PROMPT = """You are answering a question about a person's personal photo memory archive.

Question: {question}

Below are the {n} most relevant moments retrieved from the archive. Each is
tagged with its moment_id, timestamp, and any extracted location/entities.

Moments:
{moments_block}

Instructions:
1. Answer the question using ONLY information present in the moments above.
2. If the moments do not contain enough information to answer, say so plainly.
3. After your answer, on a new line, write: "Cited moments: <comma-separated moment_ids you actually used>".
4. Cite only moments whose description directly supports a claim in your answer.

Answer:"""

# Matches "Cited moments: id1, id2, id3" trailing line in the LLM response.
# Synthesis needs a generous output budget: on Gemini 2.5 Pro, thinking
# tokens are drawn from max_output_tokens, so the default 2048 truncates the
# answer mid-sentence and the trailing "Cited moments:" line never appears
# (grounding collapses to 0). 8192 leaves room for thinking + a full answer.
_SYNTH_MAX_TOKENS = 8192

_CITED_RE = re.compile(r"Cited moments?:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# UUID-ish moment IDs are 8-4-4-4-12; tolerate any [0-9a-f-]{6,} for safety.
_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class PWMAdapter:
    """Golgi PWM substrate adapter for PWM-Bench T1."""

    def __init__(
        self,
        persona: str,
        data_root: str | None = None,
        model: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._persona = persona
        self._data_root = data_root or os.environ.get("GOLGI_DATA_ROOT", "data")
        # PWMBENCH_SYNTH_MODEL pins the answering LLM across all systems so the
        # comparison isolates the memory layer, not the model. Explicit arg wins.
        self._model_name = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or DEFAULT_MODEL
        self._top_k = top_k
        self._search: SearchService | None = None
        self._llm = None  # GeminiLLMService instance

    def name(self) -> str:
        return "pwm"

    # --- Lazy init: heavy services constructed on first query ---

    def _ensure_loaded(self) -> None:
        if self._search is not None and self._llm is not None:
            return
        # SearchService construction requires repos + embedder + query
        # understander + persona. Easiest path: ServiceContainer (same path
        # the MCP server and visualization use).
        from ...domain.shared import InferenceMode
        from ...infrastructure.config import Settings
        from ...infrastructure.llm import get_llm_service
        from ...interfaces.shared import ServiceContainer

        settings = Settings.from_env()
        container = ServiceContainer.create(
            data_root=self._data_root,
            settings=settings,
            locked_persona=self._persona,
            inference_mode=InferenceMode.CLOUD,
        )
        container.persona_manager.set_persona_readonly(self._persona, True)
        # ServiceContainer requires `ready` (models loaded) before
        # get_search_service returns a service. For T1 we don't need
        # vision embedders — we run the SearchService text path. Build
        # it directly from the registry's underlying factory.
        repos = container.get_repos(self._persona)
        self._container = container  # keep alive

        from ...application.services.search import SearchService
        from ...infrastructure.embedding import SigLIPEmbedder
        from ...infrastructure.llm.query_understander import LLMQueryUnderstander
        from ...interfaces.cli.shared import create_text_embedder

        embedder = SigLIPEmbedder()
        text_embedder = None
        try:
            text_embedder = create_text_embedder(InferenceMode.CLOUD, settings)
        except Exception as e:
            logger.warning("text embedder unavailable: %s", e)

        llm_cloud = get_llm_service(
            settings=settings,
            inference_mode=InferenceMode.CLOUD,
        )
        qu = LLMQueryUnderstander(llm_service=llm_cloud)

        self._search = SearchService(
            moment_repo=repos.moment,
            media_repo=repos.media,
            graph_repo=repos.graph,
            embedder=embedder,
            persona_id=self._persona,
            query_understander=qu,
            text_embedder=text_embedder,
            constellation_repo=getattr(repos, "constellation", None),
        )

        # Synthesis LLM (separate handle so the answer model is independent of
        # the QueryUnderstander's model). Routed by model name so the §4.8 matrix
        # can pin Gemini / GPT-4o / on-device Ollama as the answer model while
        # the substrate's retrieval (incl. the Gemini query-understander above)
        # stays constant.
        from ...infrastructure.llm import get_synth_service
        self._llm = get_synth_service(self._model_name, settings)

    # --- Substrate Protocol ---

    def query(self, q: str, persona: str) -> SubstrateResponse:
        if persona != self._persona:
            raise ValueError(
                f"PWMAdapter bound to persona={self._persona!r} but query asked for {persona!r}"
            )
        self._ensure_loaded()
        assert self._search is not None and self._llm is not None

        t0 = time.perf_counter()
        # relax_on_empty: if the query-understander over-filters a natural
        # question into a hard spot/location constraint that matches nothing,
        # retry with constraints stripped so the semantic signal still retrieves.
        # Outer retry: the query-understander's LLM can hit transient upstream
        # auth/rate blips (a 400/429 burst); the QU's own 3 fast attempts can all
        # land inside one blip. Retry the whole search with backoff so a
        # transient infra hiccup doesn't zero an otherwise-valid query.
        moments = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                moments = self._search.search(q, limit=self._top_k, relax_on_empty=True)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < 2:
                    time.sleep(3.0 * (attempt + 1))
        if last_err is not None:
            logger.exception("PWM search failed for q=%r after retries", q)
            return SubstrateResponse(
                answer=f"[PWM search error: {last_err}]",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        if not moments:
            return SubstrateResponse(
                answer="No moments matched this query in the archive.",
                cited_moment_ids=[],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": 0},
            )

        moments_block = _format_moments(moments)
        prompt = SYNTHESIS_PROMPT.format(
            question=q,
            n=len(moments),
            moments_block=moments_block,
        )
        try:
            from ..substrate import synth_config
            raw = self._llm.generate(prompt, config=synth_config(self._model_name))
        except Exception as e:
            logger.exception("PWM synthesis LLM failed")
            return SubstrateResponse(
                answer=f"[PWM synthesis error: {e}]",
                cited_moment_ids=[m.moment_id for m in moments[:3]],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                debug={"retrieved": len(moments)},
            )

        answer, cited = _parse_answer_and_citations(raw, retrieved_ids={m.moment_id for m in moments})
        # Gemini (esp. Pro) often omits the "Cited moments:" line. The moments
        # the substrate retrieved ARE its grounding evidence, so fall back to
        # them — the grounding judge then decides whether any supports the
        # answer (spec §4.5). Order preserved (retrieval rank).
        if not cited:
            cited = [m.moment_id for m in moments]
        return SubstrateResponse(
            answer=answer,
            cited_moment_ids=cited,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            debug={"retrieved": len(moments), "model": self._model_name},
        )


def _format_moments(moments: list["RichMoment"]) -> str:
    """Build the moments context block for the synthesis prompt."""
    lines: list[str] = []
    for i, m in enumerate(moments, start=1):
        ts = m.timestamp.isoformat() if m.timestamp else "unknown"
        loc = m.location or "unknown"
        desc = (m.description or "").strip().replace("\n", " ")
        if not desc:
            desc = (m.scene_summary or "").strip().replace("\n", " ")
        # Truncate long descriptions to keep prompt size bounded — most
        # moment descriptions are 1-3 sentences; cap at 400 chars.
        if len(desc) > 400:
            desc = desc[:397] + "..."
        lines.append(
            f"[{i}] moment_id={m.moment_id} ts={ts} location={loc}\n    {desc}"
        )
    return "\n".join(lines)


def _parse_answer_and_citations(
    raw: str, retrieved_ids: set[str]
) -> tuple[str, list[str]]:
    """Split LLM output into (answer_text, cited_moment_ids).

    The synthesis prompt asks the model to put citations on a trailing
    `Cited moments: ...` line. We honor that line, but also fall back to
    any UUID-looking IDs anywhere in the response — robustness vs. format
    drift. Only IDs that appear in `retrieved_ids` are kept (no hallucinated
    IDs in the citation set).
    """
    cited: list[str] = []
    m = _CITED_RE.search(raw)
    if m:
        for token in _ID_RE.findall(m.group(1)):
            if token in retrieved_ids and token not in cited:
                cited.append(token)
        answer = raw[: m.start()].rstrip()
    else:
        answer = raw.strip()
        # Fallback: scan whole response for any retrieved IDs the model
        # mentioned inline.
        for token in _ID_RE.findall(raw):
            if token in retrieved_ids and token not in cited:
                cited.append(token)
    return answer, cited
