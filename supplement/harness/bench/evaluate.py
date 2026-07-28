"""T1 LLM-as-judge evaluator (spec §4.5).

Three metrics:
- `answer_correctness(answer, reference)` → 1-4 → normalized to 0-100.
- `source_grounding(cited, reference)` → 0/1 — citation overlap with the
  reference moment set.
- `hallucination_rate(answer, corpus_descriptions)` → 0/1 — LLM judge
  decides whether the answer contains claims unsupported by the corpus
  snippets it was given.

Judge model defaults to `gemini-2.5-flash` (fast + cheap; spec §10.1
explicitly leaves the judge model swappable — we can re-judge with Sonnet
4.6 for stability later).

If a rubric file exists at `docs/research/pwm-bench-t1-rubric.md` it's
loaded as the rubric body for the correctness prompt; otherwise the
inline rubric below is used (per spec §4.5 ref to "fixed rubric").
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
RUBRIC_PATH = Path("docs/research/pwm-bench-t1-rubric.md")

# Inline fallback rubric (used when RUBRIC_PATH is absent). Mirrors the
# 1-4 scale called out in spec §4.5.
DEFAULT_RUBRIC = """\
1 - Wrong: answer contradicts the reference, or is irrelevant.
2 - Partially correct: shares some elements with the reference but misses
    central content or makes one or more inaccurate claims.
3 - Mostly correct: aligns with the reference's main content; minor
    omissions or phrasing differences only.
4 - Fully correct: captures the reference's content faithfully, with
    no inaccuracies and no material omissions.
"""

_CORRECTNESS_PROMPT = """\
You are evaluating an answer to a question about someone's personal photo archive.

Question: {question}

Reference answer (ground truth, authored by the persona):
{reference}

Candidate answer (produced by a substrate under test):
{answer}

Rubric (1-4):
{rubric}

Score the candidate answer against the reference using the rubric.
Respond with EXACTLY one JSON object on a single line:
{{"score": <1|2|3|4>, "rationale": "<one sentence>"}}
"""

_GROUNDING_PROMPT = """\
You are checking whether an answer is grounded in the evidence the system cited.

Question: {question}

Answer produced by the system:
{answer}

Evidence — the moment descriptions the system cited/retrieved as its basis:
{evidence}

Spec §4.5 source grounding: does AT LEAST ONE evidence moment above substantively
support a claim made in the answer? A moment "supports" the answer when its content
corroborates a concrete claim in the answer (an activity, place, person, time, or
fact). Loose topical relatedness is NOT support.

Respond with EXACTLY one JSON object on a single line:
{{"grounded": <true|false>, "rationale": "<one sentence>"}}
"""

_HALLUCINATION_PROMPT = """\
You are checking whether an answer FABRICATES specific facts about a person's
life that are absent from their archive.

Answer to check:
{answer}

Corpus (each line is a moment description from the person's archive):
{corpus}

Mark hallucinated = true ONLY if the answer asserts a concrete, checkable fact
that is either CONTRADICTED by the corpus or has NO basis in it — for example:
- a named person, place, or event that appears nowhere in the corpus,
- a specific date, count, duration, or quantity that the corpus does not support,
- a definite claim the corpus directly contradicts.

Mark hallucinated = false (this is the default) for everything else, including:
- reasonable synthesis, summarization, generalization, or inference drawn from
  the corpus (e.g. "you seem to enjoy the outdoors" from several hiking moments),
- hedged or approximate statements ("a few times", "often", "around 2024"),
- honest "the archive doesn't contain enough information to answer" responses,
- correct paraphrase even if the exact wording isn't in the corpus.

Judge only FABRICATION of specifics, not whether every phrase is verbatim in the
corpus. When uncertain, answer false.

Respond with EXACTLY one JSON object on a single line:
{{"hallucinated": <true|false>, "rationale": "<one sentence>"}}
"""


@dataclass
class EvalScores:
    """Per-question evaluation result."""

    correctness_raw: int  # 1-4
    correctness_norm: float  # 0-100, linear: (raw-1)/3 * 100
    grounding: float  # 0.0 or 1.0
    hallucination: float  # 0.0 or 1.0
    correctness_rationale: str = ""
    hallucination_rationale: str = ""
    grounding_rationale: str = ""


class T1Evaluator:
    """Bundles judge LLM + corpus, scores a (question, answer, cited) triple."""

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        rubric_path: Path | str | None = None,
    ):
        from ..infrastructure.config import Settings
        from ..infrastructure.llm.gemini_llm_service import GeminiLLMService
        self._settings = Settings.from_env()
        self._judge = GeminiLLMService(model=judge_model, settings=self._settings)
        self._judge_model = judge_model
        self._rubric = _load_rubric(Path(rubric_path) if rubric_path else RUBRIC_PATH)

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        cited_moment_ids: list[str],
        reference_moment_ids: list[str],
        corpus_descriptions: list[str],
        cited_descriptions: list[str] | None = None,
    ) -> EvalScores:
        correctness_raw, c_rationale = self._score_correctness(
            question, answer, reference_answer
        )
        grounding, g_rationale = self._judge_grounding(
            question, answer, cited_descriptions or []
        )
        hall, h_rationale = self._judge_hallucination(answer, corpus_descriptions)
        return EvalScores(
            correctness_raw=correctness_raw,
            correctness_norm=(max(1, min(4, correctness_raw)) - 1) / 3 * 100,
            grounding=grounding,
            hallucination=hall,
            correctness_rationale=c_rationale,
            hallucination_rationale=h_rationale,
            grounding_rationale=g_rationale,
        )

    def _score_correctness(
        self, question: str, answer: str, reference: str
    ) -> tuple[int, str]:
        prompt = _CORRECTNESS_PROMPT.format(
            question=question,
            reference=reference,
            answer=answer or "(empty)",
            rubric=self._rubric,
        )
        try:
            raw = self._judge.generate(prompt)
        except Exception as e:
            logger.exception("Correctness judge failed")
            return 1, f"judge error: {e}"
        return _parse_correctness(raw)

    def _judge_grounding(
        self, question: str, answer: str, cited_descriptions: list[str]
    ) -> tuple[float, str]:
        """Spec §4.5: does ≥1 cited/retrieved moment support the answer?

        Judged rather than ID-matched against the human reference set — a
        system that retrieves a different-but-valid supporting moment should
        score as grounded. No cited evidence at all → not grounded.
        """
        if not cited_descriptions:
            return 0.0, "no cited/retrieved evidence to ground the answer"
        # Bound the evidence block; the top retrieved moments are the relevant
        # grounding candidates, and the judge only needs one that supports.
        ev = cited_descriptions[:40]
        evidence = "\n".join(f"- {d}" for d in ev if d)
        prompt = _GROUNDING_PROMPT.format(
            question=question, answer=answer or "(empty)", evidence=evidence
        )
        try:
            raw = self._judge.generate(prompt)
        except Exception as e:
            logger.exception("Grounding judge failed")
            return 0.0, f"judge error: {e}"
        return _parse_grounding(raw)

    def _judge_hallucination(
        self, answer: str, corpus_descriptions: list[str]
    ) -> tuple[float, str]:
        # Spec §4.5 defines hallucination as claims unsupported by ANY moment
        # in the substrate, so the judge must see the whole archive — not an
        # arbitrary prefix. Truncating to the first N moments wrongly flags
        # answers grounded in later moments (a 686-moment persona was scoring
        # hallucination=1.0 on correct answers). The judge model (Gemini Flash)
        # has a 1M-token window; a few thousand short descriptions fit easily.
        # Keep a high guard only to bound pathological archives.
        cap = 5000
        if len(corpus_descriptions) > cap:
            logger.warning(
                "hallucination corpus has %d moments; capping to %d for the "
                "judge prompt (archive larger than expected)",
                len(corpus_descriptions), cap,
            )
            corpus_descriptions = corpus_descriptions[:cap]
        corpus = "\n".join(f"- {d}" for d in corpus_descriptions if d)
        prompt = _HALLUCINATION_PROMPT.format(
            answer=answer or "(empty)",
            corpus=corpus or "(no corpus)",
        )
        try:
            raw = self._judge.generate(prompt)
        except Exception as e:
            logger.exception("Hallucination judge failed")
            return 0.0, f"judge error: {e}"
        return _parse_hallucination(raw)


# --- Pure helpers ---------------------------------------------------------


def source_grounding(
    cited_moment_ids: list[str], reference_moment_ids: list[str]
) -> float:
    """Spec §4.5: binary — answer cites ≥1 supporting reference moment."""
    if not reference_moment_ids or not cited_moment_ids:
        return 0.0
    ref = set(reference_moment_ids)
    cited = set(cited_moment_ids)
    return 1.0 if (ref & cited) else 0.0


def _parse_correctness(raw: str) -> tuple[int, str]:
    # The judge prompt asks for a single JSON object; tolerate stray prose.
    obj = _extract_json_obj(raw)
    if obj is None:
        # Last-ditch: scan for a 1-4 digit and use that.
        m = re.search(r"\b([1-4])\b", raw)
        return (int(m.group(1)) if m else 1, "no JSON in judge output")
    try:
        score = int(obj.get("score", 1))
    except (TypeError, ValueError):
        score = 1
    return max(1, min(4, score)), str(obj.get("rationale", ""))[:200]


def _parse_hallucination(raw: str) -> tuple[float, str]:
    obj = _extract_json_obj(raw)
    if obj is None:
        # If we can't parse, conservative default: not hallucinated.
        # (Avoids penalizing substrates for judge failures.)
        return 0.0, "no JSON in judge output"
    val = obj.get("hallucinated")
    hall = 1.0 if (val is True or str(val).lower() == "true") else 0.0
    return hall, str(obj.get("rationale", ""))[:200]


def _parse_grounding(raw: str) -> tuple[float, str]:
    obj = _extract_json_obj(raw)
    if obj is None:
        # Can't parse → conservative default: not grounded.
        return 0.0, "no JSON in judge output"
    val = obj.get("grounded")
    grounded = 1.0 if (val is True or str(val).lower() == "true") else 0.0
    return grounded, str(obj.get("rationale", ""))[:200]


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_obj(raw: str) -> dict | None:
    """Find the first {...} block and parse it. Returns None on failure."""
    if not raw:
        return None
    # Strip common markdown fences.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    for m in _JSON_RE.finditer(raw):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    # One more attempt on the whole stripped string.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _load_rubric(path: Path) -> str:
    if not path.exists():
        return DEFAULT_RUBRIC
    try:
        return path.read_text()
    except OSError:
        return DEFAULT_RUBRIC
