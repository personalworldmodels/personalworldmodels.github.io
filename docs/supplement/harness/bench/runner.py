"""T1 runner — orchestrates substrate adapters + evaluator.

Reads a JSONL of T1 tasks, queries each requested substrate, evaluates each
(task, substrate) result with the LLM-as-judge evaluator, and writes:
- A per-question CSV at the requested output path.
- An aggregate JSON summary at `<output_stem>-summary.json`.

CSV columns (locked):
    id, category, system, answer, cited_ids, correctness, grounding,
    hallucination, elapsed_ms

Aggregate JSON shape:
{
  "spec_version": "1.0-draft",
  "personas": ["hana"],
  "systems": ["pwm", ...],
  "n_questions": 3,
  "judge_model": "gemini-2.5-flash",
  "per_system": {
    "pwm": {
       "n": 3,
       "correctness_mean": 75.0,
       "grounding_rate": 0.66,
       "hallucination_rate": 0.0,
       "avg_elapsed_ms": 4200,
       "per_category": {"recurring_patterns": {...}, ...}
    }, ...
  }
}
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .evaluate import EvalScores, T1Evaluator
from .loader import T1Task

if TYPE_CHECKING:
    from .substrate import Substrate, SubstrateResponse

logger = logging.getLogger(__name__)


# Registry: maps --systems flag tokens to adapter constructors.
# Constructors take (persona, data_root) and return a Substrate.
def _build_pwm(persona: str, data_root: str):
    from .adapters.pwm import PWMAdapter
    return PWMAdapter(persona=persona, data_root=data_root)


def _build_mem0(persona: str, data_root: str):
    from .adapters.mem0 import Mem0Adapter
    return Mem0Adapter(persona=persona, data_root=data_root)


def _build_gemini_1m(persona: str, data_root: str):
    from .adapters.gemini_long_context import GeminiLongContextAdapter
    return GeminiLongContextAdapter(persona=persona, data_root=data_root)


def _build_vanilla_rag(persona: str, data_root: str):
    from .adapters.vanilla_rag import VanillaRAGAdapter
    # top_k=30 matched to PWM for a fair recall-breadth comparison.
    return VanillaRAGAdapter(persona=persona, data_root=data_root, top_k=30)


def _build_vanilla_strong(persona: str, data_root: str):
    from .adapters.vanilla_strong import VanillaStrongAdapter
    return VanillaStrongAdapter(persona=persona, data_root=data_root)


def _build_graphrag(persona: str, data_root: str):
    from .adapters.graphrag_adapter import GraphRAGAdapter
    return GraphRAGAdapter(persona=persona, data_root=data_root)


def _build_zep(persona: str, data_root: str):
    from .adapters.zep import ZepAdapter
    return ZepAdapter(persona=persona, data_root=data_root)


def _build_supermemory(persona: str, data_root: str):
    from .adapters.supermemory import SupermemoryAdapter
    return SupermemoryAdapter(persona=persona, data_root=data_root)


ADAPTER_REGISTRY: dict[str, callable] = {
    "pwm": _build_pwm,
    "mem0": _build_mem0,
    "gemini_1m": _build_gemini_1m,
    "vanilla_rag": _build_vanilla_rag,
    "vanilla_strong": _build_vanilla_strong,
    "graphrag": _build_graphrag,
    "zep": _build_zep,
    "supermemory": _build_supermemory,
    # Aliases for ergonomics.
    "gemini": _build_gemini_1m,
    "longcontext": _build_gemini_1m,
    "rag": _build_vanilla_rag,
}


@dataclass
class _RunRow:
    """One evaluated (task, system) row — flat for CSV emission."""

    id: str
    category: str
    system: str
    answer: str
    cited_ids: list[str]
    correctness: float  # normalized 0-100
    correctness_raw: int  # 1-4
    grounding: float
    hallucination: float
    elapsed_ms: float
    correctness_rationale: str = ""
    hallucination_rationale: str = ""
    grounding_rationale: str = ""


def run_t1(
    tasks_path: str | Path,
    systems: list[str],
    output_csv: str | Path,
    data_root: str = "data",
    judge_model: str | None = None,
    summary_path: str | Path | None = None,
) -> dict:
    """Run T1 over `tasks_path` × `systems`. Writes CSV + summary JSON.

    Returns the aggregate summary dict.
    """
    from .loader import load_tasks
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise ValueError(f"No tasks loaded from {tasks_path}")

    # Resolve adapter constructors first so unknown system names error
    # before we do any expensive ingest work.
    builders: list[tuple[str, callable]] = []
    for s in systems:
        s_norm = s.strip().lower()
        if s_norm not in ADAPTER_REGISTRY:
            raise ValueError(
                f"Unknown system {s!r}. Known: {sorted(ADAPTER_REGISTRY)}"
            )
        builders.append((s_norm, ADAPTER_REGISTRY[s_norm]))

    # Group tasks by persona — most runs are single-persona but the loader
    # supports multi-persona files. One adapter instance per (system, persona).
    by_persona: dict[str, list[T1Task]] = defaultdict(list)
    for t in tasks:
        by_persona[t.persona].append(t)

    evaluator = T1Evaluator(judge_model=judge_model or "gemini-2.5-flash")

    rows: list[_RunRow] = []
    for persona, persona_tasks in by_persona.items():
        # Build corpus once per persona for the hallucination + grounding judges.
        corpus, id_to_desc = _load_persona_corpus(persona=persona, data_root=data_root)
        for system_name, builder in builders:
            logger.info("Building adapter %s for persona=%s", system_name, persona)
            try:
                adapter: Substrate = builder(persona, data_root)
            except Exception as e:
                logger.exception("Adapter %s init failed", system_name)
                # Emit one synthetic error row per task so the CSV documents
                # the failure rather than silently dropping the system.
                for t in persona_tasks:
                    rows.append(_RunRow(
                        id=t.id, category=t.category, system=system_name,
                        answer=f"[adapter init failed: {e}]",
                        cited_ids=[], correctness=0.0, correctness_raw=1,
                        grounding=0.0, hallucination=0.0, elapsed_ms=0.0,
                        correctness_rationale="adapter init failed",
                    ))
                continue

            for t in persona_tasks:
                logger.info("[%s] %s: %s", system_name, t.id, t.question[:60])
                try:
                    resp: SubstrateResponse = adapter.query(t.question, persona)
                except Exception as e:
                    logger.exception("Adapter %s query failed on %s", system_name, t.id)
                    rows.append(_RunRow(
                        id=t.id, category=t.category, system=system_name,
                        answer=f"[query failed: {e}]",
                        cited_ids=[], correctness=0.0, correctness_raw=1,
                        grounding=0.0, hallucination=0.0, elapsed_ms=0.0,
                        correctness_rationale="query failed",
                    ))
                    continue

                # Resolve the system's grounding evidence for the grounding
                # judge. Prefer evidence text the adapter supplied directly
                # (e.g. Zep facts, which have no recoverable moment IDs); else
                # resolve cited moment IDs → descriptions via the corpus map.
                cited_descriptions = list(resp.cited_texts) or [
                    id_to_desc[i] for i in resp.cited_moment_ids if i in id_to_desc
                ]
                scores: EvalScores = evaluator.evaluate(
                    question=t.question,
                    answer=resp.answer,
                    reference_answer=t.reference_answer,
                    cited_moment_ids=resp.cited_moment_ids,
                    reference_moment_ids=list(t.reference_moment_ids),
                    corpus_descriptions=corpus,
                    cited_descriptions=cited_descriptions,
                )
                rows.append(_RunRow(
                    id=t.id,
                    category=t.category,
                    system=system_name,
                    answer=resp.answer,
                    cited_ids=resp.cited_moment_ids,
                    correctness=scores.correctness_norm,
                    correctness_raw=scores.correctness_raw,
                    grounding=scores.grounding,
                    hallucination=scores.hallucination,
                    elapsed_ms=resp.elapsed_ms,
                    correctness_rationale=scores.correctness_rationale,
                    hallucination_rationale=scores.hallucination_rationale,
                    grounding_rationale=scores.grounding_rationale,
                ))

    # --- Write outputs ---
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, rows)

    summary = _build_summary(
        rows=rows,
        personas=sorted(by_persona),
        systems=[name for name, _ in builders],
        judge_model=judge_model or "gemini-2.5-flash",
        n_questions=len(tasks),
    )
    if summary_path is None:
        summary_path = out_csv.with_name(f"{out_csv.stem}-summary.json")
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    return summary


def _write_csv(path: Path, rows: list[_RunRow]) -> None:
    fields = [
        "id", "category", "system", "answer", "cited_ids",
        "correctness", "correctness_raw", "grounding", "hallucination",
        "elapsed_ms", "correctness_rationale", "hallucination_rationale",
        "grounding_rationale",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([
                r.id, r.category, r.system,
                r.answer.replace("\n", " "),
                ";".join(r.cited_ids),
                f"{r.correctness:.2f}", r.correctness_raw,
                f"{r.grounding:.2f}", f"{r.hallucination:.2f}",
                f"{r.elapsed_ms:.1f}",
                r.correctness_rationale.replace("\n", " "),
                r.hallucination_rationale.replace("\n", " "),
                r.grounding_rationale.replace("\n", " "),
            ])


def _build_summary(
    rows: list[_RunRow],
    personas: list[str],
    systems: list[str],
    judge_model: str,
    n_questions: int,
) -> dict:
    per_system: dict[str, dict] = {}
    for sys_name in systems:
        sys_rows = [r for r in rows if r.system == sys_name]
        if not sys_rows:
            per_system[sys_name] = {"n": 0}
            continue
        per_cat: dict[str, dict] = {}
        for cat in sorted({r.category for r in sys_rows}):
            cat_rows = [r for r in sys_rows if r.category == cat]
            per_cat[cat] = _agg(cat_rows)
        per_system[sys_name] = {
            **_agg(sys_rows),
            "per_category": per_cat,
        }
    return {
        "spec_version": "1.0-draft",
        "personas": personas,
        "systems": systems,
        "n_questions": n_questions,
        "judge_model": judge_model,
        "per_system": per_system,
    }


def _agg(rows: list[_RunRow]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "correctness_mean": round(statistics.mean(r.correctness for r in rows), 2),
        "grounding_rate": round(statistics.mean(r.grounding for r in rows), 3),
        "hallucination_rate": round(statistics.mean(r.hallucination for r in rows), 3),
        "avg_elapsed_ms": round(statistics.mean(r.elapsed_ms for r in rows), 1),
    }


def _load_persona_corpus(
    persona: str, data_root: str
) -> tuple[list[str], dict[str, str]]:
    """Pull all moment descriptions for the judges.

    Returns (corpus, id_to_desc): the description list for the hallucination
    judge, and a moment_id → description map so the grounding judge can resolve
    a system's cited/retrieved moment IDs back to their text and decide whether
    any of them supports the answer (spec §4.5).
    """
    from ..interfaces.cli.shared import get_repos
    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    moments = repos.moment.find_all(limit=20000)
    out: list[str] = []
    id_to_desc: dict[str, str] = {}
    for m in moments:
        desc = (m.description or "").strip().replace("\n", " ")
        if desc:
            out.append(desc)
            id_to_desc[m.id] = desc
    return out, id_to_desc
