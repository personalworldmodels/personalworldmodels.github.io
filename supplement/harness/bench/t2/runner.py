"""T2 orchestrator: run the substrate query-execution agent and the flattened-
text baseline over the question set; score by EXECUTION exact-match.

`substrate`  — LLM selects a typed graph tool + args; we execute it deterministically.
`flat_text`  — strong LLM reads the same facts serialized as text and answers.

Headline: substrate accuracy vs flat_text, per category — especially on the
text-insufficient categories (count / multi_hop / superlative).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .graph_ops import TOOL_SPECS, T2Graph, execute_tool
from .tasks import T2Task, generate

logger = logging.getLogger(__name__)

SPEC_VERSION = "1.0-draft"


# --------------------------------------------------------------------------- #
# Scoring (execution exact-match, per answer kind)
# --------------------------------------------------------------------------- #
_STOP = {"the", "a", "an", "of", "at", "in", "and", "my", "i"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).strip()


def _toks(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t and t not in _STOP}


def score(answer: str, gold: str, kind: str) -> int:
    a = (answer or "").strip()
    if kind == "int":
        m = re.search(r"-?\d+", a.replace(",", ""))
        return int(bool(m) and int(m.group()) == int(gold))
    if kind == "bool":
        al = a.lower()
        said = "yes" if ("yes" in al and "no" not in al[:al.find("yes")+3]) else ("no" if "no" in al else "")
        return int(said == gold)
    if kind == "month":
        return int(gold in a or _norm(gold) in _norm(a))
    if kind == "place":
        gt, at = _toks(gold), _toks(a)
        if not gt:
            return 0
        overlap = len(gt & at) / len(gt)
        return int(overlap >= 0.6 or _norm(gold) in _norm(a))
    if kind == "objset":
        # Normalize BOTH sides identically (lowercase, strip punctuation) so a
        # label like "t-shirt" vs "t shirt" doesn't spuriously mismatch.
        gold_set = {_norm(x) for x in gold.split(",") if _norm(x)}
        ans_set = {_norm(x) for x in re.split(r"[,\n;]| and ", a) if _norm(x)}
        if not gold_set:
            return int(not ans_set or any(w in a.lower() for w in ("none", "no ", "nothing")))
        return int(gold_set == ans_set)
    return int(_norm(a) == _norm(gold))


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #
def _tool_menu() -> str:
    lines = []
    for name, (desc, argnames) in TOOL_SPECS.items():
        lines.append(f"- {name}({', '.join(argnames)}): {desc}")
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _substrate_answer(graph: T2Graph, llm, cfg, task: T2Task) -> str:
    prompt = (
        "You answer questions about a person's life by selecting EXACTLY ONE tool "
        "and its arguments. Do not compute anything yourself.\n\n"
        f"Tools:\n{_tool_menu()}\n\n"
        f"Question: {task.question}\n\n"
        'Respond with one JSON object only: {"tool": "<name>", "args": {<arg>: <value>}}. '
        "For place arguments use the place name exactly as written in the question."
    )
    raw = llm.generate(prompt, config=cfg)
    m = _JSON_RE.search(raw or "")
    if not m:
        return "[no tool selected]"
    try:
        obj = json.loads(m.group())
        result = execute_tool(graph, obj["tool"], obj.get("args", {}))
    except Exception as e:  # noqa: BLE001
        return f"[tool error: {e}]"
    if isinstance(result, (set, frozenset)):
        return ", ".join(sorted(result))
    if isinstance(result, bool):
        return "yes" if result else "no"
    return str(result)


def _flat_text_answer(facts: str, llm, cfg, task: T2Task) -> str:
    prompt = (
        "Here are facts about a person's life archive:\n\n"
        f"{facts}\n\n"
        f"Question: {task.question}\n"
        "Answer concisely with ONLY the answer (a number, a place name, a YYYY-MM month, "
        "a comma-separated list, or yes/no)."
    )
    return (llm.generate(prompt, config=cfg) or "").strip()


def _pot_records(graph: T2Graph) -> list[dict]:
    """The SAME structured data the substrate's tools read, as plain dicts — so the
    PoT baseline computes over identical structure, just with self-written code."""
    from .graph_ops import _spot_name
    recs = []
    for s in graph._spots:
        recs.append({
            "place": _spot_name(s),
            "visit_days": int(s.get("visit_days", 0) or 0),
            "moment_count": int(s.get("moment_count", 0) or 0),
            "objects": sorted(graph._objects_at(s)),
        })
    months = {}
    from collections import Counter
    mc = Counter(str(d)[:7] for d in graph._dates.values() if d)
    months = dict(mc)
    return recs, months


def _pot_answer(graph: T2Graph, llm, cfg, task: T2Task) -> str:
    """Program-of-Thoughts fair neurosymbolic baseline: the LLM writes Python over
    the same structured data and we EXECUTE it (vs. the substrate picking a
    hand-built tool). Isolates 'curated typed ops' from 'has an execution env.'"""
    recs, months = _pot_records(graph)
    prompt = (
        "You are given Python data about a person's life archive:\n"
        "places: list[dict] with keys 'place'(str), 'visit_days'(int), "
        "'moment_count'(int), 'objects'(list[str]).\n"
        "months: dict[str 'YYYY-MM' -> int moment count].\n\n"
        f"Question: {task.question}\n\n"
        "Write a Python function solve(places, months) that returns the answer "
        "(a number, a place name string, a 'YYYY-MM' month, a list of strings, or a bool). "
        "Return ONLY a fenced ```python code block defining solve."
    )
    raw = llm.generate(prompt, config=cfg) or ""
    import re as _re
    m = _re.search(r"```(?:python)?\s*(.+?)```", raw, _re.DOTALL)
    code = m.group(1) if m else raw
    ns: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, ns)  # noqa: S102 — local bench, our prompt
        result = ns["solve"](recs, months)
    except Exception as e:  # noqa: BLE001
        return f"[pot error: {e}]"
    if isinstance(result, (set, frozenset, list, tuple)):
        return ", ".join(sorted(str(x) for x in result))
    if isinstance(result, bool):
        return "yes" if result else "no"
    return str(result)


@dataclass
class _Row:
    system: str
    id: str
    category: str
    gold: str
    answer: str
    correct: int


def run_t2(
    persona: str,
    data_root: str,
    output_csv: str | Path,
    systems: list[str] | None = None,
    model: str | None = None,
    summary_path: str | Path | None = None,
) -> dict:
    from ...domain.ports.infra.llm import LLMConfig
    from ...infrastructure.config import Settings
    from ...infrastructure.llm import get_synth_service
    from ...interfaces.cli.shared import get_repos

    systems = systems or ["substrate", "flat_text"]
    model = model or os.environ.get("PWMBENCH_SYNTH_MODEL") or "gemini-2.5-flash"
    settings = Settings.from_env()
    llm = get_synth_service(model, settings)
    cfg = LLMConfig(temperature=0.0, max_output_tokens=1024)

    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    graph = T2Graph(repos, persona)
    tasks = generate(graph)
    facts = graph.serialize_facts()

    rows: list[_Row] = []
    for t in tasks:
        for sysname in systems:
            try:
                if sysname == "substrate":
                    ans = _substrate_answer(graph, llm, cfg, t)
                elif sysname == "pot":
                    ans = _pot_answer(graph, llm, cfg, t)
                else:
                    ans = _flat_text_answer(facts, llm, cfg, t)
            except Exception as e:  # noqa: BLE001
                logger.exception("T2 %s failed on %s", sysname, t.id)
                ans = f"[error: {e}]"
            rows.append(_Row(sysname, t.id, t.category, t.gold, ans, score(ans, t.gold, t.gold_kind)))

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "id", "category", "gold", "answer", "correct"])
        for r in rows:
            w.writerow([r.system, r.id, r.category, r.gold, r.answer.replace("\n", " ")[:300], r.correct])

    cats = sorted({t.category for t in tasks})
    per_system = {}
    for sysname in systems:
        srs = [r for r in rows if r.system == sysname]
        per_system[sysname] = {
            "n": len(srs),
            "accuracy": round(statistics.mean(r.correct for r in srs), 4) if srs else 0,
            "by_category": {
                c: round(statistics.mean([r.correct for r in srs if r.category == c]), 4)
                for c in cats if any(r.category == c for r in srs)
            },
        }
    summary = {
        "spec_version": SPEC_VERSION,
        "track": "T2 — neurosymbolic (graph-query execution)",
        "persona": persona,
        "model": model,
        "n_tasks": len(tasks),
        "categories": cats,
        "per_system": per_system,
        "headline_substrate_minus_flat_text": (
            round(per_system.get("substrate", {}).get("accuracy", 0)
                  - per_system.get("flat_text", {}).get("accuracy", 0), 4)
            if {"substrate", "flat_text"} <= set(systems) else None
        ),
    }
    sp = Path(summary_path) if summary_path else out.with_name(f"{out.stem}-summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    return summary
