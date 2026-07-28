"""T1 task loader — reads PWM-Bench question JSONL files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Locked categories from spec §4.3. The loader rejects unknown categories
# so a typo in the question set surfaces immediately rather than producing
# silently-uncategorized aggregates.
VALID_CATEGORIES = frozenset(
    {
        "recurring_patterns",
        "lifetime_spans",
        "multi_source",
        "counterfactual",
        "open_ended",
    }
)


@dataclass(frozen=True)
class T1Task:
    """One T1 question against a persona's archive."""

    id: str
    category: str
    persona: str
    question: str
    reference_answer: str
    reference_moment_ids: tuple[str, ...] = ()
    modalities_required: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: dict) -> T1Task:
        """Construct from a JSONL row dict; validate required fields."""
        for key in ("id", "category", "persona", "question", "reference_answer"):
            if key not in d:
                raise ValueError(f"T1Task missing required field: {key!r}")
        if d["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"T1Task {d['id']!r}: unknown category {d['category']!r}. "
                f"Valid: {sorted(VALID_CATEGORIES)}"
            )
        return cls(
            id=d["id"],
            category=d["category"],
            persona=d["persona"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            reference_moment_ids=tuple(d.get("reference_moment_ids", [])),
            modalities_required=tuple(d.get("modalities_required", [])),
        )


def load_tasks(path: str | Path) -> list[T1Task]:
    """Load T1 tasks from a JSONL file. Blank lines tolerated."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"T1 task file not found: {p}")
    tasks: list[T1Task] = []
    with p.open() as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_num}: invalid JSON: {e}") from e
            tasks.append(T1Task.from_dict(d))
    return tasks
