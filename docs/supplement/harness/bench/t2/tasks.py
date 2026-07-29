"""T2 question set: NL questions requiring exact symbolic computation, with
ground truth computed by direct graph execution (the oracle).

Categories are chosen so a strong LLM reading the same facts as flattened text
provably struggles (counting/intersection/superlative over a long list), while
execution over the structure is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .graph_ops import T2Graph, _spot_name


@dataclass(frozen=True)
class T2Task:
    id: str
    category: str
    question: str
    gold: str           # normalized gold answer
    gold_kind: str      # int | place | month | objset | bool


def _objrich_pairs(graph: T2Graph, k: int = 4) -> list[tuple[dict, dict]]:
    spots = [s for s in graph.spots_sorted() if s.get("location_name")][:12]
    pairs = []
    for i in range(len(spots)):
        for j in range(i + 1, len(spots)):
            a, b = spots[i], spots[j]
            if graph.common_objects(_spot_name(a), _spot_name(b)):
                pairs.append((a, b))
    return pairs[:k]


def generate(graph: T2Graph) -> list[T2Task]:
    tasks: list[T2Task] = []
    top = graph.spots_sorted()
    add = lambda *a: tasks.append(T2Task(f"t2_{len(tasks)+1:03d}", *a))

    # 1. exact-count
    for s in top[:5]:
        nm = _spot_name(s)
        add("exact_count", f"How many distinct days did I visit {nm}?", str(graph.visit_days(nm)), "int")
    for n in (2, 3, 5):
        add("exact_count", f"How many places have I visited on at least {n} distinct days?",
            str(graph.count_places_min_days(n)), "int")
    add("exact_count", "How many distinct places are in my archive?", str(graph.total_places()), "int")

    # 2. multi-hop set-intersection
    for a, b in _objrich_pairs(graph):
        na, nb = _spot_name(a), _spot_name(b)
        gold = ", ".join(sorted(graph.common_objects(na, nb)))
        add("multi_hop", f"Which objects appear at BOTH {na} and {nb}? List them.", gold, "objset")

    # 3. group-by / superlative
    add("superlative", "Which place have I visited on the most distinct days?", graph.rank_place_by_visit_days(1), "place")
    add("superlative", "Which place is my 2nd most-visited by distinct days?", graph.rank_place_by_visit_days(2), "place")
    add("superlative", "Which place is my 3rd most-visited by distinct days?", graph.rank_place_by_visit_days(3), "place")
    add("superlative", "Which calendar month (YYYY-MM) has the most moments?", graph.busiest_month(), "month")

    # 4. temporal recurrence
    add("temporal", "On how many distinct days did I visit my single most-frequented place?",
        str(graph.visit_days(graph.rank_place_by_visit_days(1))), "int")

    # 5. consistency / constraint
    if len(top) >= 4:
        a, b = top[0], top[3]
        na, nb = _spot_name(a), _spot_name(b)
        add("consistency", f"Did I visit {na} on more distinct days than {nb}? Answer yes or no.",
            "yes" if graph.more_often(na, nb) else "no", "bool")
        add("consistency", f"Did I visit {nb} on more distinct days than {na}? Answer yes or no.",
            "yes" if graph.more_often(nb, na) else "no", "bool")

    return tasks
