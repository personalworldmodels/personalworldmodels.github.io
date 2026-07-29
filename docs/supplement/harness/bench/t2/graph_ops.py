"""T2 graph operations: the substrate's symbolic-introspection surface.

These are the exact-computation primitives a query-execution agent runs over
the persona's KùzuDB graph — and the same functions compute the oracle ground
truth. Each is a deterministic graph query (no LLM), so execution is exact and
tool-noise-free. The substrate system selects + parameterizes one of these per
question; the flattened-text baseline must instead read serialized facts.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


def _spot_name(s: dict) -> str:
    return s.get("user_label") or s.get("spot_type") or s.get("location_name") or s["id"]


class T2Graph:
    """Cached, read-only view of the substrate's symbolic structure."""

    def __init__(self, repos, persona: str):
        self._g = repos.graph
        self._persona = persona
        self._spots = repos.graph.get_spots(persona)
        self._by_name = {_spot_name(s).lower(): s for s in self._spots}
        # all moment dates (for temporal group-by)
        moments = repos.moment.find_all(limit=20000)
        self._dates = repos.graph.get_dates_by_moments([m.id for m in moments])
        self._n_moments = len(moments)
        self._obj_cache: dict[str, set[str]] = {}

    # --- resolution ---
    def resolve_place(self, name: str) -> dict | None:
        if not name:
            return None
        key = name.strip().lower()
        if key in self._by_name:
            return self._by_name[key]
        # substring / fuzzy: best spot whose name contains or is contained
        cand = [s for n, s in self._by_name.items() if key in n or n in key]
        if cand:
            return max(cand, key=lambda s: s.get("visit_days", 0))
        # token overlap fallback
        toks = set(key.split())
        scored = [(len(toks & set(n.split())), s) for n, s in self._by_name.items()]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] else None

    def _objects_at(self, spot: dict) -> set[str]:
        loc = spot.get("location_name")
        if not loc:
            return set()
        if loc in self._obj_cache:
            return self._obj_cache[loc]
        ents = self._g.get_entities_for_location(loc) or []
        objs = {
            (e.get("label") or "").strip().lower()
            for e in ents
            if (e.get("type") or e.get("kind")) == "object" and e.get("label")
        }
        self._obj_cache[loc] = objs
        return objs

    # --- typed operations (tools + oracle) ---
    def visit_days(self, place: str) -> int:
        s = self.resolve_place(place)
        return int(s.get("visit_days", 0)) if s else 0

    def moment_count(self, place: str) -> int:
        s = self.resolve_place(place)
        return int(s.get("moment_count", 0)) if s else 0

    def total_places(self) -> int:
        return len(self._spots)

    def count_places_min_days(self, min_days: int) -> int:
        return sum(1 for s in self._spots if (s.get("visit_days") or 0) >= int(min_days))

    def rank_place_by_visit_days(self, rank: int) -> str:
        ordered = sorted(self._spots, key=lambda s: s.get("visit_days", 0), reverse=True)
        i = int(rank) - 1
        return _spot_name(ordered[i]) if 0 <= i < len(ordered) else ""

    def max_visit_days(self) -> int:
        return max((int(s.get("visit_days", 0)) for s in self._spots), default=0)

    def busiest_month(self) -> str:
        c: Counter = Counter()
        for d in self._dates.values():
            if d:
                c[str(d)[:7]] += 1
        return c.most_common(1)[0][0] if c else ""

    def common_objects(self, place_a: str, place_b: str) -> set[str]:
        a, b = self.resolve_place(place_a), self.resolve_place(place_b)
        if not a or not b:
            return set()
        return self._objects_at(a) & self._objects_at(b)

    def more_often(self, place_a: str, place_b: str) -> bool:
        return self.visit_days(place_a) > self.visit_days(place_b)

    # --- introspection for question generation + flat-text serialization ---
    def spots_sorted(self) -> list[dict]:
        return sorted(self._spots, key=lambda s: s.get("visit_days", 0), reverse=True)

    def serialize_facts(self, max_spots: int = 200, seed: int = 0) -> str:
        """Flatten the same structure to text for the unconditioned baseline.

        Shuffled (not pre-sorted) so the baseline must actually compute
        superlatives/counts over the list rather than read them off the top —
        otherwise the serialization leaks the answer (StructFact effect).
        """
        import random

        spots = list(self._spots)
        random.Random(seed).shuffle(spots)
        lines = [f"The archive spans {self._n_moments} moments across {len(self._spots)} places."]
        for s in spots[:max_spots]:
            objs = sorted(self._objects_at(s))[:25]
            lines.append(
                f"- {_spot_name(s)}: visited on {s.get('visit_days',0)} distinct days, "
                f"{s.get('moment_count',0)} moments"
                + (f"; objects seen: {', '.join(objs)}" if objs else "")
            )
        # month histogram
        c: Counter = Counter(str(d)[:7] for d in self._dates.values() if d)
        lines.append("Moments per month: " + ", ".join(f"{m}={n}" for m, n in sorted(c.items())))
        return "\n".join(lines)


# Tool dispatch table — name → (callable resolver, arg names). Used by the
# substrate agent: the LLM picks a tool + args, we execute deterministically.
TOOL_SPECS = {
    "visit_days": ("distinct days visited at a place", ["place"]),
    "moment_count": ("number of moments at a place", ["place"]),
    "total_places": ("total number of distinct places", []),
    "count_places_min_days": ("how many places visited on >= N distinct days", ["min_days"]),
    "rank_place_by_visit_days": ("the place at the given rank by distinct visit-days (1=most)", ["rank"]),
    "max_visit_days": ("the largest number of distinct days any single place was visited", []),
    "busiest_month": ("the YYYY-MM month with the most moments", []),
    "common_objects": ("set of objects seen at BOTH places", ["place_a", "place_b"]),
    "more_often": ("True if place_a was visited on more days than place_b", ["place_a", "place_b"]),
}


def execute_tool(graph: T2Graph, tool: str, args: dict):
    fn = getattr(graph, tool, None)
    if fn is None or tool not in TOOL_SPECS:
        raise ValueError(f"unknown tool {tool!r}")
    _, argnames = TOOL_SPECS[tool]
    kwargs = {}
    for a in argnames:
        v = args.get(a)
        if a in ("min_days", "rank"):
            v = int(v)
        kwargs[a] = v
    return fn(**kwargs)
