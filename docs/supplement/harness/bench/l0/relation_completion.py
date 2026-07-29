"""L0 Probe 1 — Relation-Completion (held-out edge prediction).

The most direct test of the mental-model definition (Forrester): *relations
between concepts*. We hold out a Space-Activity edge the substrate's topology
should be able to infer, and ask whether a purely *structural* solver recovers
it better than (a) popularity / co-occurrence (the "memorize the marginal" null)
and (b) raw SigLIP visual similarity. Powered on dense relations (n=160+ on
hana) where identity is sparse — this is L0 that current data *can* support.

Protocol: filtered ranking (Bordes 2013 / Nickel 2016) — Hits@1, Hits@3, MRR;
paired Wilcoxon on per-query reciprocal rank. Lit: Gentner 1983 (relation over
surface), Fodor-Pylyshyn 1988 (systematicity), Lake-Baroni 2018 (novel-combo
split), Battaglia 2018 (graph inductive bias).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def _load(persona: str, data_root: str):
    import kuzu
    import lancedb

    db = kuzu.Database(f"{data_root}/{persona}/kuzudb", read_only=True)
    conn = kuzu.Connection(db)
    tbl = lancedb.connect(f"{data_root}/{persona}/lancedb").open_table("moments")
    rows = tbl.to_arrow().to_pylist()
    vfield = next((f for f in ("visual_vector", "vector", "visual_embedding") if rows and f in rows[0]), None)
    vec = {r["id"]: np.asarray(r[vfield], float) for r in rows if vfield and r.get(vfield) is not None}
    return conn, vec


def _label_prop(conn, node: str) -> str:
    for p in ("label", "name", "value", "text"):
        try:
            r = conn.execute(f"MATCH (n:{node}) RETURN n.{p} LIMIT 1")
            if r.has_next() and r.get_next()[0] is not None:
                return p
        except Exception:  # noqa: BLE001
            continue
    return "label"


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def run_relation_completion(
    persona: str,
    data_root: str,
    src=("SHOWS_SPACE", "Space"),
    dst=("HAS_ACTIVITY", "Activity"),
    min_deg: int = 5,
    min_weight: int = 2,
) -> dict:
    conn, vec = _load(persona, data_root)
    src_rel, src_node = src
    dst_rel, dst_node = dst
    sp = _label_prop(conn, src_node)
    dp = _label_prop(conn, dst_node)

    q = (f"MATCH (m:Moment)-[:{src_rel}]->(s:{src_node}), (m)-[:{dst_rel}]->(a:{dst_node}) "
         f"RETURN s.{sp}, a.{dp}, m.id")
    edges: Counter = Counter()
    witness: dict = defaultdict(set)
    src_moments: dict = defaultdict(set)
    dst_moments: dict = defaultdict(set)
    r = conn.execute(q)
    while r.has_next():
        s, a, mid = r.get_next()
        if s is None or a is None:
            continue
        edges[(s, a)] += 1
        witness[(s, a)].add(mid)
        src_moments[s].add(mid)
        dst_moments[a].add(mid)

    deg_s, deg_a = Counter(), Counter()
    for (s, a), w in edges.items():
        deg_s[s] += w
        deg_a[a] += w
    strong_s = {s for s, w in deg_s.items() if w >= min_deg}
    strong_a = {a for a, w in deg_a.items() if w >= min_deg}
    cands = sorted(strong_a)
    testable = [(s, a) for (s, a), w in edges.items()
                if s in strong_s and a in strong_a and w >= min_weight]

    # adjacency for structural solver
    a_of_s: dict = defaultdict(set)
    for (s, a) in edges:
        a_of_s[s].add(a)
    s_of_a: dict = defaultdict(set)
    for (s, a) in edges:
        s_of_a[a].add(s)

    def neigh(s, masked):
        return a_of_s[s] - ({masked[1]} if s == masked[0] else set())

    # precompute candidate activity centroids (embedding baseline)
    a_centroid = {a: np.mean([vec[m] for m in dst_moments[a] if m in vec], axis=0)
                  for a in cands if any(m in vec for m in dst_moments[a])}

    rr = {"substrate": [], "popularity": [], "embedding": []}
    hits1 = {k: 0 for k in rr}
    hits3 = {k: 0 for k in rr}
    n = 0
    for (s_star, a_star) in testable:
        masked = (s_star, a_star)
        W = witness[(s_star, a_star)]
        # filtered candidate set: drop OTHER true activities of s_star
        true_others = a_of_s[s_star] - {a_star}
        cand = [a for a in cands if a not in true_others]
        if a_star not in cand or len(cand) < 3:
            continue
        base_n = neigh(s_star, masked)
        sub = {a: sum(
            (len(base_n & neigh(s2, masked)) / (len(base_n | neigh(s2, masked)) or 1))
            for s2 in s_of_a[a] if s2 != s_star and (s2, a) != masked
        ) for a in cand}
        pop = {a: deg_a[a] for a in cand}
        # embedding: space centroid (minus witnessing moments) vs activity centroids
        sm = [vec[m] for m in src_moments[s_star] if m in vec and m not in W]
        if sm:
            sc = np.mean(sm, axis=0)
            emb = {a: _cos(sc, a_centroid[a]) for a in cand if a in a_centroid}
        else:
            emb = {a: 0.0 for a in cand}

        n += 1
        for name, score in (("substrate", sub), ("popularity", pop), ("embedding", emb)):
            order = sorted(cand, key=lambda a: (-score.get(a, -1e9), a))
            rank = order.index(a_star) + 1
            rr[name].append(1.0 / rank)
            hits1[name] += rank == 1
            hits3[name] += rank <= 3

    def stats(name):
        return {"mrr": round(float(np.mean(rr[name])), 4) if rr[name] else 0.0,
                "hits1": round(hits1[name] / n, 4) if n else 0.0,
                "hits3": round(hits3[name] / n, 4) if n else 0.0}

    out = {
        "probe": "L0 relation-completion (held-out edge)",
        "persona": persona,
        "relation": f"{src_node}<->{dst_node}",
        "n_testable": n,
        "n_candidates": len(cands),
        "per_solver": {k: stats(k) for k in rr},
    }
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_substrate_vs"] = {}
        for b in ("popularity", "embedding"):
            if rr["substrate"] and rr[b]:
                stat, p = wilcoxon(rr["substrate"], rr[b])
                out["wilcoxon_substrate_vs"][b] = {"stat": round(float(stat), 2), "p": float(f"{p:.2e}")}
    except Exception as e:  # noqa: BLE001
        out["wilcoxon_error"] = str(e)[:80]
    return out
