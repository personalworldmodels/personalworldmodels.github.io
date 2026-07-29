"""T3 vs strong vision baseline. Does T3's substrate advantage survive a SigLIP
k-means / kNN baseline — i.e., is it episodic (stored structure) or perceptual?

Key asymmetry: the substrate predicts a MASKED moment from its STORED spot
anchor — no pixels needed. A vision baseline needs the masked moment's embedding
to cluster/retrieve it; that embedding is the target (masked). We therefore run
the vision baselines WITH LEAKAGE (granting them the masked embedding) as a
generous upper bound, and note that WITHOUT leakage they collapse to the global
mean (no signal for an unseen image). If the substrate beats global without
leakage while vision needs leakage to win, T3's advantage is episodic.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from golgi.bench.t3.reconstruction import _unit, _vec
from golgi.interfaces.cli.shared import get_repos


def load(persona, dr):
    repos = get_repos(persona=persona, data_root=dr, read_only=True)
    g = repos.graph
    moments = {m.id: m for m in repos.moment.find_all(limit=20000)}
    m2s = {}
    for s in g.get_spots(persona):
        for mid in (g.get_moments_for_spot(s["id"]) or []):
            k = mid if isinstance(mid, str) else getattr(mid, "id", None)
            if k and k not in m2s:
                m2s[k] = s["id"]
    items = []
    for mid, m in moments.items():
        v = _vec(m, "visual_embedding")
        sid = m2s.get(mid)
        if v is not None and sid is not None:
            items.append({"vec": _unit(v), "spot": sid})
    return items


def run(persona, dr, holdout=0.2, seed=0, knn=5):
    items = load(persona, dr)
    n = len(items)
    rng = np.random.default_rng(seed)
    mask = set(rng.choice(n, max(1, math.floor(holdout * n)), replace=False).tolist())
    hist = [items[i] for i in range(n) if i not in mask]
    test = [items[i] for i in sorted(mask)]
    true_mat = np.stack([t["vec"] for t in test])
    H = np.stack([x["vec"] for x in hist])

    by_spot = defaultdict(list)
    for x in hist:
        by_spot[x["spot"]].append(x["vec"])
    spot_proto = {s: _unit(np.mean(v, 0)) for s, v in by_spot.items()}
    k = len(by_spot)
    km = KMeans(n_clusters=min(k, len(H)), random_state=seed, n_init=10).fit(H)
    cl_proto = {c: _unit(H[km.labels_ == c].mean(0)) for c in range(km.n_clusters)}
    gmean = _unit(H.mean(0))

    def retr(preds):
        return round(sum(p is not None and int(np.argmax(true_mat @ p)) == j
                         for j, p in enumerate(preds)) / len(test), 3)

    sub = [spot_proto.get(x["spot"]) for x in test]                          # NO leak: stored spot
    vk = [cl_proto[int(km.predict(x["vec"][None])[0])] for x in test]         # LEAK: uses masked vec
    vn = [_unit(H[np.argsort(-(H @ x["vec"]))[:knn]].mean(0)) for x in test]  # LEAK: kNN on masked vec
    gl = [gmean] * len(test)                                                  # NO leak (= vision w/o leak)
    return {"persona": persona, "n_test": len(test), "n_spots": k,
            "substrate_NOLEAK": retr(sub), "vision_kmeans_LEAK": retr(vk),
            "vision_kNN_LEAK": retr(vn), "global_NOLEAK": retr(gl)}


if __name__ == "__main__":
    print(f"{'persona':8}{'n':>5}{'spots':>6}  {'substrate':>10}{'visKmeans':>11}{'visKNN':>9}{'global':>8}")
    print(f"{'':8}{'':>5}{'':>6}  {'(no leak)':>10}{'(LEAK)':>11}{'(LEAK)':>9}{'(no leak)':>8}")
    for persona, dr in [("hana", "data"), ("Maria", "data")]:
        o = run(persona, dr)
        print(f"{persona:8}{o['n_test']:>5}{o['n_spots']:>6}  {o['substrate_NOLEAK']:>10}"
              f"{o['vision_kmeans_LEAK']:>11}{o['vision_kNN_LEAK']:>9}{o['global_NOLEAK']:>8}")
    print("\nRead: vision baselines need the masked embedding (LEAK) to beat global;")
    print("without it they ARE global. Substrate beats global with NO leak = episodic (stored structure).")
