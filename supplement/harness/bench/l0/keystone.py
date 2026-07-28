"""L0 keystone: does substrate fidelity PREDICT consumer performance?

The cross-architecture / "personal world model" claim rests on a causal link —
a substrate whose concepts are faithful (L0) should make its consumers (T1-T4)
perform well. With a single substrate we cannot scatter across natural systems,
so we degrade the substrate's grouping at increasing noise p and measure two
DIFFERENT things at each level:

- L0 fidelity(p):  do the (corrupted) spot prototypes still recognize a held-out
  moment as its TRUE place?  (intrinsic concept fidelity — Posner-Keele
  prototype classification)
- T3 consumer(p):  using the moment's (corrupted) spot anchor, retrieve the right
  unseen moment's texture  (the T3 world-model task, retr@1)

If the two fall together as the substrate degrades, substrate fidelity predicts
consumer performance — the keystone, reported as a Pearson correlation.

Honest scope: controlled single-persona ablation with a shared corruption knob
(both metrics read the spot grouping, so co-variation is expected — that IS the
point: the grouping is their common cause). A natural cross-system / cross-persona
scatter is v1.1.
"""
from __future__ import annotations

import logging
import math
import random
from collections import defaultdict

import numpy as np

from ..t3.reconstruction import _parse_date, _unit, _vec

logger = logging.getLogger(__name__)


def _pearson(xs, ys) -> float:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_keystone(
    persona: str,
    data_root: str,
    levels=(0.0, 0.25, 0.5, 0.75, 1.0),
    holdout_frac: float = 0.20,
    seed: int = 0,
) -> dict:
    from ...interfaces.cli.shared import get_repos

    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    g = repos.graph
    moments = {m.id: m for m in repos.moment.find_all(limit=20000)}
    dates = g.get_dates_by_moments(list(moments))
    spot_ids = [s["id"] for s in g.get_spots(persona)]
    m2s = {}
    for sid in spot_ids:
        for mid in (g.get_moments_for_spot(sid) or []):
            k = mid if isinstance(mid, str) else getattr(mid, "id", None)
            if k and k not in m2s:
                m2s[k] = sid
    items = []
    for mid, m in moments.items():
        d = _parse_date(dates.get(mid))
        v = _vec(m, "visual_embedding")
        sid = m2s.get(mid)
        if d is not None and v is not None and sid is not None:
            items.append({"mid": mid, "vec": _unit(v), "spot": sid})
    n = len(items)
    rng = np.random.default_rng(seed)
    mask = set(rng.choice(n, size=max(1, math.floor(holdout_frac * n)), replace=False).tolist())
    hist = [x for i, x in enumerate(items) if i not in mask]
    test = [items[i] for i in sorted(mask)]
    true_mat = np.stack([x["vec"] for x in test])
    all_spots = list({x["spot"] for x in items})

    rows = []
    for p in levels:
        r = random.Random(seed)
        # corrupt grouping: with prob p, reassign a moment's spot to a random one
        def corrupt(s):
            return r.choice(all_spots) if r.random() < p else s
        hist_c = [{**x, "cspot": corrupt(x["spot"])} for x in hist]
        test_c = [{**x, "cspot": corrupt(x["spot"])} for x in test]
        by = defaultdict(list)
        for x in hist_c:
            by[x["cspot"]].append(x["vec"])
        proto = {s: _unit(np.stack(v).mean(0)) for s, v in by.items()}
        if not proto:
            continue
        pids = list(proto)
        P = np.stack([proto[s] for s in pids])

        # L0 fidelity: nearest corrupted-prototype == TRUE spot?
        l0 = np.mean([pids[int(np.argmax(P @ x["vec"]))] == x["spot"] for x in test_c])
        # T3 consumer: predict from the moment's (corrupted) spot prototype, retrieve@1
        hit = 0
        for j, x in enumerate(test_c):
            pred = proto.get(x["cspot"])
            if pred is None:
                continue
            if int(np.argmax(true_mat @ pred)) == j:
                hit += 1
        t3 = hit / len(test_c)
        rows.append({"noise": p, "l0_fidelity": round(float(l0), 4), "t3_retr_top1": round(t3, 4)})

    r = _pearson([x["l0_fidelity"] for x in rows], [x["t3_retr_top1"] for x in rows])
    return {
        "track": "L0 keystone — substrate fidelity vs consumer performance",
        "persona": persona,
        "design": "single-persona degradation ablation (shared spot-grouping corruption knob)",
        "levels": rows,
        "pearson_l0_vs_t3": round(r, 3),
        "interpretation": (
            "positive r => as substrate concept-fidelity (L0) degrades, the world-model "
            "consumer (T3) degrades with it: substrate fidelity predicts consumer performance."
        ),
    }
