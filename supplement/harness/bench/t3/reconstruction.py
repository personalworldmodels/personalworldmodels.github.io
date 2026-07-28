"""PWM-Bench T3 — world-model track, as masked-moment latent RECONSTRUCTION.

A world model's job is not to guess which of N symbols comes next — it is to
reconstruct the *texture* of an unobserved piece of the world in representation
space (LeCun/JEPA: predict the embedding of a masked target from its context;
MAE: mask randomly, not the future). T3 therefore masks a random ~20% of the
persona's moments and, for each, predicts its SigLIP visual embedding (768-d —
the actual visual texture) from context, scoring by cosine to the true latent
and by whether the prediction RETRIEVES the true moment (specificity).

The cross-architecture test (does the substrate help a world-model consumer?):
- `substrate`  — predict from the moment's SPOT PROTOTYPE: the mean visual
  latent of *visible* moments the substrate anchored at the same place. Uses
  the substrate's structure. Falls back to the global mean for moments at a
  place with no visible history.
- `temporal`   — predict from the mean of the K visible moments nearest in time
  (raw temporal locality, no place knowledge). Unconditioned.
- `global`     — the global mean latent. Structure-free floor.

If the substrate's anchoring is real, knowing "this is the home-studio anchor"
reconstructs the texture of an unseen moment there better than time-locality or
a global average — and RETRIEVES the right moment, which a generic average
cannot. Reported overall and on the anchored subset (where the substrate acts).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SPEC_VERSION = "1.0-draft"
DEFAULT_SEED = 0
DEFAULT_HOLDOUT = 0.20
TEMPORAL_K = 10


def _vec(m, attr: str):
    v = getattr(m, attr, None)
    if v is None:
        return None
    if hasattr(v, "to_list"):
        v = v.to_list()
    elif hasattr(v, "vector"):
        v = list(v.vector)
    a = np.asarray(v, dtype=np.float32)
    return a if a.size else None


def _unit(a: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(a))
    return a / n if n else a


def _parse_date(raw):
    if isinstance(raw, date):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _raw_location_key(loc) -> str | None:
    """Stable key for a moment's RAW location (reverse-geocoded name / GPS),
    independent of the substrate's spot anchoring."""
    if loc is None:
        return None
    for attr in ("name", "location_name", "display_name", "label"):
        v = getattr(loc, attr, None)
        if v:
            return str(v).strip().lower()
    return str(loc).strip().lower() or None


def build_moment_sequence(repos, persona: str) -> list[dict]:
    """Time-ordered moments with unit visual latent, anchored spot label, and
    RAW location key (for the location-conditioned fair baseline)."""
    g = repos.graph
    moments = {m.id: m for m in repos.moment.find_all(limit=20000)}
    dates = g.get_dates_by_moments(list(moments))
    spot_by_id = {s["id"]: s for s in g.get_spots(persona)}
    m2s: dict[str, str] = {}
    for sid in spot_by_id:
        for mid in (g.get_moments_for_spot(sid) or []):
            k = mid if isinstance(mid, str) else getattr(mid, "id", None)
            if k and k not in m2s:
                m2s[k] = sid
    # raw per-moment location (substrate-independent): the panel's "same location
    # conditioning" control — group by GPS/geocoded name, NOT by anchored spot.
    try:
        raw_locs = g.get_locations_by_moments(list(moments))
    except Exception:  # noqa: BLE001
        raw_locs = {}
    items = []
    for mid, m in moments.items():
        d = _parse_date(dates.get(mid))
        v = _vec(m, "visual_embedding")
        if d is None or v is None:
            continue
        items.append({"mid": mid, "date": d, "vec": _unit(v), "spot": m2s.get(mid),
                      "loc": _raw_location_key(raw_locs.get(mid))})
    items.sort(key=lambda x: x["date"])
    return items, spot_by_id


def _spot_name(spot_id, spot_by_id) -> str:
    if not spot_id:
        return "(unanchored)"
    s = spot_by_id.get(spot_id, {})
    return s.get("user_label") or s.get("spot_type") or s.get("location_name") or spot_id


def run_t3(
    persona: str,
    data_root: str,
    output_csv: str | Path,
    holdout_frac: float = DEFAULT_HOLDOUT,
    seed: int = DEFAULT_SEED,
    temporal_k: int = TEMPORAL_K,
    summary_path: str | Path | None = None,
) -> dict:
    from ...interfaces.cli.shared import get_repos

    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    items, spot_by_id = build_moment_sequence(repos, persona)
    n = len(items)
    if n < 20:
        raise RuntimeError(f"Only {n} embedded+dated moments — too few for T3 reconstruction.")

    rng = np.random.default_rng(seed)
    mask_idx = set(rng.choice(n, size=max(1, math.floor(holdout_frac * n)), replace=False).tolist())
    hist = [x for i, x in enumerate(items) if i not in mask_idx]
    mask = [items[i] for i in sorted(mask_idx)]

    global_mean = _unit(np.stack([x["vec"] for x in hist]).mean(axis=0))
    by_spot: dict[str, list[np.ndarray]] = defaultdict(list)
    for x in hist:
        if x["spot"]:
            by_spot[x["spot"]].append(x["vec"])
    proto = {s: _unit(np.stack(v).mean(axis=0)) for s, v in by_spot.items()}
    # raw-location prototypes: same idea as the spot prototype, but grouped by
    # RAW geocoded location (no substrate anchoring). The fair location-conditioned
    # baseline — if the substrate beats this, the *anchoring* adds value beyond GPS.
    by_loc: dict[str, list[np.ndarray]] = defaultdict(list)
    for x in hist:
        if x["loc"]:
            by_loc[x["loc"]].append(x["vec"])
    loc_proto = {lk: _unit(np.stack(v).mean(axis=0)) for lk, v in by_loc.items()}

    hist_sorted = hist  # already time-ordered
    hist_dates = [h["date"] for h in hist_sorted]

    def temporal_pred(item):
        # K visible moments nearest in time (reconstruction context, both sides)
        order = sorted(range(len(hist_sorted)), key=lambda i: abs((hist_dates[i] - item["date"]).days))[:temporal_k]
        return _unit(np.stack([hist_sorted[i]["vec"] for i in order]).mean(axis=0)) if order else global_mean

    def predict(name, item):
        if name == "substrate":
            p = proto.get(item["spot"])
            return (p, True) if p is not None else (global_mean, False)
        if name == "location_raw":
            p = loc_proto.get(item["loc"])
            return (p, p is not None) if p is not None else (global_mean, False)
        if name == "temporal":
            return temporal_pred(item), False
        return global_mean, False

    true_mat = np.stack([x["vec"] for x in mask])
    # The anchored subset is a property of the MASKED MOMENT (does its spot have
    # a prototype from visible history), so every predictor is scored on the
    # same subset — an apples-to-apples "where the substrate can act" view.
    anchored_idx = [j for j, item in enumerate(mask) if item["spot"] in proto]
    rows = []
    per = {}
    for name in ("substrate", "location_raw", "temporal", "global"):
        coss, ranks = [], []
        t1 = t5 = 0
        for j, item in enumerate(mask):
            pred, anchored = predict(name, item)
            cos = float(pred @ item["vec"])
            sims = true_mat @ pred
            rank = int((np.argsort(-sims) == j).nonzero()[0][0]) + 1
            coss.append(cos)
            ranks.append(rank)
            t1 += rank == 1
            t5 += rank <= 5
            rows.append({
                "predictor": name, "mid": item["mid"], "spot": _spot_name(item["spot"], spot_by_id),
                "anchored": int(item["spot"] in proto), "cosine": round(cos, 4), "rank": rank,
                "in_top1": int(rank == 1), "in_top5": int(rank <= 5),
            })
        per[name] = {
            "n": len(mask),
            "cosine": round(statistics.mean(coss), 4),
            "retr_top1": round(t1 / len(mask), 4),
            "retr_top5": round(t5 / len(mask), 4),
            "median_rank": statistics.median(ranks),
            "cosine_anchored": round(statistics.mean(coss[i] for i in anchored_idx), 4) if anchored_idx else None,
            "retr_top1_anchored": round(sum(1 for i in anchored_idx if ranks[i] == 1) / len(anchored_idx), 4) if anchored_idx else None,
        }
    n_anchored = len(anchored_idx)

    headline = {
        "delta_retr_top1_substrate_minus_temporal": round(per["substrate"]["retr_top1"] - per["temporal"]["retr_top1"], 4),
        "delta_retr_top1_substrate_minus_global": round(per["substrate"]["retr_top1"] - per["global"]["retr_top1"], 4),
        # the panel's key control: does the substrate's ANCHORING beat raw GPS/location?
        "delta_retr_top1_substrate_minus_location_raw": round(per["substrate"]["retr_top1"] - per["location_raw"]["retr_top1"], 4),
        "delta_retr_top1_anchored_substrate_minus_location_raw": (
            round(per["substrate"]["retr_top1_anchored"] - per["location_raw"]["retr_top1_anchored"], 4)
            if per["substrate"]["retr_top1_anchored"] is not None and per["location_raw"]["retr_top1_anchored"] is not None else None
        ),
        "substrate_retr_top1_vs_chance_x": round(per["substrate"]["retr_top1"] / (1.0 / len(mask)), 1) if per["substrate"]["retr_top1"] else 0.0,
        "delta_cosine_anchored_substrate_minus_global": (
            round(per["substrate"]["cosine_anchored"] - per["global"]["cosine_anchored"], 4)
            if per["substrate"]["cosine_anchored"] is not None else None
        ),
    }

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["predictor", "mid", "spot", "anchored", "cosine", "rank", "in_top1", "in_top5"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "spec_version": SPEC_VERSION,
        "track": "T3 — world model (masked-moment latent reconstruction)",
        "persona": persona,
        "n_moments": n,
        "n_masked": len(mask),
        "n_masked_with_substrate_anchor": n_anchored,
        "masking": f"random {int(holdout_frac*100)}% (seed={seed})",
        "target": "SigLIP visual embedding (768-d)",
        "random_retr_top1": round(1.0 / len(mask), 4),
        "per_predictor": per,
        "headline_substrate_vs_unconditioned": headline,
        "copresence": "omitted — contacts present on <5% of moments for this persona",
    }
    sp = Path(summary_path) if summary_path else out.with_name(f"{out.stem}-summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    logger.info("T3 wrote %s (+ summary)", out)
    return summary
