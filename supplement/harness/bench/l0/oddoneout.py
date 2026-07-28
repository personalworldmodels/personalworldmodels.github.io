"""L0 odd-one-out probe: does the substrate's concept beat raw perception?

Given three instances (two of concept A, one of concept B), pick the odd one.
Two solvers on the SAME triplets isolate the substrate's contribution:
- embedding_only: pure perception — odd = least similar to the other two's
  centroid (raw SigLIP, no memory).
- substrate_concept: embedding+graph — each instance is assigned to its nearest
  CONCEPT PROTOTYPE built from HELD-OUT instances (Posner-Keele generalization,
  not in-sample clustering); odd = the minority concept.

Concepts are drawn from the substrate's anchors. `concept="identity"` (who is
present, via graph contacts) transcends appearance and is the valid test;
`concept="place"` is perceptual and serves as the redundancy control.
"""
from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict

import numpy as np

from ..t3.reconstruction import _unit, _vec

logger = logging.getLogger(__name__)


def _concept_members(repos, persona: str, concept: str) -> dict[str, list[np.ndarray]]:
    """concept_id -> list of unit visual embeddings of its member moments."""
    g = repos.graph
    moments = {m.id: m for m in repos.moment.find_all(limit=20000)}
    vec = {}
    for mid, m in moments.items():
        v = _vec(m, "visual_embedding")
        if v is not None:
            vec[mid] = _unit(v)

    members: dict[str, list[np.ndarray]] = defaultdict(list)
    if concept == "place":
        for sid in {s["id"] for s in g.get_spots(persona)}:
            for mid in (g.get_moments_for_spot(sid) or []):
                k = mid if isinstance(mid, str) else getattr(mid, "id", None)
                if k in vec:
                    members[sid].append(vec[k])
    elif concept == "identity":
        try:
            self_id = g.get_persona_self(persona)
        except Exception:
            self_id = None
        contacts = g.find_contacts_by_moments(list(moments))
        for mid, cs in contacts.items():
            for cid in (cs or []):
                if cid and cid != self_id and mid in vec:
                    members[cid].append(vec[mid])
    else:
        raise ValueError(f"unknown concept {concept!r}")
    return members


def run_oddoneout(
    persona: str, data_root: str, concept: str = "identity",
    min_members: int = 3, n_triplets: int = 400, seed: int = 0,
) -> dict:
    from ...interfaces.cli.shared import get_repos

    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    members = {c: v for c, v in _concept_members(repos, persona, concept).items() if len(v) >= min_members}
    if len(members) < 2:
        return {"concept": concept, "n_concepts": len(members), "error": "too few concepts for odd-one-out"}

    rng = random.Random(seed)
    proto, test = {}, {}
    for cid, vs in members.items():
        v = vs[:]
        rng.shuffle(v)
        h, t = v[: len(v) // 2], v[len(v) // 2:]
        if h and t:
            proto[cid] = _unit(np.mean(h, axis=0))   # prototype from HELD-OUT half
            test[cid] = t
    pids = list(proto)
    P = np.stack([proto[c] for c in pids])
    nearest = lambda v: pids[int(np.argmax(P @ v))]

    pool = [c for c in test if len(test[c]) >= 2]
    triplets = []
    for _ in range(n_triplets):
        if len(pool) < 2:
            break
        a, b = rng.sample(pool, 2)
        a1, a2 = rng.sample(test[a], 2)
        triplets.append((a1, a2, rng.choice(test[b])))   # gold odd index = 2

    def emb_only(t):
        sc = [float(_unit(np.mean([t[j] for j in range(3) if j != i], axis=0)) @ t[i]) for i in range(3)]
        return int(np.argmin(sc))

    def substrate(t):
        labs = [nearest(t[0]), nearest(t[1]), nearest(t[2])]
        c = Counter(labs)
        for i, l in enumerate(labs):
            if c[l] == 1:
                return i
        return -1

    acc = lambda f: round(sum(f(t) == 2 for t in triplets) / len(triplets), 3) if triplets else 0.0
    e, s = acc(emb_only), acc(substrate)
    return {
        "concept": concept,
        "n_concepts": len(members),
        "n_triplets": len(triplets),
        "embedding_only": e,
        "substrate_concept": s,
        "delta_substrate_minus_embedding": round(s - e, 3),
        "chance": round(1 / 3, 3),
    }


def run_l0(
    persona: str,
    data_root: str,
    concepts=("identity", "place"),
    output=None,
    seed: int = 0,
) -> dict:
    """Run L0 over the given concepts and (optionally) write a summary JSON.

    Mirrors the run_t2/run_t3 pattern: returns the summary dict and, if `output`
    is given, writes it there. Δ>0 on a non-perceptual concept (identity) is the
    mental-model-fidelity signal; place is the perceptual redundancy control.
    """
    import json
    from pathlib import Path

    from .keystone import run_keystone

    per = {c: run_oddoneout(persona, data_root, concept=c, seed=seed) for c in concepts}
    keystone = run_keystone(persona, data_root, seed=seed)
    summary = {
        "spec_version": "1.0-draft",
        "track": "L0 — mental-model fidelity (odd-one-out)",
        "persona": persona,
        "concepts": list(concepts),
        "per_concept": per,
        "keystone": keystone,
        "headline": {
            **{f"delta_{c}": per[c].get("delta_substrate_minus_embedding")
               for c in concepts if not per[c].get("error")},
            "keystone_pearson_l0_vs_t3": keystone.get("pearson_l0_vs_t3"),
        },
        "note": (
            "delta = substrate_concept - embedding_only. >0 means the substrate's "
            "concept beats raw perception (mental-model fidelity); expected positive "
            "on non-perceptual concepts (identity), negative on perceptual (place)."
        ),
    }
    if output:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2))
    return summary
