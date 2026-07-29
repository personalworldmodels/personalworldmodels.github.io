"""Critique #3 — measurement overlap. Re-run T3 in an INDEPENDENT encoder space.

SigLIP is the substrate's latents AND the T3 scoring space, so T3's win could be
SigLIP self-consistency ("SigLIP groups places") rather than episodic structure.
We re-embed every photo with DINOv2 (image-only self-supervised, no text, no
relation to SigLIP), build spot prototypes from DINOv2 vectors (spots come from
location/graph, NOT SigLIP), and predict a masked moment's DINOv2 embedding,
scored by retrieval in DINOv2 space. If the substrate still beats global/temporal
here, the T3 advantage is the structure, not the encoder. If not, it was an
artifact. Honest find-out.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from golgi.bench.t3.reconstruction import _parse_date
from golgi.interfaces.cli.shared import get_repos

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def load_dino():
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    return proc, model


def embed_photos(paths, proc, model, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        inp = proc(images=imgs, return_tensors="pt").to(DEV)
        with torch.no_grad():
            v = model(**inp).pooler_output  # CLS-pooled, 768-d, DINOv2
        out.append(v.float().cpu().numpy())
    return np.concatenate(out)


def load_items(persona, dr):
    r = get_repos(persona=persona, data_root=dr, read_only=True)
    g = r.graph
    photo_dir = f"{dr}/{persona}/photo"
    idx = {f: os.path.join(photo_dir, f) for f in os.listdir(photo_dir)} if os.path.isdir(photo_dir) else {}
    moments = {m.id: m for m in r.moment.find_all(limit=20000)}
    dates = g.get_dates_by_moments(list(moments))
    m2s = {}
    for s in g.get_spots(persona):
        for mid in (g.get_moments_for_spot(s["id"]) or []):
            k = mid if isinstance(mid, str) else getattr(mid, "id", None)
            if k and k not in m2s:
                m2s[k] = s["id"]
    items = []
    for mid, m in moments.items():
        ph = None
        if getattr(m, "media_id", None):
            md = r.media.find_by_id(m.media_id)
            b = os.path.basename(str(getattr(md, "path", ""))) if md else ""
            ph = idx.get(b)
        sid = m2s.get(mid)
        if ph and sid is not None:
            items.append({"mid": mid, "photo": ph, "spot": sid, "date": _parse_date(dates.get(mid))})
    return items


def run(persona, dr, proc, model, holdout=0.2, seed=0):
    items = load_items(persona, dr)
    vecs = embed_photos([x["photo"] for x in items], proc, model)
    for x, v in zip(items, vecs):
        x["vec"] = _unit(v)
    n = len(items)
    rng = np.random.default_rng(seed)
    mask = set(rng.choice(n, max(1, math.floor(holdout * n)), replace=False).tolist())
    hist = [items[i] for i in range(n) if i not in mask]
    test = [items[i] for i in sorted(mask)]
    tmat = np.stack([t["vec"] for t in test])

    by = defaultdict(list)
    for x in hist:
        by[x["spot"]].append(x["vec"])
    proto = {s: _unit(np.mean(v, 0)) for s, v in by.items()}
    gmean = _unit(np.stack([x["vec"] for x in hist]).mean(0))

    def temporal_pred(x):
        if x["date"] is None:
            return gmean
        near = sorted((h for h in hist if h["date"]), key=lambda h: abs((h["date"] - x["date"]).days))[:5]
        return _unit(np.mean([h["vec"] for h in near], 0)) if near else gmean

    def retr(preds):
        return round(sum(p is not None and int(np.argmax(tmat @ p)) == j
                         for j, p in enumerate(preds)) / len(test), 3)

    sub = [proto.get(x["spot"]) for x in test]
    tmp = [temporal_pred(x) for x in test]
    glb = [gmean] * len(test)
    return {"persona": persona, "encoder": "DINOv2", "n_test": len(test), "n_train": len(hist),
            "substrate": retr(sub), "temporal": retr(tmp), "global": retr(glb)}


if __name__ == "__main__":
    proc, model = load_dino()
    print(f"DINOv2 on {DEV}")
    print(f"{'persona':8}{'n_test':>7}{'substrate':>11}{'temporal':>10}{'global':>8}  vs SigLIP(sub/glob)")
    ref = {"hana": "0.212/0.019", "Maria": "0.500/0.100"}
    for persona, dr in [("hana", "data"), ("Maria", "data")]:
        o = run(persona, dr, proc, model)
        print(f"{persona:8}{o['n_test']:>7}{o['substrate']:>11}{o['temporal']:>10}{o['global']:>8}  ({ref.get(persona,'')})")
    print("\nKEY: substrate >> global in DINOv2 space => T3 is episodic structure, not SigLIP self-consistency.")
