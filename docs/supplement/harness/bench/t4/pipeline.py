"""PWM-Bench T4 — diffusion track runner (exploratory tier, spec §7).

Wires the substrate's entity-reference surface (`substrate_refs.py`) into a
diffusion model and produces, for each anchored entity, TWO generations:

- **substrate-conditioned** — IP-Adapter on the entity's *real* reference photo
  (the substrate's representative imagery) + the relational prompt §7.3.
- **unconditioned** — the same prompt, IP-Adapter scale 0 (prompt only, no
  reference). The §7.6 baseline (unconditioned SDXL) made local.

If the substrate's entity-anchoring + reference-imagery path is real, the
conditioned generation should preserve the entity's identity (higher CLIP-I vs
the reference photo) while the unconditioned one renders a generic stranger.

Model: SDXL + IP-Adapter is the spec reference (§7.2). On Apple-Silicon MPS SDXL
is ~7 GB and minutes/image; the exploratory tier needs only 1–3 demos, so the
default is the lighter, well-supported **SD 1.5 + IP-Adapter** (an "equivalent
diffusion model", explicitly permitted by §7.7). Pass model="sdxl" to use the
spec reference where time/hardware allow. Both go through one code path.

Optional metrics (§7.5), computed with the substrate's own SigLIP encoder so the
identity space matches what the substrate indexes:
- **CLIP-I**: cosine(generated, reference photo) — identity preservation.
- **CLIP-T**: cosine(generated, prompt text) — prompt following.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

SPEC_VERSION = "1.0-draft"

# Model presets: (repo, ip_adapter_repo, ip_subfolder, ip_weight, is_sdxl)
_MODELS = {
    "sd15": (
        "runwayml/stable-diffusion-v1-5",
        "h94/IP-Adapter", "models", "ip-adapter_sd15.bin", False,
    ),
    "sdxl": (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.bin", True,
    ),
}

_NEG_PROMPT = "blurry, deformed, distorted, low quality, extra limbs, watermark, text"


# --------------------------------------------------------------------------- #
# Diffusion pipeline (lazy — only imported/loaded when actually generating)
# --------------------------------------------------------------------------- #
class _DiffusionRunner:
    def __init__(self, model: str = "sd15", device: str | None = None, steps: int = 30):
        import torch

        if model not in _MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {list(_MODELS)}")
        repo, ip_repo, ip_sub, ip_weight, is_sdxl = _MODELS[model]
        self.model = model
        self.is_sdxl = is_sdxl
        self.steps = steps
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(repo, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                repo, torch_dtype=dtype, safety_checker=None
            )
        pipe = pipe.to(self.device)
        pipe.load_ip_adapter(ip_repo, subfolder=ip_sub, weight_name=ip_weight)
        self.pipe = pipe
        self._torch = torch

    def generate(
        self,
        prompt: str,
        reference_image,
        ip_scale: float,
        seed: int,
        size: int,
    ):
        from PIL import Image

        self.pipe.set_ip_adapter_scale(ip_scale)
        gen = self._torch.Generator(device="cpu").manual_seed(seed)
        # IP-Adapter always needs an ip_adapter_image arg; ip_scale=0 makes it
        # a no-op, giving the clean unconditioned baseline through one code path.
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")
        kwargs = dict(
            prompt=prompt,
            negative_prompt=_NEG_PROMPT,
            ip_adapter_image=reference_image,
            num_inference_steps=self.steps,
            generator=gen,
        )
        if self.is_sdxl:
            kwargs.update(height=size, width=size)
        else:
            kwargs.update(height=size, width=size)
        out = self.pipe(**kwargs)
        return out.images[0]


# --------------------------------------------------------------------------- #
# CLIP-I / CLIP-T via the substrate's SigLIP encoder
# --------------------------------------------------------------------------- #
def _siglip():
    from ...infrastructure.embedding import SigLIPEmbedder
    return SigLIPEmbedder()


def _cos(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return 0.0
    return float(a @ b / (na * nb))


def _embed_image(emb, path: Path):
    return emb.embed_image(Path(path)).to_list()


def _embed_text(emb, text: str):
    return emb.embed_text(text).to_list()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_t4(
    persona: str,
    data_root: str,
    output_dir: str | Path,
    model: str = "sd15",
    limit: int = 3,
    steps: int = 30,
    size: int = 512,
    ip_scale: float = 0.7,
    seed: int = 0,
    compute_metrics: bool = True,
    summary_path: str | Path | None = None,
) -> dict:
    """Run the T4 diffusion track for one persona.

    Produces, per entity reference, a substrate-conditioned and an unconditioned
    generation, saves both, and (optionally) reports CLIP-I / CLIP-T. Returns the
    spec-shaped summary dict and writes it as JSON.
    """
    from ...interfaces.cli.shared import get_repos
    from .substrate_refs import build_entity_references

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    repos = get_repos(persona=persona, data_root=data_root, read_only=True)
    refs = build_entity_references(repos, persona, limit=limit)
    if not refs:
        raise RuntimeError(
            f"No anchored entity references with reference imagery for persona "
            f"{persona!r} — the substrate exposed no usable entity+photo pairs."
        )

    logger.info("T4: built %d entity references; loading %s ...", len(refs), model)
    runner = _DiffusionRunner(model=model, steps=steps)

    emb = _siglip() if compute_metrics else None
    generations = []

    for i, ref in enumerate(refs):
        ref_path = ref.reference_image_path
        stem = f"{i:02d}_{ref.entity_kind}_{_slug(ref.entity_label)}"
        cond_path = out / f"{stem}_conditioned.png"
        uncond_path = out / f"{stem}_unconditioned.png"

        logger.info("T4 [%d/%d] %r — prompt: %s", i + 1, len(refs), ref.entity_label, ref.prompt)

        t0 = time.time()
        img_cond = runner.generate(ref.prompt, ref_path, ip_scale=ip_scale, seed=seed, size=size)
        img_cond.save(cond_path)
        t_cond = time.time() - t0

        t1 = time.time()
        img_uncond = runner.generate(ref.prompt, ref_path, ip_scale=0.0, seed=seed, size=size)
        img_uncond.save(uncond_path)
        t_uncond = time.time() - t1

        rec = {
            "index": i,
            "entity_kind": ref.entity_kind,
            "entity_label": ref.entity_label,
            "other_entity": ref.other_entity,
            "activity": ref.activity,
            "prompt": ref.prompt,
            "reference_image": str(ref_path),
            "conditioned_image": str(cond_path),
            "unconditioned_image": str(uncond_path),
            "ip_scale": ip_scale,
            "seconds_conditioned": round(t_cond, 1),
            "seconds_unconditioned": round(t_uncond, 1),
        }

        if emb is not None:
            ref_vec = _embed_image(emb, ref_path)
            cond_vec = _embed_image(emb, cond_path)
            uncond_vec = _embed_image(emb, uncond_path)
            txt_vec = _embed_text(emb, ref.prompt)
            rec["metrics"] = {
                "clip_i_conditioned": round(_cos(cond_vec, ref_vec), 4),
                "clip_i_unconditioned": round(_cos(uncond_vec, ref_vec), 4),
                "clip_t_conditioned": round(_cos(cond_vec, txt_vec), 4),
                "clip_t_unconditioned": round(_cos(uncond_vec, txt_vec), 4),
            }
            rec["metrics"]["clip_i_delta_cond_minus_uncond"] = round(
                rec["metrics"]["clip_i_conditioned"] - rec["metrics"]["clip_i_unconditioned"], 4
            )
        generations.append(rec)
        logger.info("T4 [%d/%d] saved (%.0fs cond / %.0fs uncond)", i + 1, len(refs), t_cond, t_uncond)

    summary = _summarize(persona, model, size, steps, ip_scale, seed, refs, generations)
    sp = Path(summary_path) if summary_path else out / "t4-summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    logger.info("T4 wrote %d demo pairs + summary to %s", len(generations), out)
    return summary


def _summarize(persona, model, size, steps, ip_scale, seed, refs, generations) -> dict:
    metricked = [g for g in generations if "metrics" in g]
    headline = None
    if metricked:
        import statistics
        headline = {
            "mean_clip_i_conditioned": round(
                statistics.mean(g["metrics"]["clip_i_conditioned"] for g in metricked), 4),
            "mean_clip_i_unconditioned": round(
                statistics.mean(g["metrics"]["clip_i_unconditioned"] for g in metricked), 4),
            "mean_clip_i_delta": round(
                statistics.mean(g["metrics"]["clip_i_delta_cond_minus_uncond"] for g in metricked), 4),
            "mean_clip_t_conditioned": round(
                statistics.mean(g["metrics"]["clip_t_conditioned"] for g in metricked), 4),
            "mean_clip_t_unconditioned": round(
                statistics.mean(g["metrics"]["clip_t_unconditioned"] for g in metricked), 4),
        }
    return {
        "spec_version": SPEC_VERSION,
        "track": "T4 — diffusion (personalized image generation), exploratory tier",
        "persona": persona,
        "model": model,
        "diffusion_model": _MODELS[model][0],
        "ip_adapter": _MODELS[model][1] + "/" + _MODELS[model][3],
        "image_size": size,
        "inference_steps": steps,
        "ip_adapter_scale": ip_scale,
        "seed": seed,
        "n_entity_references": len(refs),
        "n_demo_pairs": len(generations),
        "conditions": ["substrate-conditioned (IP-Adapter + relational prompt)",
                       "unconditioned (prompt only, ip_scale=0)"],
        "entity_references": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(r).items() if k != "debug"}
            for r in refs
        ],
        "generations": generations,
        "headline_clip_i_substrate_vs_unconditioned": headline,
        "metrics_note": (
            "CLIP-I/CLIP-T computed with the substrate's SigLIP encoder "
            "(ViT-B-16-SigLIP). Exploratory tier — reported, not gated (§7.5)."
        ),
    }


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "entity"
