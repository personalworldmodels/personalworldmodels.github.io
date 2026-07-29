"""PWM-Bench T4 — diffusion track (personalized image generation, §7).

Exploratory tier (§3): demonstrates that the substrate's entity-anchoring +
reference-imagery path wires into a diffusion model and produces
identity-preserving personalized generations — substrate-conditioned (IP-Adapter
on the entity's real reference photo + relational prompt) vs unconditioned.

- `build_entity_references`: the substrate→diffusion bridge (entity + reference
  image + "<entity> at <other-entity> doing <activity>" prompt).
- `run_t4`: run both conditions, save images, optionally report CLIP-I/CLIP-T.
"""
from .substrate_refs import EntityReference, build_entity_references

__all__ = ["EntityReference", "build_entity_references", "run_t4"]


def run_t4(*args, **kwargs):
    """Lazy passthrough so importing the package doesn't pull in torch/diffusers."""
    from .pipeline import run_t4 as _run_t4
    return _run_t4(*args, **kwargs)
