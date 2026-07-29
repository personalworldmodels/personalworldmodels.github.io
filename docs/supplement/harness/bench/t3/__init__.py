"""PWM-Bench T3 — world-model track (masked-moment latent reconstruction).

The T1 track shows the substrate helps an LLM consumer; T3 tests whether the
same substrate helps a world-model-class consumer — one that reconstructs the
texture of an unobserved moment in representation space (LeCun/JEPA), rather
than guessing a discrete next label. See reconstruction.py for the rationale.
"""
from .reconstruction import build_moment_sequence, run_t3

__all__ = ["build_moment_sequence", "run_t3"]
