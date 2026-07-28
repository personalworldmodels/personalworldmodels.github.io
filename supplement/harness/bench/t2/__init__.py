"""PWM-Bench T2 — neurosymbolic track (graph-query execution).

Tests whether the substrate's EXPLICIT symbolic structure lets a query-execution
agent answer exact-computation questions (count / multi-hop / superlative) that a
strong LLM reading the same facts as flattened text fails at. Reframed from VIGA
scene reconstruction; see docs/pwm-bench-t2-assessment.md.
"""
from .runner import run_t2

__all__ = ["run_t2"]
