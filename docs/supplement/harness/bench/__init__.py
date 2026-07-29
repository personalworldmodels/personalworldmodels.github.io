"""PWM-Bench harness — cross-architecture personal memory substrate benchmark.

See `docs/pwm-bench-spec.md` for the full spec. This package implements the
T1 LLM track (§4): personal-history question answering with LLM-as-judge
evaluation.

Public surface:
- `Substrate` / `SubstrateResponse`: the per-architecture adapter contract.
- `load_tasks(path)`: read a T1 task JSONL file.
- `evaluate(...)`: LLM-as-judge correctness, grounding, hallucination.
- `adapters.pwm.PWMAdapter`: the reference substrate (Golgi).
- `adapters.mem0.Mem0Adapter`: Mem0 baseline (optional dep).
- `adapters.gemini_long_context.GeminiLongContextAdapter`: long-context baseline.
"""
from .loader import T1Task, load_tasks
from .substrate import Substrate, SubstrateResponse

__all__ = [
    "Substrate",
    "SubstrateResponse",
    "T1Task",
    "load_tasks",
]
