"""Substrate Protocol — what every T1 adapter must implement.

A `Substrate` answers a natural-language question about a persona's archive
and returns a `SubstrateResponse` containing an answer string, the moment IDs
the answer is grounded in (for §4.5 source-grounding evaluation), elapsed
wall-clock latency, and an optional token count.

This contract is intentionally minimal — it is the cross-architecture
interface boundary. Per-adapter setup (loading the persona corpus into Mem0,
warming an LLM, etc.) happens in `__init__`; per-query work happens in
`query()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


def synth_config(model_name: str, temperature: float = 0.2):
    """LLMConfig for benchmark answer synthesis, model-aware.

    Cloud frontier models need a generous output budget (8192) so reasoning
    models' thinking tokens don't truncate the answer. On-device 4B models
    (qwen/gemma/... via Ollama) are far slower per token and produce short
    answers, so an 8192 budget wastes minutes per query on a runaway thinking
    trace — cap them at 2048. Override with PWMBENCH_MAX_OUTPUT_TOKENS.
    """
    import os

    from ..domain.ports.infra.llm import LLMConfig

    on_device = (":" in model_name) or model_name.lower().startswith(
        ("qwen", "gemma", "llama", "mistral", "phi")
    )
    default = 2048 if on_device else 8192
    mx = int(os.environ.get("PWMBENCH_MAX_OUTPUT_TOKENS", default))
    return LLMConfig(temperature=temperature, max_output_tokens=mx)


@dataclass
class SubstrateResponse:
    """Per-query output from any T1 substrate adapter."""

    answer: str
    cited_moment_ids: list[str] = field(default_factory=list)
    # Optional: the TEXT of the evidence the substrate retrieved/cited. Used by
    # the grounding judge for systems that can't expose source moment IDs (e.g.
    # Zep returns extracted facts, not moments). When empty, the runner falls
    # back to resolving cited_moment_ids → descriptions.
    cited_texts: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    tokens_used: int | None = None
    # Free-form per-adapter diagnostic payload (channel scores, retrieval
    # counts, etc.). Not consumed by the evaluator; written to logs only.
    debug: dict | None = None


@runtime_checkable
class Substrate(Protocol):
    """T1 substrate adapter protocol.

    Implementations: see `golgi.bench.adapters.{pwm,mem0,gemini_long_context}`.
    """

    def name(self) -> str:
        """Stable identifier for this substrate (CSV column + summary key)."""
        ...

    def query(self, q: str, persona: str) -> SubstrateResponse:
        """Answer `q` about `persona`'s archive, cite moment IDs if grounded."""
        ...
