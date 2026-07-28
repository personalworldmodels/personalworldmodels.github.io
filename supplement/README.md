# Personal World Models — Released Materials

Companion materials for *Personal World Models* (Azab & Benavente, Kinship
Technologies, 2026). Paper and runnable package: https://personalworldmodels.github.io

## Contents

| Path | Backs paper section |
|---|---|
| `rubrics/judge_describe.md` | Appendix A — visual-description judge rubric |
| `rubrics/judge_features.md` | Appendix A — feature-extraction judge rubric |
| `rubrics/judge_plans.md` | Appendix A — narrative-synthesis judge rubric |
| `rubrics/t1-judge-rubric.md` | §4.3 — agent-benchmark correctness/grounding rubric |
| `stage_io_reference.md` | Appendix A — per-stage input/output reference, cloud vs. on-device |
| `prompts.md` | Appendix A — the four ingestion-stage prompts, verbatim |
| `configs.md` | Appendix A — model, quantization (Q4_K_M via Ollama), and decoding configurations; Anchor and Routine promotion thresholds |
| `harness/bench/` | §4.3–4.4 — benchmark harness: runner, judge (`evaluate.py`), system adapters (Mem0, Zep, Supermemory, GraphRAG, flat-retrieval controls), T3 reconstruction |
| `harness/scripts/` | T3 DINOv2 replication, vision baseline, read-by-any-model driver |

Boundary segmentation has no LLM judge rubric by design: it is scored by exact
match (SAME/DIFFERENT) against ground truth (see `stage_io_reference.md`).

Worked examples in the rubrics use invented names and places; scoring criteria
are unchanged. Code defaults may postdate the paper's runs — set reference and
judge models explicitly (see `configs.md`) to match the configurations reported.

## What is not released

The evaluation archive is the personal photo library of one of the authors and
is not released, nor are the benchmark question set and per-question outputs
derived from it (see the paper's Ethics Statement). The intended reproduction
is to run the released pipeline and harness on your own camera roll.

## Running the pipeline

The GOLGI package (macOS arm64, Python 3.12) is downloadable from the site
above; it ingests a photo directory end-to-end in cloud (Gemini) or on-device
(Ollama) mode and exposes the committed structure over MCP.
