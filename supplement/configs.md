# GOLGI Model, Decoding, and Threshold Configurations

## Models

- Cloud reference: Gemini 2.0 Flash (paper runs; current code default is `gemini-2.5-flash` — pin explicitly to reproduce)
- On-device: qwen3-vl:4b and gemma3:4b via Ollama, Q4_K_M quantization
- Agent benchmark (§4.3): answer model Gemini 3.5 Flash, judge Gemini 2.5 Flash

## Decoding configurations (per classifier)

| Stage | temperature | max_output_tokens | response type |
|---|---|---|---|
| LLMConfig | 0.3 | 2048 | None |
| ACTIVITY_FILTER | 0.1 | 512 | application/json |
| ACTIVITY_SELECTOR | 0.3 | 150 | application/json |
| BOUNDARY_DISAMBIGUATOR | 0.0 | 10 | None |
| BOUNDARY_DISAMBIGUATOR_ON_DEVICE | 0.0 | 10 | None |
| BOUNDARY_DISAMBIGUATOR_THINKING | 0.0 | 64 | None |
| PLAN_DISAMBIGUATOR_ON_DEVICE | 0.0 | 128 | application/json |
| PLAN_DISAMBIGUATOR_ON_DEVICE_THINKING | 0.0 | 256 | None |
| CHRONICLE_DAY_ON_DEVICE | 0.3 | 256 | application/json |
| CHRONICLE_DAY_ON_DEVICE_THINKING | 0.3 | 512 | None |
| CHRONICLE_SUMMARIZER | 0.7 | 300 | None |
| COLLECTION_SUMMARIZER | 0.7 | 300 | None |
| EVENT_DISAMBIGUATOR | 0.0 | 200 | None |
| PLAN_CLUSTERER | 0.3 | 2048 | None |
| PLAN_MEMORABILITY | 0.0 | 256 | application/json |
| PLAN_FOCUS_FILTER | 0.0 | 2048 | application/json |
| PLAN_SUMMARIZE | 0.3 | 512 | application/json |
| PLAN_COVER_SELECTOR | 0.2 | 512 | None |
| PLAN_COVER_SELECTOR_SMALL | 0.0 | 128 | application/json |
| PLAN_DISAMBIGUATOR | 0.0 | 100 | None |
| PLAN_HIERARCHY_CLUSTERER | 0.3 | 2048 | None |
| PROJECT_CLASSIFIER | 0.3 | 512 | None |
| PROJECT_CONVERGENCE | 0.0 | 256 | application/json |
| PROJECT_FILTER | 0.0 | 512 | application/json |
| PROJECT_LABEL | 0.3 | 128 | application/json |
| ROUTINE_CLASSIFIER | 0.3 | 512 | None |
| ROUTINE_COHERENCE | 0.0 | 256 | application/json |
| ROUTINE_FILTER | 0.0 | 512 | application/json |
| ROUTINE_LABEL | 0.3 | 128 | application/json |
| CONTACT_CLASSIFIER | 0.3 | 512 | None |
| SELF_CLASSIFIER | 0.3 | 1024 | None |
| SPOTS_CLASSIFIER | 0.3 | 1024 | None |
| INTEREST_CLASSIFIER | 0.2 | 1024 | application/json |
| QUERY_UNDERSTANDER | 0.0 | 2048 | application/json |
| VIBE_EXTRACTOR | 0.0 | 2048 | application/json |
| FACET_EXTRACTOR | 0.0 | 2048 | application/json |

Benchmark synthesis: temperature 0.2; max_output_tokens 8192 (cloud) / 2048 (on-device).

## Promotion thresholds and stage heuristics

| Stage | Heuristic | Thresholds |
|---|---|---|
| `load.burst` | Groups visually similar photos within a short time window — visual similarity ≥0.85 + timestamp ≤20s | `VISUAL_THRESHOLD=0.85`, `TIME_WINDOW_SECONDS=20` |
| `primitives.faces` | Groups similar faces using InsightFace embeddings — clusters become candidate contacts | — |
| `anchors.contacts` | Leiden algorithm clusters face embeddings into identity groups, then a frequency filter (2+ faces across 2+ distinct days) removes one-off appearances before LLM validation | `MIN_FACES=2`, `MIN_DISTINCT_DAYS=2` |
| `anchors.spots` | Groups moments by location proximity, then requires at least 3 moments across 2+ different days — filters out places you just passed through before LLM classification | `MIN_MOMENTS=3`, `MIN_VISIT_DAYS=2` |
| `anchors.home` | Scores locations by night/weekend presence, consecutive stays, and recency — no LLM, purely algorithmic | `MIN_MOMENTS=10`, `SCORE_THRESHOLD=0.3`, `MIN_SPAN_DAYS=7` |
| `anchors.self` | Ranks face clusters by timeline span (35%), selfie ratio (25%), home presence (20%), and co-occurrence diversity (20%) — top-3 candidates go to LLM for final identification | `MIN_FACES=5`, `MIN_DISTINCT_DAYS=3` |
| `narrative.plans` | Groups moments by activity, time, location, and description similarity using Leiden algorithm — LLM then names each group | `MIN_SIMILARITY=0.2`, `RESOLUTION=1.0` |
| `narrative.collections` | Finds repeating patterns via signature similarity and Leiden clustering — validates cyclical recurrence before LLM labeling | `MIN_SPAN_DAYS=14` |
| `narrative.chronicles` | Builds full temporal hierarchy: day → time-of-day → week → weekend → month → quarter → year. Splits using time gaps (8h+), location jumps (50km+), and visual discontinuity — ambiguous splits go to LLM | `HARD_SPLIT_GAP_HOURS=8`, `LOCATION_JUMP_KM=50` |

Routine promotion additionally uses signature weights (time-slot 0.15, day-pattern 0.10, activity 0.40, spatial 0.20, plan-title 0.15), minimum signature similarity 0.51, and cyclical-recurrence gates on interval regularity; see the paper's §3 and the GOLGI package.
