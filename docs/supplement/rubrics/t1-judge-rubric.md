# PWM-Bench T1 - LLM-as-Judge Rubric

**Spec ref:** §4.5 of `docs/pwm-bench-spec.md` (v0.5, target lock 2026-06-15).
**Question set:** `docs/research/pwm-bench-t1-hana-questions.jsonl` (50 questions, stratified across the 5 categories defined in §4.3).
**Persona:** `hana` (686 moments, 855 media files, April 2025 - May 2026, photo modality).

This rubric is normative for v1.0. It is consumed by the T1 harness (Agent A's scaffold, tracked under task #44) when scoring system answers against reference answers.

---

## 1. Answer correctness (1-4 scale)

The judge model assigns a single integer score to each system answer, comparing it against the human-authored reference answer plus the supplied reference moment descriptions.

- **4 - Fully correct & grounded.** The answer is factually correct given the reference, AND cites moments (via `cited_moment_ids`) that substantively support the claim. The system gets the names, places, dates, or counts right, and the citations point to moments whose descriptions visibly back the claim.
- **3 - Mostly correct, weak grounding.** The answer is correct or very close to the reference, but citations are missing, partial, or only loosely related. A reader who trusted the answer would not be misled, but cannot verify it from the citations alone.
- **2 - Partial / vague.** The answer captures the general direction of the reference but misses specifics (e.g., names a single instance when the question asks about a recurring pattern, or gives a generic answer that would be true of many people).
- **1 - Wrong / unsupported.** The answer contradicts the reference, fabricates details (people, places, dates not in the corpus), or has no factual relationship to the question.

**Normalization to 0-100 per spec §4.5:**

```
score_100 = (raw - 1) * 100 / 3
```

So raw scores map to: 1 → 0, 2 → 33.3, 3 → 66.7, 4 → 100.

The **primary T1 metric** is mean `score_100` across the 50 questions, reported per LLM model and per substrate (Golgi, Mem0, Zep, Vanilla RAG).

---

## 2. Source grounding (binary, per question)

A question is **source-grounded** if either:
- ≥1 of the system's `cited_moment_ids` appears in the question's `reference_moment_ids`, OR
- The judge determines that ≥1 cited moment substantively supports the answer (judge has access to cited-moment descriptions and the reference moments).

Aggregate metric: fraction of the 50 questions that are source-grounded. **Success criterion S1.2 (locked):** ≥70%.

Rationale for the disjunction: PWM-Bench question authors cannot enumerate every supporting moment in a 686-moment archive; a system citation that is semantically valid but lies outside the author-curated reference set should not be penalised.

---

## 3. Hallucination (binary, per response)

A response is **hallucinatory** if it contains ≥1 specific factual claim - a name, place, date, count, or quoted text - that **no moment in the supplied corpus subset substantively supports**. The judge has access to:
- The system's full answer text.
- The system's `cited_moment_ids` with descriptions.
- The reference moments with descriptions.

Vague claims ("I often work on hardware") are not hallucinatory by default; they are scored under correctness (likely 2 - vague).

Aggregate metric: hallucination rate = fraction of responses flagged. **Success criterion S1.3 (locked):** ≤10%.

---

## 4. Judge instructions (prompt schema)

The judge model receives a single JSON payload per question:

```json
{
  "question": "...",
  "reference_answer": "...",
  "reference_moments": [
    {"id": "uuid", "description": "..."},
    ...
  ],
  "system_answer": "...",
  "system_cited_moments": [
    {"id": "uuid", "description": "..."},
    ...
  ]
}
```

The judge returns a single JSON object per question:

```json
{
  "correctness": 1 | 2 | 3 | 4,
  "grounded": true | false,
  "hallucinated": true | false,
  "rationale": "1-3 sentences explaining the scores"
}
```

The `rationale` is required so that judge decisions are auditable when contested.

---

## 5. Recommended judge model

- **Runs:** `gemini-2.5-flash` (cheap, fast, sufficient for the rubric).
- **Stability re-judging:** `claude-sonnet-4-6` on a stratified 20-question sample (4 per category) to validate `gemini-2.5-flash` correctness scores within ±0.5 raw points on average.

Per spec §10.1, the judge model identifier and its full prompt template must be published in supplementary materials when reporting results.

---

## 6. Token efficiency (reported, not gated)

Per spec §4.5, total tokens per query (substrate query + LLM input + LLM output) is reported as a distribution across the 50 questions. No gating threshold in v1.0; it is included to support cross-substrate token-cost comparisons (Mem0 vs Zep vs Golgi vs Vanilla RAG).

---

## 7. Per-category reporting

Results are reported both as a single mean `score_100` and stratified by question category, since the success criterion S1.1 is gated on ≥3 of 5 categories beating the best baseline:

- `recurring_patterns` (q_001 - q_010)
- `lifetime_spans` (q_011 - q_020)
- `multi_source` (q_021 - q_030)
- `counterfactual` (q_031 - q_040)
- `open_ended` (q_041 - q_050)
