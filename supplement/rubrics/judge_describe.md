# Judge: load.describe

Evaluate on-device model photo descriptions against cloud ground truth.

## Instructions

1. Read the JSONL file provided by the user
2. Skip the first line (`_type: "meta"`) — it's metadata
3. For each result line, compare `cloud_result.description` vs `device_output.description`
4. **Look at the actual photo** at `image_path` to ground your assessment
5. Score each sample on a 1-4 scale (see rubric below)
6. Write a summary report at the end

## JSONL Structure

Each result line has:
- `sample_id` — unique photo ID
- `image_path` — path to the actual photo (read it!)
- `cloud_result.description` — Gemini description (reference, not necessarily perfect)
- `device_output.description` — on-device model description
- `score` — automated cosine similarity (for reference, but you judge quality holistically)

## Rubric

| Score | Verdict | Meaning |
|-------|---------|---------|
| 4 | PASS | Excellent — accurate, captures the scene well, comparable to or better than cloud |
| 3 | PASS | Adequate — captures the gist correctly, minor omissions or slight inaccuracies |
| 2 | FAIL | Poor — significant inaccuracies, hallucinations, or missing the main point |
| 1 | FAIL | Failure — empty, nonsensical, or describes something entirely different |

**Accuracy = (scores 3 + 4) / total samples**

## What to assess

- **Accuracy**: does the description match what's actually in the photo?
- **Completeness**: are the key elements mentioned?
- **Hallucinations**: does it describe things not in the photo?
- **Specificity**: "Japanese mirin rice wine" is better than "bottle"
- **Artifacts**: watch for model artifacts like "Draft 3:" prefixes, repeated text, or thinking traces

## Output format

IMPORTANT: Write your reasoning BEFORE the score. Analyze first, then decide.

For each sample, write:
```
### {sample_id}
Reasoning: {1-2 sentences analyzing accuracy, completeness, and hallucinations — reference what you see in the photo}
Hallucinations: [list or "none"]
Score: X/4 (PASS|FAIL)
```

End with a summary:
```
## Summary
- Samples: N
- Pass (3-4): X (Y%)
- Fail (1-2): X (Y%)
- Accuracy: Y%
- Score distribution: 4=N, 3=N, 2=N, 1=N
```
