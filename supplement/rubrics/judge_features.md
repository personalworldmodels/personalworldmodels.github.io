# Judge: primitives.features

Evaluate on-device model photo entity extraction against cloud ground truth.

## Instructions

1. Read the JSONL file provided by the user
2. Skip the first line (`_type: "meta"`) — it's metadata
3. For each result line, compare `cloud_result` vs `device_output`
4. **Look at the actual photo** at `image_path` to ground your assessment
5. Score each sample on a 1-4 scale (see rubric below)
6. Write a summary report at the end

## JSONL Structure

Each result line has:
- `sample_id` — unique photo ID
- `image_path` — path to the actual photo (read it!)
- `cloud_result` — Gemini extraction (ground truth): `{humans, objects, activities, spaces}`
- `device_output` — on-device model extraction: same schema
- `score` — automated Jaccard score (for reference, but you judge quality holistically)
- `substeps` — individual extraction prompts and raw responses (useful for diagnosing failures)

## Rubric

| Score | Verdict | Meaning |
|-------|---------|---------|
| 4 | PASS | Equivalent or better than cloud — captures the same entities, no hallucinations |
| 3 | PASS | Minor differences — missed 1-2 items or used synonyms (e.g., "sofa" vs "couch") |
| 2 | FAIL | Significant gaps — missed major entities, only generic labels, or clear hallucinations |
| 1 | FAIL | Failure — empty, nonsensical, or entirely wrong |

**Accuracy = (scores 3 + 4) / total samples**

## What to assess

- **Humans**: correct count and labels (man/woman/child)?
- **Objects**: specific labels vs generic ones ("cocktail" vs "food")?
- **Activities**: meaningful actions captured?
- **Spaces**: correct environment type?
- **Hallucinations**: things the device claims are there but aren't in the photo
- **Synonym matches**: cases where device and cloud said the same thing differently (not a penalty)

## Output format

IMPORTANT: Write your reasoning BEFORE the score. Analyze first, then decide.

For each sample, write:
```
### {sample_id}
Reasoning: {1-2 sentences analyzing what the device got right, what it missed, and what it hallucinated — reference what you see in the photo}
Hallucinations: [list or "none"]
Missed: [list or "none"]
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
