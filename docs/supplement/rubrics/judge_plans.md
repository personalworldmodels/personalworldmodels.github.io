# Judge: Plan Quality

Evaluate on-device model plan generation holistically. Unlike primitives/features which compare field-by-field, plans are evaluated as coherent memory units.

## Instructions

1. Read the JSONL file provided by the user
2. Skip the first line (`_type: "meta"`) — it's metadata
3. For each plan, **look at ALL the photos** in `moment_photos` to understand the experience
4. Also look at any `discarded_photos` to verify exclusions were correct
5. Score each plan on a 1-4 scale using the holistic rubric below
6. Write a summary report at the end

## JSONL Structure

Each plan line has:
- `plan_id` — unique plan ID
- `title` — the plan title (device output)
- `summary` — the plan description (device output)
- `moment_photos` — list of photo paths included in the plan
- `discarded_photos` — list of photo paths that were considered but excluded
- `moment_count` — number of moments in the plan
- `cloud_title` — cloud-generated title (ground truth, if available)
- `cloud_summary` — cloud-generated summary (ground truth, if available)

## Holistic Rubric

Plans are memories. Evaluate them as such.

| Score | Verdict | Meaning |
|-------|---------|---------|
| 4 | PASS | **Worth remembering + Faithful + Well-grouped** — This is a coherent experience someone would want to recall. The title captures the essence, the summary describes what actually happened, and the included photos belong together. |
| 3 | PASS | **Worth remembering + Minor issues** — The experience is valid and memorable, but the title is generic ("Lisbon Evening") or the summary misses key details visible in the photos. Or: one photo feels slightly out of place but doesn't ruin coherence. |
| 2 | FAIL | **Not worth remembering OR unfaithful** — Either: (a) trivial content that adds no memory value (single elevator selfie, random food shot), or (b) title/summary describes something different from what the photos show, or (c) clear grouping errors (unrelated photos mixed together). |
| 1 | FAIL | **Failure** — Empty, nonsensical, or the photos have no coherent relationship at all. |

**Accuracy = (scores 3 + 4) / total plans**

## What to assess

### 1. Worth Remembering? (Memory Value)

Ask: "Would someone want to look back at this plan in 6 months?"

**YES indicators:**
- Named event or location (premiere, winery tour, birthday dinner)
- Activity with narrative arc (cooking session, pottery class, hiking)
- Social gathering with context (game night with friends, family dinner)
- Cultural or travel experience (calcotada, wine cellar visit)
- Even small moments if distinctive (specific street scene, meaningful selfie)

**NO indicators:**
- Single generic photo with no context (random elevator selfie)
- Pure documentation without experience (receipt photo, product shot)
- Arbitrary grouping of disconnected photos

### 2. Title & Summary Faithfulness (Accuracy)

Look at the photos, then read the title and summary.

**Check for:**
- **Title specificity**: "Alentejo Winery Tour" > "Wine Experience" > "Lisbon Day"
- **Title captures experience TYPE**: "Noodle Workshop" > "Ramen & Dessert" (workshop vs just food)
- **Summary accuracy**: Does it describe what you actually see in the photos?
- **Key elements captured**: If there's a named location/event visible, is it mentioned?

**Writing style (IMPORTANT):**
- **Impersonal style**: Write as if this is the owner's memory, not about them
- **NO owner name**: "Attended a workshop..." ✅ NOT "Alex attended..." ❌
- **USE contact names**: "...with Ana and Carlos" ✅ when contacts appear in photos
- Think of it like a photo album caption that belongs to the person

**Red flags:**
- Title mentions things not in the photos
- Summary describes a different activity
- Generic aggregation that could apply to any photos ("spent time in the city")
- Uses owner's name in summary (should be impersonal)
- Misses contact names when people are clearly present

### 3. Grouping Quality (Exclusions)

If `discarded_photos` is not empty, check if the exclusions were correct.

**Correct exclusions:**
- Photos from a different time/place that happened to be close in time
- Duplicates or near-duplicates
- Transitional photos (walking between locations)

**Incorrect exclusions:**
- Photos that clearly belong to the same experience
- The "best" photo of the group was excluded
- Key context was removed (arrival shot, group photo)

Also check the included photos:
- Do they all belong together?
- Is there a photo that clearly doesn't fit?

## Output Format

IMPORTANT: Write your reasoning BEFORE the score. Look at the photos, think, then decide.

The output should be **auditable** — include enough metadata that someone can verify your assessment.

For each plan, write:
```
### {plan_id}

**Title:** "{title}"
**Summary:** "{full summary or first 150 chars}..."
**Date:** {anchor_date}
**Photos:** {moment_count} included, {discarded_count} excluded
**Photo files:** {comma-separated list of filenames, e.g., IMG_6801.jpg, IMG_6802.jpg, ...}

**Memory Value:** {1-2 sentences — is this worth remembering? why/why not?}
**Faithfulness:** {1-2 sentences — does title/summary match photos?}
**Grouping:** {1 sentence — are photos correctly grouped? any exclusion issues?}

Score: X/4 (PASS|FAIL)
```

End with a summary:
```
## Summary

- Plans evaluated: N
- Pass (3-4): X (Y%)
- Fail (1-2): X (Y%)
- Accuracy: Y%
- Score distribution: 4s=N, 3s=N, 2s=N, 1s=N

### Observations

**Common issues:**
- [List recurring problems]

**Strengths:**
- [What the model does well]

**Recommendations:**
- [Suggestions for improvement]
```

## Examples

### Example: Score 4 (PASS)

**Title:** "Ceramics Workshop"
**Summary:** "Spent the afternoon shaping clay pots and glazing pieces with pink accents at a ceramics studio."
**Date:** 2025-02-11
**Photos:** 8 included, 0 excluded
**Photo files:** IMG_6505.jpg, IMG_6510.jpg, IMG_6515.jpg, IMG_6520.jpg, IMG_6525.jpg, IMG_6530.jpg, IMG_6535.jpg, IMG_6540.jpg

**Memory Value:** Clear activity with narrative arc — arriving at workshop, working with clay, finished pieces. Worth remembering as a creative afternoon.
**Faithfulness:** Title captures the experience TYPE (workshop). Summary is impersonal (no owner name), describes visible process (clay, glazing, pink).
**Grouping:** All photos are from the same workshop session, no strays.

Score: 4/4 (PASS)

---

### Example: Score 2 (FAIL)

**Title:** "Elevator Selfie Moment"
**Summary:** "A person took a photo in an elevator."
**Date:** 2025-02-05
**Photos:** 1 included, 0 excluded
**Photo files:** IMG_6401.jpg

**Memory Value:** Single generic selfie with no distinctive context. Not memorable.
**Faithfulness:** Technically accurate but minimally descriptive.
**Grouping:** N/A (single photo)

Score: 2/4 (FAIL) — trivial content, no memory value

---

### Example: Score 3 (PASS)

**Title:** "Lisbon Evening"
**Summary:** "Alex enjoyed dinner and drinks at several locations around the city center."
**Date:** 2025-03-07
**Photos:** 6 included, 1 excluded
**Photo files:** IMG_7101.jpg, IMG_7105.jpg, IMG_7110.jpg, IMG_7115.jpg, IMG_7120.jpg, IMG_7125.jpg

**Memory Value:** Social outing with multiple venues — worth remembering as a night out.
**Faithfulness:** Title is generic (which Lisbon evening?). Summary uses owner name "Alex" instead of impersonal style. Could mention specific restaurants (a restaurant sign is visible).
**Grouping:** Included photos all from same evening. Excluded photo was a random street shot — correct exclusion.

Score: 3/4 (PASS) — valid memory, but generic title + uses owner name (should be "Enjoyed dinner and drinks...")
