# Stage I/O Reference: Cloud vs On-Device

Real examples from the Golgi pipeline showing exactly what goes in and comes out for each stage, and how the on-device architecture differs from cloud.

---

## Stage 1: `primitives.features` (Entity Extraction)

### Architecture comparison

| Aspect | Cloud (gemini-2.0-flash) | On-device (qwen3.5:0.8b) |
|--------|--------------------------|---------------------------|
| LLM calls | 1 call, all categories at once | 4 calls, 1 per category (humans, objects, activities, spaces) |
| Image | Full-resolution original | Resized to 1024px max, JPEG 85% quality |
| Prompt | Single comprehensive system prompt extracting 5 entity types | 4 small focused prompts, one per category |
| Output | Single JSON with `entities[]` array | 4 separate JSON responses, merged |

**Why:** Small models hallucinate when asked to extract everything at once. Splitting into per-category prompts reduces the cognitive load and fits the smaller context window. Text extraction is dropped for on-device (not reliable at <4B parameters).

### Cloud path

**Input:** Full-resolution photo (HEIC/JPEG, any size)

**Prompt (system):**
```
You are a visual entity extraction expert. Analyze the image at two levels:
(1) fine-grained (small objects, text, materials) and (2) scene-level (environment/context).

Extract entities in five categories:

1) humans - label, count, name
2) objects - label, count, name, object_category, attributes
3) text - text, text_type
4) activities - label, name, activity_category
5) spaces - label, name, space_category

Output JSON only: {"entities": [...]}
```

**Output example:**
```json
{
  "humans": [{"label": "woman", "count": 2}],
  "objects": [{"label": "wine glass", "count": 3, "object_category": "other"},
              {"label": "priorat bottle", "count": 1, "object_category": "food"}],
  "activities": [{"label": "dining", "activity_category": "social"}],
  "spaces": [{"label": "dining room", "space_category": "indoor"}]
}
```

### On-device path

**Input:** Photo resized to 1024px max dimension, JPEG quality 85%

**Prompt 1 - Humans:**
```
List every person visible in this image.

For each person:
- label: "man", "woman", "child", or "person"
- count: how many of this type
- name: if identifiable, else null

Example: {"humans": [{"label": "woman", "count": 2}, {"label": "child", "count": 1}]}

If no people are visible, return: {"humans": []}
Output valid JSON only.
```

**Prompt 2 - Objects:**
```
List physical objects visible in this image. Include animals, clothing, food, vehicles,
plants, and items.

For each object:
- label: specific name (e.g., "dog" not "animal", "macbook" not "laptop")
- count: how many
- object_category: "animal" | "plant" | "vehicle" | "food" | "document" | "other"

Output valid JSON only. Use lowercase labels. Include clothing as objects.
```

**Prompt 3 - Activities:**
```
What activities or actions are happening in this image?

For each activity:
- label: what is happening (e.g., "hiking", "taking selfie", "dining", "working")
- activity_category: "social" | "solo" | "other"

If no clear activity, return: {"activities": []}
Output valid JSON only.
```

**Prompt 4 - Spaces:**
```
Describe the environment or location type in this image.

For each space:
- label: type of space (e.g., "kitchen", "park", "restaurant", "street")
- name: specific name if identifiable, else null
- space_category: "indoor" | "outdoor" | "nature" | "venue" | "landmark" | "other"

Output valid JSON only.
```

**Output example** (merged from 4 calls):
```json
{
  "humans": [{"label": "woman", "count": 2}],
  "objects": [{"label": "wine glass", "count": 2, "object_category": "other"}],
  "activities": [{"label": "eating", "activity_category": "social"}],
  "spaces": [{"label": "restaurant", "space_category": "indoor"}]
}
```

---

## Stage 2: `load.describe` (Memory Synthesis)

### Architecture comparison

| Aspect | Cloud (gemini-2.0-flash) | On-device (qwen3.5:0.8b) |
|--------|--------------------------|---------------------------|
| LLM calls | 1 | 1 |
| Image | Full-resolution original | Resized to 1024px max, JPEG 85% quality |
| Temperature | 0.7 | 0.3 |
| Max tokens | Unlimited | 150 |
| Prompt | Detailed system prompt with rules + examples | Simplified prompt with minimal rules |

**Why:** Smaller image reduces inference time and memory. Lower temperature + token cap prevent the model from rambling or hallucinating details not visible in the photo.

### Cloud path

**Prompt (system):**
```
You are a memory synthesizer. Analyze this photo and create a brief, factual memory
in one sentence.

The memory should:
- Be written in third person descriptive voice (no "I" or "my")
- Be plain and factual, not poetic or elaborate
- Don't be verbose. Avoid listing everything you see and focus on what would be
  salient to a human taking that image as personal media.
- Focus on what is visible, not feelings or interpretations
- Only include details with clear visual evidence

Examples:
- "Technical notes in a whiteboard about context engineering"
- "Person fishing at the lake house in winter clothes"
- "Family dinner table with a lit menorah and sufganiyot, a Hanukkah celebration"

Do not invent content or extrapolate content.
```

**Output example:**
```
The earthenware teapot sat next to wine glasses and a bottle of Priorat on the wooden table.
```

### On-device path

**Prompt (system):**
```
Describe this photo in one factual sentence.

Rules:
- Third person (no "I" or "my")
- Focus on what's visible
- Be concise

Examples:
- "Person fishing at a lake in winter clothes"
- "Family dinner with lit candles on the table"
- "Technical diagram on a whiteboard"

Return ONLY the sentence, no quotes.
```

**Output example:**
```
A teapot and wine glasses on a wooden table.
```

---

## Stage 3: `narrative.boundary` (Day Boundary Detection)

### Architecture comparison

| Aspect | Cloud (gemini-2.0-flash) | On-device (non-thinking) | On-device (thinking) |
|--------|--------------------------|--------------------------|----------------------|
| LLM calls | 1 | 1 | 1 |
| Strategy | Full prompt with 3 reasoning questions + signals | Few-shot message turns (7 examples) + single question | Few-shot with short reasoning traces (7 examples) |
| System prompt | Detailed "perceptual day" concept explanation | One-line instruction | One-line instruction + "write ONE short reason" |
| Output format | "SAME" or "DIFFERENT" | "SAME" or "NEW" | Short reason + "SAME" or "NEW" |

**Why:** Small models can't reason about the "perceptual day" concept from a single prompt. Few-shot examples teach by demonstration instead of explanation. The thinking variant allows models with reasoning capabilities to show their work.

### Cloud path

**Input:**
```
Photo 1: 11:26 PM - The earthenware teapot sat next to wine glasses and a bottle of
  Priorat on the wooden table.
  Location: Barcelona
Photo 2: 12:15 PM next afternoon - The laptop screen showed a diagram titled "Multiple
  Stage, Agentic Workflow Dive Deep" outlining tasks and outputs.
  Location: Barcelona
Gap: 36.8 hours
Signals: time_gap (36.8h, confidence 0.95)
```

**Prompt:** Full "perceptual day" explanation with 3 reasoning questions:
1. Is there a clear "sleep break" between them?
2. Are they part of the same continuous activity/event?
3. Does the content suggest a natural day transition?

**Output:** `DIFFERENT`

### On-device path (non-thinking)

**System:** `Is this a new day or the same day? Answer SAME or NEW. SAME day = still awake, no sleep break. NEW day = woke up from sleep.`

**Few-shot examples (7 turns):**
```
User: Photo 1: 3:00 PM - Walking through a market
      Photo 2: 8:30 PM same evening - Dinner at a restaurant
      Gap: 5.5 hours
Assistant: SAME

User: Photo 1: 10:30 PM - Reading in bed
      Photo 2: 8:15 AM next morning - Coffee in the kitchen
      Gap: 9.8 hours
Assistant: NEW
... (5 more examples)
```

**User message:**
```
Photo 1: 11:26 PM - The earthenware teapot sat next to wine glasses and a bottle of
  Priorat on the wooden table.
Photo 2: 12:15 PM next afternoon - The laptop screen showed a diagram.
Gap: 36.8 hours
```

**Output:** `SAME` (incorrect — the model missed the 36.8h overnight gap)

### On-device path (thinking models)

Same structure but few-shot responses include brief reasoning:
```
Assistant: Night to morning, slept overnight. NEW
Assistant: Same afternoon into evening, no sleep. SAME
```

**System:** `Is this a new day or the same day? Write ONE short reason then answer SAME or NEW. SAME = still awake, no sleep. NEW = woke up from sleep.`

---

## Stage 4: Plan Pipeline (Multi-Step Clustering)

### Architecture comparison

| Step | Cloud | On-device | Change |
|------|-------|-----------|--------|
| **Coherence check** | Part of unified prompt (Pass 0) | Skipped (handled by memorability) | Simplified |
| **Memorability** | Part of unified prompt (Pass 1) | Separate call with yes/no + reason | Decomposed for verifiable output |
| **Noise filter** | Part of unified prompt (Pass 2) | Separate call: keep/remove moment IDs | Decomposed for verifiable output |
| **Summarize** | Part of unified prompt (Pass 3) | Separate call: title + summary | Decomposed for verifiable output |
| **Cover selection** | Full candidate set + reasoning | Max 6 candidates + `/nothink` prefix | Reduced input + suppressed reasoning |

**Why:** The cloud prompt is a single multi-pass call (coherence -> memorability -> noise -> summarize). Small models can't hold all 4 passes in context simultaneously. Decomposing into 3 stepwise calls lets each step be independently verified and keeps the context window manageable.

### Cloud path (single call)

**Input:** Thumbnails + moment descriptions + contacts + sibling clusters + home context

**Prompt structure:**
```
PASS 0: Coherence Check — is this ONE experience or MULTIPLE?
PASS 1: Is This a Memorable Experience? — strict yes/no with examples
PASS 2: Filter Noise — remove logistics, unrelated moments
PASS 3: Validate & Summarize — title + summary + kept moment IDs
```

**Output:**
```json
{
  "noise_analysis": "Filtered 2 commute photos and 1 receipt",
  "is_plan": true,
  "title": "Wine Tasting Evening",
  "summary": "Wine tasting at Bodega Marin with Ana and Carlos. Tried local
    Priorat wines paired with cheese and olives on the terrace.",
  "moment_ids": ["m1", "m2", "m3", "m5"],
  "merge_with": []
}
```

### On-device path (3 stepwise calls)

**Call 1 — Memorability check:**
```
Look at these photos from 2025-01-15.

Moments:
- m1 (19:24): Wine glasses on a table
- m2 (19:45): Cheese board with olives
- m3 (20:10): People toasting

Would someone want to look back at this in a year? Be STRICT.
```
Output: `{"is_memorable": true, "reason": "wine tasting with friends"}`

**Call 2 — Focus filter:**
```
These photos might show multiple events mixed together.
Identify the MAIN experience and keep only photos that belong to it.

Moments:
- m1 (19:24): Wine glasses on a table
- m2 (19:45): Cheese board with olives
- m3 (20:10): People toasting
- m4 (20:30): Receipt photo
```
Output: `{"main_event": "wine tasting", "keep_ids": ["m1", "m2", "m3"], "removed_ids": ["m4"]}`

**Call 3 — Summarize:**
```
Generate a title and summary for this experience.

Moments:
- m1 (19:24): Wine glasses on a table
- m2 (19:45): Cheese board with olives
- m3 (20:10): People toasting
People: Ana, Carlos
```
Output: `{"title": "Wine Tasting Evening", "summary": "Wine tasting with Ana and Carlos.", "merge_with": []}`

---

## Scoring Methods

| Stage | Metric | Range | Description |
|-------|--------|-------|-------------|
| `primitives.features` | Jaccard similarity | 0.0–1.0 | Per-category label overlap, averaged across 4 categories |
| `load.describe` | Cosine similarity | 0.0–1.0 | Embedding similarity (nomic-embed-text) between cloud and device descriptions |
| `narrative.boundary` | Exact match | 0 or 1 | Binary: same decision as ground truth |
| Plan pipeline | Qualitative | N/A | Number of plans + moment coverage + manual assessment |
