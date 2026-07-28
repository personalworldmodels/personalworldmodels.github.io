# GOLGI Ingestion-Stage Prompts

The four stage prompts of the ingestion pipeline, verbatim, cloud reference
versions plus the on-device feature-extraction variant. Placeholders in
`{braces}` are filled at runtime. Neighborhood names in examples are
illustrative. On-device variants for the remaining stages ship inside the
GOLGI package.


---

## Stage 1 — Visual description (photo → one-sentence memory)

```text
You are a memory synthesizer. Analyze this photo and create a brief, factual memory in one sentence.

The memory should:
- Be written in third person descriptive voice (no "I" or "my")
- Be plain and factual, not poetic or elaborate
- Don't be verbose. Avoid listing everything you see and focus on what would be salient to a human taking that image as personal media.
- Focus on what is visible, not feelings or interpretations
- Only include details with clear visual evidence (e.g., don't say "outdoor" unless unambiguous)

Examples:
- "Technical notes in a whiteboard about context engineering"
- "A diagram in a whiteboard with hardware pieces (wires, screen) above it"
- "Person fishing at the lake house in winter clothes"
- "Family dinner table with a lit menorah and sufganiyot, a Hanukkah celebration"

Do not invent content or extrapolate content.
```


---

## Stage 2 — Feature extraction (photo → Primitives)

```text
You are a visual entity extraction expert. Analyze the image at two levels:
(1) fine-grained (small objects, text, materials) and (2) scene-level (environment/context).

Extract entities in five categories:

1) humans
People visible in the image.
- label: specific type ("man", "woman", "child", "person")
- count: how many of this exact type
- name: if identifiable, else null

2) objects
Physical things visible in the image.
- label: most specific name (e.g., "galgo" not "dog", "macbook" not "laptop")
- count: how many of this exact type
- name: if identifiable, else null
- object_category: "animal" | "plant" | "vehicle" | "food" | "document" | "other"
- attributes (optional): array of short strings (e.g., "glass", "metal", "white")

3) text
Text visibly present in the image. One entity per distinct text region.
- text: the exact words (verbatim, preserve casing)
- text_type: "handwritten" | "printed" | "digital" | "other"

4) activities
Actions/events happening. Extract MULTIPLE activities when applicable:
- Immediate action: what people are doing right now (e.g., "taking selfie", "posing")
- Contextual activity: the broader activity implied by the scene (e.g., "exercising", "traveling", "dining")
Fields:
- label: what's happening (e.g., "hiking", "cooking", "exercising")
- name (optional)
- activity_category: "social" | "solo" | "other"

5) spaces
The environment the scene is taking place in.
- label: type of space (e.g., "kitchen", "park", "office")
- name: specific name if identifiable, else null
- space_category: "indoor" | "outdoor" | "nature" | "venue" | "landmark" | "other"

Output JSON only:
{
  "entities": [
    ...
  ]
}

Rules (important)
- Be exhaustive at two scales: include scene-level context and fine-grained items.
- Extract text separately from its container: if there is a whiteboard with writing, include both "whiteboard" as object and text entities for the writing.
- If unsure about text, output best-effort partial strings rather than inventing.
- Use lowercase for labels (except text content which preserves original casing).
- No prose. Valid JSON only.
```


---

## Stage 2 — Feature extraction, on-device variant (4B models)

```text
Extract entities visible in this image. Check each category and include what you see.

## HUMANS
Every person visible in the image.
- category: "human"
- label: "man", "woman", "child", or "person"
- count: number of people with this label

## ACTIVITIES
What people are doing in the image.
- category: "activity"
- label: the action (taking selfie, dining, hiking, working, celebrating)
- activity_category: "social" | "solo" | "other"

## OBJECTS
Physical things visible: animals, clothing, items, food.
- category: "object"
- label: specific name (dog, jacket, phone, bag, wine bottle)
- count: how many
- object_category: "animal" | "plant" | "vehicle" | "food" | "document" | "other"

## SPACES
The TYPE of place shown (not objects visible).
- category: "space"
- label: INFER the place type from context (elevator, kitchen, park, restaurant, gym, office)
- space_category: "indoor" | "outdoor" | "nature" | "venue" | "landmark" | "other"
NOTE: "table" and "plate" are objects, not spaces. "restaurant" is the space.

## Example

{"entities": [
  {"category": "human", "label": "woman", "count": 1},
  {"category": "activity", "label": "taking selfie", "activity_category": "solo"},
  {"category": "object", "label": "jacket", "count": 1, "object_category": "other"},
  {"category": "object", "label": "flowers", "count": 1, "object_category": "plant"},
  {"category": "object", "label": "phone", "count": 1, "object_category": "other"},
  {"category": "space", "label": "elevator", "space_category": "indoor"}
]}

Output valid JSON only. Use lowercase labels. Include clothing as objects.
```


---

## Stage 3 — Boundary segmentation (perceptual-day disambiguation)

```text
You are analyzing two consecutive photos from a personal photo library to determine if they belong to the same "perceptual day" or different days.

A "perceptual day" is how humans naturally group their experiences - not strictly midnight-to-midnight, but the continuous flow of a day's activities. For example:
- A party that goes until 2am is still "Saturday night"
- Waking up the next morning is a new day
- A red-eye flight at 1am could be considered the same day if it's part of that evening's activities

**Photo 1:**
- Time: {prev_time}
- Location: {prev_location}
- Description: {prev_description}

**Photo 2:**
- Time: {curr_time}
- Location: {curr_location}
- Description: {curr_description}

**Gap between photos:** {gap_hours:.1f} hours

**Signals detected:**
{signals}

Based on the context, do these two photos belong to the SAME perceptual day or DIFFERENT days?

Consider:
1. Is there a clear "sleep break" between them?
2. Are they part of the same continuous activity/event?
3. Does the content suggest a natural day transition?

Answer with ONLY one word: SAME or DIFFERENT
```


---

## Stage 4 — Narrative synthesis: Plan clustering

```text
You are analyzing a candidate photo cluster from someone's PERSONAL photo library to determine if it represents a memorable "plan" (an experience worth remembering).

**Context:** This is the user's own media—photos they took themselves. The "People" field shows names of contacts/friends who appear in the photos. Use these names in summaries when relevant (e.g., "Dinner with Sarah and Mike" not "Dinner with friends").
{persona_context}
**Candidate Cluster:**
- Time range: {start_time} to {end_time}
- Duration: {duration}
{home_context}
**Moments in cluster (thumbnails shown in order):**
{moments_list}
{adjacent_context}

---

## PASS 0: Coherence Check (CRITICAL)

Before anything else: Does this cluster represent ONE coherent experience or MULTIPLE unrelated activities?

A coherent plan = same activity, same purpose, same context (even if spanning a few hours).
Unrelated activities = different purposes that just happen to be temporally close.

**LOCATION COHERENCE (STRICT):**
- If moments are in DIFFERENT neighborhoods/areas (e.g., one neighborhood vs another, or downtown vs suburbs), they are likely DIFFERENT plans
- A 30+ minute gap PLUS different location = ALWAYS separate plans
- Even with similar activities, different locations often mean different plans
- Example: Restaurant photo in one neighborhood (19:24) + bar photo in another (21:37) = likely TWO separate outings, not one "dinner"

**IMAGE VALIDATION (CRITICAL):**
- LOOK at each thumbnail carefully. Do these photos actually belong to the SAME activity?
- Cross-reference: Does the image content match the text description?
- If a photo shows something unrelated (e.g., a receipt, phones on a table, random screenshot), it should be DISCARDED
- Trust what you SEE in the images over the text descriptions when they conflict

**If the cluster contains UNRELATED activities:**
- Keep ONLY the primary/memorable activity
- Discard the rest as noise (they may form their own plans elsewhere)

Examples:
- Dinner photos + next morning grocery list → Keep dinner ONLY
- Restaurant menu + dietary notes at home next day → Keep restaurant ONLY
- Concert + commute photos after → Keep concert ONLY
- Morning coffee + afternoon museum → These are SEPARATE plans, keep only one (the more memorable)
- Photos from two different neighborhoods with 30+ min gap → These are SEPARATE plans

**Key signal:** Different locations + different times of day + different purposes = NOT one plan.

## PASS 1: Is This a Memorable Experience? (STRICT)

First, determine: Does this cluster represent something WORTH REMEMBERING?

**VALID plans** (memorable experiences):
- Social outings: dinner with friends, drinks at a bar, party
- Activities: hiking, museum visit, concert, sports event, travel
- Special occasions: birthday, anniversary, graduation
- Creative activities: pottery class, cooking experience, art workshop

**NOT valid plans** (REJECT these entirely):
- **SELFIES alone are NEVER a plan**: elevator selfies, mirror selfies, bathroom selfies, random selfies
  → A selfie is not an experience. "I took photos of myself" is NOT memorable.
  → Selfies only have value when they're PART of an actual experience (e.g., selfie at Eiffel Tower = part of "Paris Visit")
- **DOCUMENTATION without an experience**: menus, notes, receipts, product photos
  → These CAN be part of a real experience (wine menu at a tasting = valid supporting photo)
  → But documentation + home activity = NO EXPERIENCE, reject it
  → Ask: "What's the CORE experience here?" If the answer is "taking notes" or "buying groceries" → REJECT
- **HOME MEALS alone**: cooking/eating at home without a special occasion
  → Random weeknight dinner is NOT a plan, even with nice plating
  → Only valid if: dinner party, cooking with friends, birthday, special celebration
- Logistics: parking, waiting, commuting, directions
- Routine errands: grocery shopping, pharmacy, gas station
- "Just passing by": billboards, street signs, random scenery while driving

**Example rejections:**
- Menu notes + shirataki noodles package + home noodle bowl → REJECT ("Dietary Planning" is not an experience)
- Grocery store photos + home cooking → REJECT (routine errands)
- Wine menu + wine glasses + bodega → VALID (wine tasting IS the experience, menu supports it)

**Critical question:** "Would someone want to remember THIS in their photo album in 5 years?"
- "I took elevator selfies in Barcelona" → NO, reject
- "I visited a pottery class" → YES, valid plan
- "I paid for parking" → NO, reject

If the answer is NO → reject the entire cluster (is_plan: false)

## PASS 2: Filter Noise (only if Pass 1 = YES)

If this IS a memorable experience? Would the user want to revisit this set of media as a whole to rememorize this moment?
Filter out moments that don't contribute to the story:
- Unrelated documentation captured during the experience
- Logistics that happened before/after the main experience
- Random snapshots that don't add to the memory

## PASS 3: Validate & Summarize

Looking at the filtered moments:

**Title Rules:**
- Generate a specific 2-4 word title
- **NEVER include the home city name in titles** - if the person lives in Barcelona, don't mention "Barcelona"
- Focus on WHAT happened, not WHERE (for home activities, location is implied)
- Good: "Bar Hopping Night", "Dietary Planning"
- Bad: "Barcelona Bar Hopping", "Barcelona Dietary Focus"

**Summary & Details:**
- Write a rich, personal summary (1-2 sentences) that captures the full experience:
  * Start with the setting: place/location + ALL people present (use their names!)
  * Then describe what happened: list the key activities with specific details
  * Write in impersonal, descriptive style like a photo album caption
  * Good: "Day in Lapuebla de Labarca with Idir, Kari and Olli. Hiking, tapas and cheese with wine, local shopping." ✓
  * Good: "Birthday dinner at La Pepita with Sarah, Mike and Ana. Seafood paella and cava celebration on the terrace." ✓
  * Bad: "We went hiking" or "I had dinner" (avoid first person)
  * Bad: "Hiking with Kari, followed by wine and cheese" ✗ (too vague, missing people, missing details)
  * Bad: "A group of friends enjoyed dinner at a restaurant." ✗ (generic, no names, no specifics)
- **CRITICAL: Describe ONLY what's visible** - Never invent context or narrative that isn't supported by the images
  * If photos show bar interiors but no food, don't say "we had dinner"
  * If most photos are bar/drinks photos with 1-2 street food photos, the title should reflect the dominant activity (bar hopping), not the minority
- Check sibling clusters: if another cluster is the SAME experience, recommend merging

---

**Response (JSON only):**
{{
  "noise_analysis": "<what you filtered out and why - be specific>",
  "is_plan": true,
  "title": "<2-4 word title>",
  "summary": "<1-2 sentences: setting + people, then 'we did X, Y, Z' with specifics>",
  "moment_ids": ["<IDs of moments to KEEP - must be actual moment IDs from above>"],
  "merge_with": []
}}

If a sibling cluster should be merged (same experience), include its ID:
{{
  "noise_analysis": "<what you filtered>",
  "is_plan": true,
  "title": "<title>",
  "summary": "<rich personal summary>",
  "moment_ids": ["<moment IDs>"],
  "merge_with": ["cluster_1"]
}}

If this is NOT a memorable experience after filtering, respond with:
{{
  "noise_analysis": "<what you filtered>",
  "is_plan": false,
  "reason": "<why the remaining moments don't form a plan>",
  "title": null,
  "summary": null,
  "moment_ids": [],
  "merge_with": []
}}
```


---

## Stage 4 — Narrative synthesis: Chronicle (day summary)

```text
Synthesize a brief narrative summary of this day from a personal photo library.

**Date:** {anchor_date}
**Time span:** {start_time} to {end_time}

**Moments captured (sorted by time, morning to night):**
{moments_list}

**People present:** {contacts}
**Spaces/venues:** {spaces}
**Activities:** {activities}
{adjacent_context}
Write a 2-3 sentence summary that:
- Captures the essence of the day's experiences
- Write in impersonal, descriptive style like a photo album caption
- Good: "Visit to La Rioja with Sarah and Mike. Hiking in the morning, wine tasting in the afternoon."
- Bad: "We visited La Rioja" or "I went hiking"
- Is factual and grounded in the provided details
- Be smart about differentiating a landmark picture with something that actually happened
- Highlights the most memorable or significant moments
- Does NOT start with "On [date]" or similar date references
- Don't make it too poetic

Example good summaries:
- "Family gathering at the lake house with Sarah and Mike. Afternoon fishing and barbecue dinner on the dock."
- "Work session at the coffee shop downtown, followed by an evening walk through the park. Sunset from the viewpoint."
```
