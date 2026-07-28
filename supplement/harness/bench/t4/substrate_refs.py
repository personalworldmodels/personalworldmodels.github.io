"""PWM-Bench T4 — the substrate→diffusion bridge (entity-reference surface, spec §8.1).

This is the load-bearing half of T4: it reads the *anchored entities + reference
imagery* the substrate already maintains and turns them into exactly what a
diffusion consumer needs — a reference image path (for IP-Adapter) and a
relational conditioning prompt of the spec's form

    "<entity> at <other-entity> doing <activity>"   (§7.3)

Nothing here is diffusion-specific; it is the substrate's §8.1-T4 contract made
concrete for the Golgi reference implementation. `pipeline.py` consumes it.

Substrate access notes (Golgi specifics, learned by probing the live DBs):
- People are `Contact`s; the best face is `contact.representative_face_id`, whose
  id encodes the source moment: ``face:<moment_id>:<idx>``. The moment's
  ``media_ids[0]`` resolves to a `Media` whose `.path` is STALE (points at an old
  ``data/personas/<persona>/photo`` layout) — we resolve by *basename* under the
  live photo dir (``data/<persona>/photo/``).
- Places are `Spot`s; a labelled spot ("Home Lab/Studio") + its moments give both
  the "<other-entity>" and a cover image.
- Activities come from `find_activities_by_moment_id` on the entity's moments —
  the substrate's own "what" labels, not invented.
- `get_all_contacts(persona)` filters by a normalised slug that does not always
  match stored ``persona_id``; we therefore call it unfiltered and filter in
  Python on ``persona_id`` (robust across both observed data roots).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Live photo directories, keyed by persona. The substrate's Media.path is stale
# (old data/personas/<p>/photo layout); real files live here. Resolved by
# basename. Override the root via PWMBENCH_T4_PHOTO_ROOT.
_PHOTO_ROOT = os.environ.get("PWMBENCH_T4_PHOTO_ROOT", "./data")


def _photo_dir(persona: str) -> Path:
    return Path(_PHOTO_ROOT) / persona / "photo"


@dataclass
class EntityReference:
    """One anchored entity + its reference image + a relational prompt.

    This is the unit the diffusion track conditions on: `entity` is who/what,
    `reference_image_path` is the substrate's representative imagery for it (the
    IP-Adapter input), and `prompt` is the spec §7.3 relational template the
    substrate assembled from its own who/where/what knowledge.
    """

    persona: str
    entity_kind: str  # "contact" | "spot"
    entity_id: str
    entity_label: str  # "<entity>" in the template
    reference_image_path: Path
    prompt: str  # "<entity> at <other-entity> doing <activity>"
    other_entity: str | None = None  # "<other-entity>"
    activity: str | None = None  # "<activity>"
    source_moment_id: str | None = None
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Substrate readers
# --------------------------------------------------------------------------- #
def _named_contacts(graph, persona: str) -> list:
    """Confirmed, *named* contacts for the persona.

    Uses the unfiltered query then filters on persona_id + name, because the
    persona-filtered overload normalises the slug in a way that misses the
    stored ids on the bench data roots.
    """
    contacts = graph.get_all_contacts()
    out = []
    for c in contacts:
        d = c.__dict__ if hasattr(c, "__dict__") else {}
        if d.get("persona_id") != persona:
            continue
        name = d.get("name")
        if not name:
            continue
        status = str(d.get("status", "")).lower()
        # Confirmed people only: PatternStatus.CONTACT / "contact" / "confirmed".
        if "discard" in status:
            continue
        out.append(c)
    return out


def _labelled_spots(graph, persona: str) -> list:
    """Spots that carry a human-meaningful label, most-visited first."""
    spots = [
        s for s in graph.get_spots(persona)
        if (s.get("user_label") or s.get("location_name"))
        and str(s.get("status", "")).lower() != "discarded"
    ]
    spots.sort(key=lambda s: -(s.get("moment_count") or 0))
    return spots


def _resolve_reference_image(repos, moment_id: str, persona: str) -> Path | None:
    """Resolve a moment → its on-disk reference photo (basename under live dir)."""
    m = repos.moment.find_by_id(moment_id)
    if m is None:
        return None
    media_ids = getattr(m, "media_ids", None) or []
    photo_dir = _photo_dir(persona)
    for med_id in media_ids:
        media = repos.media.find_by_id(med_id)
        if media is None:
            continue
        basename = os.path.basename(str(media.path))
        candidate = photo_dir / basename
        if candidate.exists():
            return candidate
        # tolerate case/extension drift (.jpg vs .JPG etc.)
        stem = candidate.stem
        for p in photo_dir.glob(stem + ".*"):
            return p
    return None


def _activity_label(graph, moment_id: str) -> str | None:
    """The substrate's own dominant activity label for a moment (a verb phrase)."""
    try:
        acts = graph.find_activities_by_moment_id(moment_id) or []
    except Exception:  # noqa: BLE001
        return None
    # Prefer a depictable, present-progressive-friendly label; skip postures.
    skip = {"crouching", "standing", "sitting", "observing", "looking"}
    labels = [getattr(a, "label", None) for a in acts if getattr(a, "label", None)]
    for lab in labels:
        if lab.lower() not in skip:
            return lab
    return labels[0] if labels else None


def _spot_label(spot: dict) -> str:
    """Human-meaningful place label. Prefer the substrate's user_label; otherwise
    shorten a verbose reverse-geocoded location_name to its first two components
    (street + neighbourhood), which reads better in a generation prompt than a
    full address string."""
    lab = spot.get("user_label")
    if lab:
        return lab
    loc = spot.get("location_name") or spot.get("id")
    parts = [p.strip() for p in str(loc).split(",") if p.strip()]
    return ", ".join(parts[:2]) if len(parts) > 2 else str(loc)


def _entity_moment_ids(graph, contact_id: str) -> list[str]:
    moms = graph.find_moments_by_contact(contact_id) or []
    return [mm if isinstance(mm, str) else getattr(mm, "id", None) for mm in moms]


def _best_spot_for_moments(graph, persona: str, moment_ids: list[str]) -> dict | None:
    """The labelled spot that hosts the most of `moment_ids` (the entity's place)."""
    if not moment_ids:
        return None
    mset = set(moment_ids)
    best, best_n = None, 0
    for s in _labelled_spots(graph, persona):
        smoms = graph.get_moments_for_spot(s["id"]) or []
        smset = {mm if isinstance(mm, str) else getattr(mm, "id", None) for mm in smoms}
        n = len(mset & smset)
        if n > best_n:
            best, best_n = s, n
    return best


# --------------------------------------------------------------------------- #
# Public: build entity references for the diffusion track
# --------------------------------------------------------------------------- #
def build_entity_references(
    repos,
    persona: str,
    limit: int = 3,
) -> list[EntityReference]:
    """Assemble up to `limit` entity references (§7.3) from the substrate.

    Strategy: prefer Contacts (people) — IP-Adapter identity preservation is
    most meaningful on faces. For each named contact we resolve the
    representative reference photo, find the labelled spot they co-occur with
    most ("<other-entity>"), and pull the substrate's activity label
    ("<activity>"), yielding the relational prompt. Falls back to labelled Spots
    (place-as-entity) when too few people are available, so contact-sparse
    personas still produce demos.
    """
    graph = repos.graph
    refs: list[EntityReference] = []

    # ---- People (Contacts) -------------------------------------------------
    for c in _named_contacts(graph, persona):
        if len(refs) >= limit:
            break
        rfid = getattr(c, "representative_face_id", None)
        if not rfid or not str(rfid).startswith("face:"):
            continue
        src_moment = str(rfid).split(":")[1]
        img = _resolve_reference_image(repos, src_moment, persona)
        if img is None:
            logger.debug("No reference image for contact %s", c.name)
            continue

        moment_ids = _entity_moment_ids(graph, c.id)
        spot = _best_spot_for_moments(graph, persona, moment_ids) or (
            _labelled_spots(graph, persona)[:1] or [None]
        )[0]
        other = _spot_label(spot) if spot else None
        # activity from the source moment first, else any of the entity's moments
        activity = _activity_label(graph, src_moment)
        if not activity:
            for mid in moment_ids:
                activity = _activity_label(graph, mid)
                if activity:
                    break

        prompt = _compose_prompt(c.name, other, activity)
        refs.append(EntityReference(
            persona=persona, entity_kind="contact", entity_id=c.id,
            entity_label=c.name, reference_image_path=img, prompt=prompt,
            other_entity=other, activity=activity, source_moment_id=src_moment,
            debug={"n_moments": len(moment_ids)},
        ))

    # ---- Places (Spots) — fallback / supplement ----------------------------
    if len(refs) < limit:
        for s in _labelled_spots(graph, persona):
            if len(refs) >= limit:
                break
            label = _spot_label(s)
            smoms = graph.get_moments_for_spot(s["id"]) or []
            smids = [mm if isinstance(mm, str) else getattr(mm, "id", None) for mm in smoms]
            img = None
            activity = None
            for mid in smids:
                img = img or _resolve_reference_image(repos, mid, persona)
                activity = activity or _activity_label(graph, mid)
                if img and activity:
                    break
            if img is None:
                continue
            prompt = _compose_prompt(label, None, activity, place_entity=True)
            refs.append(EntityReference(
                persona=persona, entity_kind="spot", entity_id=s["id"],
                entity_label=label, reference_image_path=img, prompt=prompt,
                other_entity=None, activity=activity,
                source_moment_id=smids[0] if smids else None,
                debug={"moment_count": s.get("moment_count")},
            ))

    return refs[:limit]


def _compose_prompt(
    entity: str,
    other_entity: str | None,
    activity: str | None,
    place_entity: bool = False,
) -> str:
    """Spec §7.3 template: "<entity> at <other-entity> doing <activity>".

    Degrades gracefully when a slot is missing (contact-sparse / activity-sparse
    moments), but keeps the relational structure the substrate provides.
    """
    if place_entity:
        # entity *is* a place; phrase as a scene at that place.
        base = f"a photo at {entity}"
        if activity:
            base += f", someone {activity}"
        return base + ", realistic, natural lighting"

    base = f"a photo of {entity}"
    if other_entity:
        base += f" at {other_entity}"
    if activity:
        base += f" {activity}"
    return base + ", realistic portrait, natural lighting"
