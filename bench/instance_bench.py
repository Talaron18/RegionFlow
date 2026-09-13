"""
Build the same-category instance-selectivity benchmark.

THE GAP. Stage 0 could not compute its own leakage metric, for a reason that
has nothing to do with Stage 0: no editing benchmark ships masks for the
NON-selected instances of the target's category. PIE-Bench and MagicBrush both
stratify by mask size; neither stratifies by instance multiplicity. So "did the
edit hit the other three cans" has never been measurable, which means nobody
has reported it, which is why it is worth measuring.

WHAT MAKES IT BUILDABLE NOW. ImgEdit ships object-level annotation -- counts,
boxes, segmentation -- as a separate artifact (ImgEdit_recap_mask). Joining it
against the edit triplets gives, for each edit, every instance of the target's
category and which one was edited. That join is the benchmark. HumanEdit's
Counting and Relation splits are a second, independent source and should be
used as a held-out check rather than merged, so a systematic bias in one
annotation pipeline does not become the benchmark's bias.

NO GPU. This is a filtering and bookkeeping job over annotation files. It runs
on the CPU while the gates are training, and it is the deliverable that
survives regardless of how the gates come out: a benchmark plus the measurement
that existing few-step editors leak is a contribution on its own.

TARGET SIZE. 300-500 cases, stratified. Not thousands. The metric has to be
verifiable by hand -- Stage 0's rule that automated verdicts were wrong every
time they fired applies with full force to a benchmark nobody has validated.
Budget for annotating a 50-case audit subset by hand and reporting agreement.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class InstanceCase:
    """One benchmark item.

    target_mask      the instance the instruction refers to
    sibling_masks    OTHER instances of the SAME category, not to be changed
    distractor_masks instances of DIFFERENT categories, also not to be changed
    """

    case_id: str
    image_path: str
    category: str
    instruction: str
    target_mask_ref: str
    sibling_mask_refs: list = field(default_factory=list)
    distractor_mask_refs: list = field(default_factory=list)
    n_siblings: int = 0
    target_area_frac: float = 0.0
    sibling_area_frac: float = 0.0
    source_dataset: str = ""
    notes: str = ""


# --------------------------------------------------------------- selection --

MIN_SIBLINGS = 1
MIN_AREA_FRAC = 0.005       # below this SAM masks are unreliable and dE is noise
MAX_AREA_FRAC = 0.40        # above this "background" is barely defined
MAX_IOU_WITH_SIBLING = 0.10  # overlapping instances make leakage unattributable


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def qualifies(target: np.ndarray, siblings: list, image_area: int) -> str | None:
    """Return a rejection reason, or None if the case is usable."""
    frac = target.sum() / image_area
    if len(siblings) < MIN_SIBLINGS:
        return f"only {len(siblings)} same-category siblings"
    if frac < MIN_AREA_FRAC:
        return f"target area {frac:.4f} below floor"
    if frac > MAX_AREA_FRAC:
        return f"target area {frac:.4f} above ceiling"
    for s in siblings:
        if iou(target, s) > MAX_IOU_WITH_SIBLING:
            return "target overlaps a sibling; leakage would be unattributable"
        if s.sum() / image_area < MIN_AREA_FRAC:
            return "a sibling is below the area floor; dE on it would be noise"
    return None


def build(
    records,
    load_masks,
    out_path: str | Path,
    source_dataset: str,
    limit: int | None = None,
) -> dict:
    """Filter annotated records into benchmark cases.

    records:    iterable of dicts with at least
                {id, image_path, instruction, category, target_id, instances:
                 [{id, category, mask_ref}]}
    load_masks: callable(record) -> {instance_id: bool ndarray}
    """
    cases, rejected = [], defaultdict(int)

    for rec in records:
        if limit and len(cases) >= limit:
            break
        try:
            masks = load_masks(rec)
        except Exception as exc:                       # noqa: BLE001
            rejected[f"mask load failed: {type(exc).__name__}"] += 1
            continue

        tid = rec["target_id"]
        if tid not in masks:
            rejected["target mask missing"] += 1
            continue
        target = masks[tid]
        area = int(target.size)

        cat = rec["category"]
        sibs, dists = [], []
        for inst in rec["instances"]:
            if inst["id"] == tid or inst["id"] not in masks:
                continue
            (sibs if inst["category"] == cat else dists).append(inst)

        reason = qualifies(target, [masks[s["id"]] for s in sibs], area)
        if reason:
            rejected[reason.split(";")[0]] += 1
            continue

        cases.append(InstanceCase(
            case_id=str(rec["id"]),
            image_path=str(rec["image_path"]),
            category=cat,
            instruction=rec["instruction"],
            target_mask_ref=str(masks_ref(rec, tid)),
            sibling_mask_refs=[str(masks_ref(rec, s["id"])) for s in sibs],
            distractor_mask_refs=[str(masks_ref(rec, d["id"])) for d in dists],
            n_siblings=len(sibs),
            target_area_frac=float(target.sum() / area),
            sibling_area_frac=float(
                sum(masks[s["id"]].sum() for s in sibs) / area),
            source_dataset=source_dataset,
        ))

    out = {
        "source_dataset": source_dataset,
        "n_cases": len(cases),
        "rejected": dict(rejected),
        "selection": {
            "min_siblings": MIN_SIBLINGS,
            "min_area_frac": MIN_AREA_FRAC,
            "max_area_frac": MAX_AREA_FRAC,
            "max_iou_with_sibling": MAX_IOU_WITH_SIBLING,
        },
        "cases": [asdict(c) for c in cases],
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def masks_ref(rec, instance_id) -> str:
    return f"{rec['id']}::{instance_id}"


# ----------------------------------------------------------------- metric --

def instance_selectivity(source, edited, target_mask, sibling_masks,
                         eps: float = 1e-6) -> dict:
    """The number the benchmark exists to produce.

    leakage_ratio = mean |change| on siblings / mean |change| on target

        0.0   perfect: siblings untouched
        1.0   the edit hit siblings as hard as the target -- category-level,
              not instance-level, editing

    Reported as a RATIO rather than an absolute, because an edit that barely
    moves anything would otherwise score well on an absolute sibling change.
    That is the same degenerate-null trap `decoupling.guard_degenerate`
    catches, and it will fire here too if target change is near zero -- hence
    the explicit `valid` flag rather than a silent division.
    """
    src = np.asarray(source, np.float32) / 255.0
    ed = np.asarray(edited, np.float32) / 255.0
    diff = np.abs(src - ed).mean(axis=-1)

    tm = np.asarray(target_mask) > 0
    tgt_change = float(diff[tm].mean()) if tm.any() else 0.0

    sib_changes = []
    for sm in sibling_masks:
        sm = np.asarray(sm) > 0
        if sm.any():
            sib_changes.append(float(diff[sm].mean()))

    union = tm.copy()
    for sm in sibling_masks:
        union |= np.asarray(sm) > 0
    bg = float(diff[~union].mean()) if (~union).any() else 0.0

    sib = float(np.mean(sib_changes)) if sib_changes else 0.0
    valid = tgt_change > 10 * (bg + eps)

    return {
        "target_change": tgt_change,
        "sibling_change": sib,
        "background_change": bg,
        "leakage_ratio": sib / (tgt_change + eps),
        # Sibling change above background is the sharper question: an edit can
        # have a low ratio simply because the whole image shifted slightly.
        "sibling_excess_over_bg": sib - bg,
        "n_siblings": len(sib_changes),
        "valid": bool(valid),
        "invalid_reason": None if valid else
        "target change is not distinguishable from background; the edit did "
        "not take, so leakage is not measurable on this case",
    }


def aggregate(results: list) -> dict:
    usable = [r for r in results if r["valid"]]
    if not usable:
        return {"n": 0, "n_valid": 0, "note": "no case had a measurable edit"}
    ratios = np.array([r["leakage_ratio"] for r in usable])
    return {
        "n": len(results),
        "n_valid": len(usable),
        "leakage_ratio_mean": float(ratios.mean()),
        "leakage_ratio_median": float(np.median(ratios)),
        "leakage_ratio_p90": float(np.percentile(ratios, 90)),
        "frac_cases_ratio_above_0p5": float((ratios > 0.5).mean()),
        "sibling_excess_mean": float(
            np.mean([r["sibling_excess_over_bg"] for r in usable])),
    }
