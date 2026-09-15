"""
MagicBrush x COCO: the join that makes the leakage metric computable.

WHY THIS EXISTS. Stage 0 recorded that "no benchmark ships masks for the
non-selected instances of the target's category". That is true of every
*editing* benchmark, and it is why Gate C sat unrun for three stages. It is not
true of MagicBrush, because MagicBrush's source images are sampled from MS COCO
and its `img_id` IS the COCO image id. COCO ships every instance of every one of
its 80 thing classes. So the sibling masks were always there; nothing had
followed the id back.

Verified on dev before this was written: 266/266 turn-1 images resolve to COCO,
142 of them (53%) have >=2 same-category instances above the 0.5% area floor,
and 91 of those groups have 3+ instances.

WHAT IT PRODUCES. `records` and `load_masks`, matching the contract that
`instance_bench.build()` already expects:

    records:    {id, image_path, instruction, category, target_id, instances}
    load_masks: callable(record) -> {instance_id: bool ndarray}

MagicBrush is also the only source in play that has a ground-truth EDITED image,
so it is the only one on which the leakage metric can be calibrated at all
(RefEdit-Bench and MIRA-Bench are both VLM-scored precisely because they have no
GT target). `target_img` is carried through for that.

THE THREE FILTERS, in the order they reject.

  1. multi-instance      the image has >=2 instances of some category, above the
                         area floor. Loose: 53% of dev.
  2. instruction match   the instruction names one of THOSE categories. This is
                         the filter nobody has measured, and the one that
                         decides whether train is worth 25GB.
  3. target resolution   the MagicBrush edit mask overlaps one specific COCO
                         instance. Below MIN_TARGET_IOU the edit did not land on
                         an annotated instance and the case is not an instance
                         edit, whatever the instruction says.

Filter 3's rejection count is the number that matters. Filters 1 and 2 only
establish that the instruction names a category that happens to have siblings.

GEOMETRY IS THE RISK. MagicBrush images are DALL-E 2 outputs and need not share
COCO's resolution or aspect ratio. Every mask is resampled to the MagicBrush
frame with nearest-neighbour, and `aspect_mismatch` counts cases where the two
aspect ratios disagree by more than ASPECT_TOL. If that counter is large, the
IoU join is being computed across a geometric distortion and NOTHING
downstream is trustworthy -- read it before reading the case count.

NOTE on `instance_bench.py`: `qualifies()` interpolates the area value into its
return string, so `build()`'s `rejected` histogram gets a unique key per case.
Key on a stable prefix before reading that histogram. This module keeps its own
counters and does not depend on it.

USAGE

    python magicbrush_coco.py --split dev --overlays 12
    python magicbrush_coco.py --split dev --build --out bench/cases_dev.json
    python magicbrush_coco.py --self-test        # geometry only, no data needed
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------- constants --

MIN_AREA_FRAC = 0.005      # matches instance_bench.MIN_AREA_FRAC
MIN_SIBLINGS = 1           # >=1 sibling == >=2 same-category instances
MIN_TARGET_IOU = 0.30      # legacy IoU path only; superseded by coverage
#: Filter 3, measured on dev rather than guessed. `best coverage` is bimodal --
#: 24 rows in [0.9,1.0], 21 in [0.0,0.2], and [0.5,0.9] essentially empty -- so
#: the coverage threshold sits in a wide valley and its exact value is not
#: load-bearing. The MARGIN is: of 25 rows clearing coverage, only 9 have a
#: single instance covered. The rest have the inpainting blob spanning 2-5
#: same-category instances, and everything inside a DALL-E 2 inpainting region
#: was regenerated, so those siblings are contaminated by construction.
#: Accepted count is flat at 9 across margin 0.2-0.4 and coverage 0.5-0.9.
MIN_COVERAGE = 0.70
MIN_MARGIN = 0.20
ASPECT_TOL = 0.05          # relative aspect-ratio disagreement tolerated
MAGICBRUSH_ALPHA_EDIT = 128  # alpha < 128 is the edit region. NOT the RGB.

#: COCO class name -> extra surface forms. Only where the plural rule or the
#: everyday word differs from the class name. Deliberately conservative: a
#: wrong synonym silently mis-assigns a category, and this is a benchmark.
SYNONYMS = {
    "tv": ["tv", "television", "monitor", "screen"],
    "couch": ["couch", "sofa", "settee"],
    "potted plant": ["potted plant", "plant", "houseplant", "pot plant"],
    "dining table": ["dining table", "table"],
    "cell phone": ["cell phone", "phone", "mobile phone", "smartphone"],
    "motorcycle": ["motorcycle", "motorbike", "bike"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "airplane": ["airplane", "aeroplane", "plane", "jet"],
    "sports ball": ["sports ball", "ball"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer"],
    "wine glass": ["wine glass", "wineglass"],
    "remote": ["remote", "remote control"],
    "mouse": ["mouse", "computer mouse"],
    "person": ["person", "man", "woman", "boy", "girl", "guy", "lady",
               "child", "kid", "people", "men", "women", "children", "kids"],
}

#: Plurals the -s/-es rule gets wrong.
IRREGULAR_PLURAL = {
    "person": "people", "child": "children", "man": "men", "woman": "women",
    "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
    "scissors": "scissors", "broccoli": "broccoli", "sandwich": "sandwiches",
    "bus": "buses", "glass": "glasses", "couch": "couches", "bench": "benches",
}


def surface_forms(cat_name: str) -> list[str]:
    """Every string that should count as naming this COCO category."""
    base = SYNONYMS.get(cat_name, [cat_name])
    out = set()
    for w in base:
        out.add(w)
        if w in IRREGULAR_PLURAL:
            out.add(IRREGULAR_PLURAL[w])
        elif w.endswith(("s", "x", "ch", "sh")):
            out.add(w + "es")
        elif w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            out.add(w[:-1] + "ies")
        else:
            out.add(w + "s")
    return sorted(out)


_FORM_CACHE: dict[str, re.Pattern] = {}


def names_category(instruction: str, cat_name: str) -> bool:
    """Whole-word match of any surface form of `cat_name` in `instruction`."""
    pat = _FORM_CACHE.get(cat_name)
    if pat is None:
        alt = "|".join(re.escape(f) for f in surface_forms(cat_name))
        pat = _FORM_CACHE[cat_name] = re.compile(rf"\b(?:{alt})\b", re.I)
    return bool(pat.search(instruction))


# ---------------------------------------------------------------- geometry --

def poly_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Rasterise a COCO polygon annotation.

    Uses pycocotools when importable (handles holes correctly via the even-odd
    rule); otherwise fills each polygon, which slightly over-covers donut
    shapes. The fallback keeps this runnable on a box without the C extension.
    """
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for poly in segmentation:
        if len(poly) >= 6:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def ann_to_mask(ann, height: int, width: int) -> np.ndarray:
    seg = ann.get("segmentation")
    if isinstance(seg, list):
        try:
            from pycocotools import mask as mask_utils  # noqa: PLC0415
            rles = mask_utils.frPyObjects(seg, height, width)
            return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
        except Exception:  # noqa: BLE001 -- fall back to the PIL rasteriser
            return poly_to_mask(seg, height, width)
    # iscrowd RLE. These are filtered out upstream; decode only if reached.
    from pycocotools import mask as mask_utils  # noqa: PLC0415
    return mask_utils.decode(seg).astype(bool)


def resize_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    if mask.shape == (h, w):
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize((w, h), Image.NEAREST), dtype=bool)


#: How the COCO frame maps into the MagicBrush frame. MagicBrush images are
#: 1024x1024 DALL-E outputs; COCO images are not square, so SOMETHING was done
#: to make them square, and a plain stretch is only one of the possibilities.
#: All three agree exactly at the frame centre and diverge toward the edges --
#: which is why a wrong choice here looks like intermittent drift rather than a
#: constant offset, and why the centre-most case in a batch always looks fine.
#: Do not assume: `--probe-geometry` measures which one the data actually used.
GEOMETRIES = ("stretch", "center_crop", "pad")


def map_to_edit_frame(mask: np.ndarray, coco_hw, edit_hw, mode: str) -> np.ndarray:
    ch, cw = coco_hw
    if mode == "stretch":
        m = mask
    elif mode == "center_crop":
        s = min(ch, cw)
        y0, x0 = (ch - s) // 2, (cw - s) // 2
        m = mask[y0:y0 + s, x0:x0 + s]
    elif mode == "pad":
        s = max(ch, cw)
        m = np.zeros((s, s), dtype=bool)
        y0, x0 = (s - ch) // 2, (s - cw) // 2
        m[y0:y0 + ch, x0:x0 + cw] = mask
    else:
        raise ValueError(f"unknown geometry {mode!r}; expected one of {GEOMETRIES}")
    return resize_mask(m, edit_hw)


def apply_rect(mask: np.ndarray, rect, out_hw) -> np.ndarray:
    """Map a COCO-frame mask through a fitted rectangle. Mirrors frame_fit."""
    try:
        from frame_fit import apply_rect as _ar  # noqa: PLC0415
        return _ar(mask, rect, out_hw)
    except ImportError:
        pass
    x0, y0, w, h = (int(round(v)) for v in rect)
    H, W = mask.shape
    x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1))
    sub = mask[y0:min(H, y0 + max(1, h)), x0:min(W, x0 + max(1, w))]
    oh, ow = out_hw
    return np.asarray(Image.fromarray(sub.astype(np.uint8) * 255)
                      .resize((ow, oh), Image.NEAREST), dtype=bool)


def load_frames(path: str | Path) -> dict:
    """Per-image crop rectangles from frame_fit.py.

    263/266 dev rectangles are squares of side min(W,H), but the offset along
    the long axis spans the full range -- only 11% are near-centred. There is
    no global convention to assume, which is why this file exists.
    """
    frames = json.loads(Path(path).read_text())
    v = np.array([f["ncc_fit"] for f in frames.values()])
    print(f"frames: {len(frames)} fitted, ncc_fit median {np.median(v):.4f}, "
          f"min {v.min():.4f}")
    return frames


def coverage(edit: np.ndarray, inst: np.ndarray) -> float:
    """|edit & inst| / |inst| -- how much of the INSTANCE the edit region covers.

    The right statistic when the edit mask is loose. MagicBrush masks are
    crowdworker DALL-E 2 inpainting regions, median 10% of frame and p90 34%,
    so a correctly aligned tight COCO instance inside one caps out at roughly
    area(inst)/area(edit) on IoU -- which is exactly the 0.10 median IoU that
    read as misalignment. Coverage is ~1.0 for a contained instance regardless
    of how generous the blob around it is.

    Coverage alone is not sufficient: a blob spanning the frame covers every
    instance. The margin over the best SIBLING is what makes it discriminative.
    """
    n = int(inst.sum())
    return float((edit & inst).sum() / n) if n else 0.0


def precision(edit: np.ndarray, inst: np.ndarray) -> float:
    """|edit & inst| / |edit| -- how much of the EDIT lands on the instance."""
    n = int(edit.sum())
    return float((edit & inst).sum() / n) if n else 0.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def decode_edit_mask(mask_bytes: bytes) -> np.ndarray:
    """MagicBrush's mask is the ALPHA channel: alpha < 128 is the edit region.

    Not the RGB channels. This cost Stage 0 time and is asserted here rather
    than commented, because a mask read from RGB is ~the complement and every
    IoU downstream would be quietly wrong.
    """
    img = Image.open(io.BytesIO(mask_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = np.asarray(img)[..., 3]
    return alpha < MAGICBRUSH_ALPHA_EDIT


# -------------------------------------------------------------- coco index --

def load_coco(ann_dir: str | Path, splits=("train2014", "val2014")) -> tuple[dict, dict]:
    """{image_id: {h, w, anns}}, {category_id: name}. ~2GB resident for both."""
    index, cats = {}, {}
    ann_dir = Path(ann_dir)
    for split in splits:
        path = ann_dir / f"instances_{split}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Expected COCO annotations unzipped there; "
                "annotations_trainval2014.zip provides both splits.")
        blob = json.loads(path.read_text())
        cats.update({c["id"]: c["name"] for c in blob["categories"]})
        for im in blob["images"]:
            index[im["id"]] = {"h": im["height"], "w": im["width"], "anns": []}
        for a in blob["annotations"]:
            if not a.get("iscrowd") and a["image_id"] in index:
                index[a["image_id"]]["anns"].append(a)
    return index, cats


def norm_img_id(raw) -> int | None:
    """MagicBrush's img_id is a STRING ('140513'). Every join needs this."""
    digits = "".join(c for c in str(raw) if c.isdigit())
    return int(digits) if digits else None


# ------------------------------------------------------------ magicbrush io --

def read_magicbrush(root: str | Path, split: str = "dev", turn: int = 1) -> list[dict]:
    """Turn-1 rows only. Later turns have a model output as their source."""
    import pyarrow.parquet as pq  # noqa: PLC0415 -- keeps --self-test dep-free

    files = sorted(glob.glob(str(Path(root) / "data" / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no {split}-*.parquet under {root}/data. `hf download --include` "
            "exits 0 when its pattern matches nothing; check the file list.")
    table = pq.read_table(files)
    cols = table.schema.names
    for need in ("img_id", "turn_index", "instruction", "mask_img", "source_img"):
        if need not in cols:
            raise KeyError(f"column {need!r} absent; parquet has {cols}")
    data = table.to_pydict()

    rows = []
    for i in range(table.num_rows):
        if data["turn_index"][i] != turn:
            continue
        rows.append({
            "img_id": data["img_id"][i],
            "instruction": data["instruction"][i],
            "mask_img": data["mask_img"][i],
            "source_img": data["source_img"][i],
            "target_img": data.get("target_img", [None] * table.num_rows)[i],
        })
    return rows


def _img_bytes(cell):
    """HF Image features arrive as {'bytes': ..., 'path': ...}."""
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
        raise ValueError("image cell has neither bytes nor path")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unhandled image cell type {type(cell).__name__}")


# ------------------------------------------------------------------ matching --

def prefilter(row: dict, coco: dict, cats: dict, stats: collections.Counter):
    """Filters 1 and 2 plus the edit-mask decode. Geometry-independent.

    Returns (rec, per_cat, named, edit) or None. Shared by resolve_case and
    probe_geometry so the two cannot drift apart.
    """
    img_id = norm_img_id(row["img_id"])
    rec = coco.get(img_id)
    if rec is None:
        stats["no COCO image for img_id"] += 1
        return None

    area = rec["h"] * rec["w"]

    # filter 1 -- categories with >=2 instances above the area floor
    per_cat = collections.defaultdict(list)
    for a in rec["anns"]:
        if a["area"] / area >= MIN_AREA_FRAC:
            per_cat[a["category_id"]].append(a)
    multi = {c: v for c, v in per_cat.items() if len(v) >= MIN_SIBLINGS + 1}
    if not multi:
        stats["no category with >=2 instances above area floor"] += 1
        return None

    # filter 2 -- the instruction names one of those categories
    named = [c for c in multi if names_category(row["instruction"], cats[c])]
    if not named:
        stats["instruction names no multi-instance category"] += 1
        return None
    if len(named) > 1:
        stats["_ambiguous instruction (resolved by mask)"] += 1

    edit = decode_edit_mask(_img_bytes(row["mask_img"]))
    return rec, per_cat, multi, named, edit


def resolve_case(row: dict, coco: dict, cats: dict, stats: collections.Counter,
                 geometry: str = "center_crop", frames: dict | None = None,
                 min_ncc: float = 0.95, min_coverage: float = MIN_COVERAGE,
                 min_margin: float = MIN_MARGIN):
    """Apply the three filters to one MagicBrush row.

    With `frames`, masks are mapped through that image's fitted rectangle and
    filter 3 is coverage-plus-margin. Without, it falls back to the global
    geometry and IoU -- kept only so the old path stays reproducible; the dev
    measurement says it accepts less than half as many cases and its median
    best-IoU on correct matches is 0.10.

    Returns (record, masks) or (None, None), incrementing `stats` with the
    reason. Reasons are STABLE STRINGS -- no interpolated values -- so the
    histogram aggregates.
    """
    pre = prefilter(row, coco, cats, stats)
    if pre is None:
        return None, None
    rec, per_cat, multi, named, edit = pre

    ch, cw = rec["h"], rec["w"]
    eh, ew = edit.shape

    fr = None
    if frames is not None:
        fr = frames.get(str(row["img_id"]))
        if not fr:
            stats["no fitted frame for this image"] += 1
            return None, None
        if fr["ncc_fit"] < min_ncc:
            stats["frame did not register"] += 1
            return None, None

    def to_edit(ann):
        m = ann_to_mask(ann, ch, cw)
        if fr is not None:
            return apply_rect(m, fr["rect"], (eh, ew))
        return map_to_edit_frame(m, (ch, cw), (eh, ew), geometry)

    # filter 3 -- which specific instance did the edit land on?
    scored, masks = [], {}
    for cid in named:
        for a in multi[cid]:
            m = to_edit(a)
            masks[str(a["id"])] = m
            scored.append((coverage(edit, m) if fr is not None else iou(edit, m),
                           cid, a, m))
    if not scored:
        stats["edit mask matches no annotated instance"] += 1
        return None, None

    scored.sort(key=lambda t: t[0], reverse=True)
    top, best_cat, best, _ = scored[0]

    if fr is not None:
        runner = scored[1][0] if len(scored) > 1 else 0.0
        if top < min_coverage:
            stats["edit region contains no instance"] += 1
            return None, None
        if top - runner < min_margin:
            stats["edit region spans several same-category instances"] += 1
            return None, None
        score_fields = {
            "coverage": round(top, 4),
            "sibling_coverage": round(runner, 4),
            "margin": round(top - runner, 4),
            "precision": round(precision(edit, masks[str(best["id"])]), 4),
            "target_iou": round(iou(edit, masks[str(best["id"])]), 4),
            "ncc_fit": fr["ncc_fit"],
            "n_covered": sum(1 for s in scored if s[0] >= min_coverage),
        }
    else:
        if top < MIN_TARGET_IOU:
            stats["edit mask matches no annotated instance"] += 1
            return None, None
        score_fields = {"target_iou": round(top, 4)}

    # keep only the winning category: siblings are SAME-category by definition
    keep = {str(a["id"]) for a in multi[best_cat]}
    masks = {k: v for k, v in masks.items() if k in keep}
    for cid, anns in per_cat.items():          # distractors: other categories
        if cid == best_cat:
            continue
        for a in anns:
            masks[str(a["id"])] = to_edit(a)

    record = {
        "id": f"mb-{row['img_id']}",
        "image_path": f"magicbrush::{row['img_id']}",
        "instruction": row["instruction"],
        "category": cats[best_cat],
        "target_id": str(best["id"]),
        "instances": [
            {"id": str(a["id"]), "category": cats[cid], "mask_ref": str(a["id"])}
            for cid, anns in per_cat.items() for a in anns
        ],
        "n_siblings": len(multi[best_cat]) - 1,
        "coco_image_id": norm_img_id(row["img_id"]),
        "geometry": "fitted_rect" if fr is not None else geometry,
        "rect": fr["rect"] if fr is not None else None,
        "edit_hw": [eh, ew],
        "coco_hw": [ch, cw],
        **score_fields,
    }
    stats["ACCEPTED"] += 1
    return record, masks


def probe_coverage(rows, coco, cats, frames, limit: int = 300, min_ncc: float = 0.95):
    """Measure the coverage distribution so the thresholds are read, not guessed.

    For each row clearing filters 1-2 with a registered frame, compute coverage
    of every instance of a named multi-instance category, then report:

      best        coverage of the best-covered instance
      margin      best minus the SECOND best (the best sibling)
      precision   fraction of the edit region landing on the best instance

    If target resolution works, `best` is bimodal -- a pile near 1.0 where the
    edit contains an instance, and a pile near 0 where it does not -- and the
    thresholds fall in the valley. A unimodal smear means coverage is no better
    than IoU was, and filter 3 needs a different idea rather than a new number.
    """
    stats = collections.Counter()
    best_l, second_l, margin_l, prec_l, iou_l, nover_l = [], [], [], [], [], []

    for row in rows:
        if len(best_l) >= limit:
            break
        pre = prefilter(row, coco, cats, stats)
        if pre is None:
            continue
        rec, _per_cat, multi, named, edit = pre
        fr = frames.get(str(row["img_id"]))
        if not fr or fr["ncc_fit"] < min_ncc:
            stats["no registered frame"] += 1
            continue

        ch, cw = rec["h"], rec["w"]
        eh, ew = edit.shape
        covs = []
        for cid in named:
            for a in multi[cid]:
                m = apply_rect(ann_to_mask(a, ch, cw), fr["rect"], (eh, ew))
                covs.append((coverage(edit, m), precision(edit, m), iou(edit, m)))
        if not covs:
            continue
        covs.sort(reverse=True)
        best_l.append(covs[0][0])
        prec_l.append(covs[0][1])
        iou_l.append(covs[0][2])
        second_l.append(covs[1][0] if len(covs) > 1 else 0.0)
        margin_l.append(covs[0][0] - (covs[1][0] if len(covs) > 1 else 0.0))
        nover_l.append(sum(1 for c in covs if c[0] >= 0.7))

    n = len(best_l)
    if not n:
        print("no rows reached the coverage probe")
        return {}
    best, margin, prec = np.array(best_l), np.array(margin_l), np.array(prec_l)
    second, iou_a = np.array(second_l), np.array(iou_l)

    def hist(name, v):
        print(f"\n  {name}")
        counts, edges = np.histogram(v, bins=10, range=(0, 1))
        for c, lo in zip(counts, edges[:-1]):
            bar = "#" * int(round(40 * c / max(counts.max(), 1)))
            print(f"    {lo:.1f}-{lo + 0.1:.1f} {bar:<40} {c}")

    print("\n" + "=" * 70)
    print(f"coverage probe over {n} rows clearing filters 1-2 with a frame")
    print("=" * 70)
    for name, v in (("best coverage", best), ("best SIBLING coverage", second),
                    ("margin (best - sibling)", margin),
                    ("precision of best", prec), ("best IoU (the old stat)", iou_a)):
        print(f"  {name:<26} median {np.median(v):.3f}  mean {v.mean():.3f}"
              f"  p90 {np.percentile(v, 90):.3f}")
    hist("best coverage", best)
    hist("margin over best sibling", margin)

    print(f"\n  instances at coverage >= 0.7 per row: "
          f"{sorted(collections.Counter(nover_l).items())}")
    print("    (rows with several are the ambiguous ones the margin must reject)")

    print("\n  decision table -- rows accepted at each (coverage, margin):\n")
    cov_grid = (0.5, 0.6, 0.7, 0.8, 0.9)
    mar_grid = (0.0, 0.1, 0.2, 0.3, 0.4)
    print("        margin " + "".join(f"{m:>8.1f}" for m in mar_grid))
    for c in cov_grid:
        cells = [int(((best >= c) & (margin >= m)).sum()) for m in mar_grid]
        print(f"  cov >= {c:<4} " + "".join(f"{x:>8}" for x in cells))
    print(f"\n  (of {n} rows; the old IoU>=0.30 rule accepted "
          f"{int((iou_a >= 0.30).sum())})")
    return {"best": best, "margin": margin, "precision": prec, "iou": iou_a}


def probe_geometry(rows, coco, cats, limit: int = 120) -> dict:
    """Measure which frame mapping the data actually used.

    For every row clearing filters 1-2, score each candidate geometry by the
    best IoU it achieves between the edit mask and any instance of a named
    category. The right transform should win by a wide margin; a narrow one
    means the sample is centre-heavy and the question is not yet settled.
    """
    scores = {g: [] for g in GEOMETRIES}
    sink = collections.Counter()
    for row in rows:
        pre = prefilter(row, coco, cats, sink)
        if pre is None:
            continue
        rec, _per_cat, multi, named, edit = pre
        ch, cw = rec["h"], rec["w"]
        eh, ew = edit.shape
        anns = [a for cid in named for a in multi[cid]]
        for g in GEOMETRIES:
            scores[g].append(max(
                iou(edit, map_to_edit_frame(ann_to_mask(a, ch, cw), (ch, cw), (eh, ew), g))
                for a in anns))
        if len(scores["stretch"]) >= limit:
            break

    n = len(scores["stretch"])
    print(f"\ngeometry probe over {n} rows clearing filters 1-2:\n")
    print(f"  {'mode':<13} {'mean':>7} {'median':>7} {'>=0.5':>7} {'>=0.8':>7}")
    out = {}
    for g in GEOMETRIES:
        v = np.array(scores[g]) if n else np.zeros(1)
        out[g] = float(v.mean())
        print(f"  {g:<13} {v.mean():>7.3f} {np.median(v):>7.3f} "
              f"{(v >= 0.5).mean():>7.1%} {(v >= 0.8).mean():>7.1%}")
    win = max(out, key=out.get)
    runner = sorted(out.values())[-2]
    print(f"\n  winner: {win} (mean {out[win]:.3f}, "
          f"{out[win] - runner:+.3f} over next)")
    if out[win] - runner < 0.05:
        print("  MARGIN IS NARROW -- inspect overlays under each mode before choosing")
    return out


# --------------------------------------------------------------- overlays --

def write_overlay(row, record, masks, out_dir: Path):
    """Target in green, siblings in red. Look at these before recording a number."""
    src = Image.open(io.BytesIO(_img_bytes(row["source_img"]))).convert("RGB")
    eh, ew = record["edit_hw"]
    if src.size != (ew, eh):
        src = src.resize((ew, eh), Image.BILINEAR)
    arr = np.asarray(src).astype(np.float32)

    tgt = masks[record["target_id"]]
    sib_ids = [i["id"] for i in record["instances"]
               if i["category"] == record["category"] and i["id"] != record["target_id"]]
    sib = np.zeros_like(tgt)
    for sid in sib_ids:
        if sid in masks:
            sib |= masks[sid]

    arr[tgt] = 0.45 * arr[tgt] + 0.55 * np.array([60, 220, 90], np.float32)
    arr[sib] = 0.45 * arr[sib] + 0.55 * np.array([235, 60, 60], np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{record['id']}_{record['category'].replace(' ', '-')}_iou{record['target_iou']:.2f}.png"
    Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)


# ------------------------------------------------------------------- main --

def run(args) -> dict:
    coco, cats = load_coco(args.coco_annotations)
    rows = read_magicbrush(args.magicbrush, args.split)
    print(f"{len(rows)} turn-1 rows in {args.split}; "
          f"{len(coco)} COCO images indexed")

    geometry = args.geometry
    frames = load_frames(args.frames) if args.frames else None

    if args.probe_coverage:
        if not frames:
            raise SystemExit("--probe-coverage needs --frames from frame_fit.py")
        probe_coverage(rows, coco, cats, frames, limit=args.probe_limit,
                       min_ncc=args.min_ncc)
        return {}

    if args.probe_geometry or (geometry == "auto" and not frames):
        probe = probe_geometry(rows, coco, cats, limit=args.probe_limit)
        if geometry == "auto":
            geometry = max(probe, key=probe.get)
            print(f"\nusing geometry={geometry}")
        if args.probe_geometry:
            return {"probe": probe}
    if frames:
        print(f"\nfitted frames, coverage>={args.min_coverage}, "
              f"margin>={args.min_margin}, ncc>={args.min_ncc}")
    else:
        print(f"\ngeometry={geometry}, min_target_iou={MIN_TARGET_IOU} "
              "(legacy path -- prefer --frames)")

    stats = collections.Counter()
    records, all_masks, kept_rows = [], {}, []
    for row in rows:
        if args.limit and len(records) >= args.limit:
            break
        rec, masks = resolve_case(row, coco, cats, stats, geometry=geometry,
                                  frames=frames, min_ncc=args.min_ncc,
                                  min_coverage=args.min_coverage,
                                  min_margin=args.min_margin)
        if rec is None:
            continue
        records.append(rec)
        all_masks[rec["id"]] = masks
        kept_rows.append(row)

    n = len(rows)
    print(f"\nACCEPTED {stats['ACCEPTED']}/{n} "
          f"({stats['ACCEPTED'] / n:.1%})\n")
    print("rejections, in filter order:")
    for reason in ("no COCO image for img_id",
                   "no category with >=2 instances above area floor",
                   "instruction names no multi-instance category",
                   "edit mask matches no annotated instance",
                   "no fitted frame for this image",
                   "frame did not register",
                   "edit region contains no instance",
                   "edit region spans several same-category instances"):
        if stats[reason]:
            print(f"  {stats[reason]:>4}  {reason}")
    print(f"\n  aspect_mismatch (>{ASPECT_TOL:.0%}): {stats['aspect_mismatch']}"
          "   <- if large, the IoU join is across a distortion; stop here")
    print(f"  ambiguous instructions resolved by mask: "
          f"{stats['_ambiguous instruction (resolved by mask)']}")

    if records:
        cat_hist = collections.Counter(r["category"] for r in records)
        sib_hist = collections.Counter(r["n_siblings"] for r in records)
        ious = np.array([r["target_iou"] for r in records])
        print(f"\ncategories: {cat_hist.most_common(12)}")
        print(f"siblings per case: {sorted(sib_hist.items())}")
        print(f"target IoU: min {ious.min():.2f} "
              f"median {np.median(ious):.2f} max {ious.max():.2f}")
        if "coverage" in records[0]:
            cov = np.array([r["coverage"] for r in records])
            mar = np.array([r["margin"] for r in records])
            print(f"coverage:   min {cov.min():.2f} median {np.median(cov):.2f}")
            print(f"margin:     min {mar.min():.2f} median {np.median(mar):.2f}")

    if args.overlays:
        out = Path(args.overlay_dir)
        for row, rec in list(zip(kept_rows, records))[:args.overlays]:
            write_overlay(row, rec, all_masks[rec["id"]], out)
        print(f"\n{min(args.overlays, len(records))} overlays -> {out}/  "
              "LOOK AT THESE before recording any number")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"split": args.split, "n": len(records),
             "rejected": {k: v for k, v in stats.items() if k != "ACCEPTED"},
             "selection": {"min_area_frac": MIN_AREA_FRAC,
                           "min_siblings": MIN_SIBLINGS,
                           "min_target_iou": MIN_TARGET_IOU},
             "cases": records}, indent=1))
        print(f"\nwrote {args.out}")

    if args.build:
        try:
            from bench import instance_bench as ib  # noqa: PLC0415
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import instance_bench as ib  # noqa: PLC0415
        out = ib.build(records, lambda r: all_masks[r["id"]],
                       args.out or f"cases_{args.split}.json",
                       source_dataset=f"magicbrush-{args.split}xcoco")
        print(f"\ninstance_bench.build(): {out['n_cases']} cases")
        print(f"  its own rejections: {out['rejected']}")

    return {"records": records, "masks": all_masks, "stats": stats}


def self_test():
    """Geometry only. No parquet, no COCO, no network."""
    # PIL's polygon fill is endpoint-INCLUSIVE: a 10..60 box is 51x51 = 2601,
    # not 50x50. That is a small systematic over-coverage versus pycocotools,
    # which is why pycocotools is preferred in ann_to_mask when importable.
    sq = [[10, 10, 10, 60, 60, 60, 60, 10]]
    m = poly_to_mask(sq, 100, 100)
    assert m.sum() == 2601, m.sum()
    assert iou(m, m) == 1.0
    assert iou(m, np.zeros_like(m)) == 0.0

    half = poly_to_mask([[10, 10, 10, 60, 35, 60, 35, 10]], 100, 100)
    assert 0.45 < iou(m, half) < 0.55, iou(m, half)

    r = resize_mask(m, (50, 50))
    assert r.shape == (50, 50)
    assert 550 <= r.sum() <= 700, r.sum()

    buf = io.BytesIO()
    a = np.full((20, 20, 4), 255, np.uint8)
    a[5:15, 5:15, 3] = 0                       # transparent == edit region
    Image.fromarray(a, "RGBA").save(buf, format="PNG")
    e = decode_edit_mask(buf.getvalue())
    assert e.sum() == 100, e.sum()
    assert e[10, 10] and not e[0, 0]

    assert names_category("make the left bowl red", "bowl")
    assert names_category("change the TVs to blue", "tv")
    assert names_category("recolour the man's shirt", "person")
    assert not names_category("make the bowling ball red", "bowl")
    assert not names_category("paint the wall green", "bowl")
    assert "people" in surface_forms("person")
    assert "sandwiches" in surface_forms("sandwich")
    assert norm_img_id("140513") == 140513 and norm_img_id("COCO_000140513") == 140513

    # end-to-end on synthetic data: three bowls, the edit lands on the second
    def box(x0, y0, x1, y1):
        return [[x0, y0, x0, y1, x1, y1, x1, y0]]

    anns = [{"id": 1, "category_id": 51, "area": 2500, "segmentation": box(10, 10, 60, 60)},
            {"id": 2, "category_id": 51, "area": 2500, "segmentation": box(110, 10, 160, 60)},
            {"id": 3, "category_id": 51, "area": 2500, "segmentation": box(10, 110, 60, 160)},
            {"id": 9, "category_id": 62, "area": 2500, "segmentation": box(110, 110, 160, 160)}]
    coco = {140513: {"h": 200, "w": 200, "anns": anns}}
    cats = {51: "bowl", 62: "chair"}

    alpha = np.full((200, 200, 4), 255, np.uint8)
    alpha[10:61, 110:161, 3] = 0                    # transparent over bowl id=2
    buf = io.BytesIO()
    Image.fromarray(alpha, "RGBA").save(buf, format="PNG")
    row = {"img_id": "140513", "instruction": "make the right bowl blue",
           "mask_img": {"bytes": buf.getvalue()}, "source_img": None}

    st = collections.Counter()
    rec, masks = resolve_case(row, coco, cats, st)
    assert rec is not None, dict(st)
    assert rec["target_id"] == "2", rec["target_id"]
    assert rec["category"] == "bowl" and rec["n_siblings"] == 2, rec
    assert rec["target_iou"] > 0.95, rec["target_iou"]
    assert {"1", "2", "3", "9"} <= set(masks), sorted(masks)
    assert st["aspect_mismatch"] == 0

    # an instruction naming a category with no siblings is rejected at filter 2
    st2 = collections.Counter()
    row2 = dict(row, instruction="make the chair blue")
    assert resolve_case(row2, coco, cats, st2)[0] is None
    assert st2["instruction names no multi-instance category"] == 1, dict(st2)

    # an edit landing on no annotated instance is rejected at filter 3
    a3 = np.full((200, 200, 4), 255, np.uint8)
    a3[180:200, 180:200, 3] = 0
    b3 = io.BytesIO()
    Image.fromarray(a3, "RGBA").save(b3, format="PNG")
    st3 = collections.Counter()
    row3 = dict(row, mask_img={"bytes": b3.getvalue()})
    assert resolve_case(row3, coco, cats, st3)[0] is None
    assert st3["edit mask matches no annotated instance"] == 1, dict(st3)

    # geometry: the three modes agree at the centre and diverge at the edges.
    # That is the whole reason a wrong choice reads as intermittent drift.
    ch, cw = 480, 640
    centre = poly_to_mask([[310, 230, 310, 250, 330, 250, 330, 230]], ch, cw)
    edge = poly_to_mask([[600, 230, 600, 250, 630, 250, 630, 230]], ch, cw)
    mapped_c = {g: map_to_edit_frame(centre, (ch, cw), (1024, 1024), g) for g in GEOMETRIES}
    mapped_e = {g: map_to_edit_frame(edge, (ch, cw), (1024, 1024), g) for g in GEOMETRIES}
    assert iou(mapped_c["stretch"], mapped_c["center_crop"]) > 0.4, "centre should agree"
    assert iou(mapped_e["stretch"], mapped_e["center_crop"]) == 0.0, "edge must diverge"
    assert mapped_e["center_crop"].sum() == 0, "right edge falls outside a centred crop"
    assert mapped_e["pad"].sum() > 0

    # end-to-end on a NON-square COCO frame: two bowls, edit on the right one,
    # MagicBrush frame square. Only center_crop should resolve it.
    ch, cw, s = 480, 640, 480
    anns_ns = [{"id": 11, "category_id": 51, "area": 10000,
                "segmentation": [[200, 180, 200, 300, 320, 300, 320, 180]]},
               {"id": 12, "category_id": 51, "area": 10000,
                "segmentation": [[360, 180, 360, 300, 480, 300, 480, 180]]}]
    coco_ns = {700001: {"h": ch, "w": cw, "anns": anns_ns}}
    x0 = (cw - s) // 2
    a_ns = np.full((1024, 1024, 4), 255, np.uint8)
    k = 1024 / s
    a_ns[int(180 * k):int(300 * k), int((360 - x0) * k):int((480 - x0) * k), 3] = 0
    b_ns = io.BytesIO()
    Image.fromarray(a_ns, "RGBA").save(b_ns, format="PNG")
    row_ns = {"img_id": "700001", "instruction": "make the right bowl blue",
              "mask_img": {"bytes": b_ns.getvalue()}, "source_img": None}

    st_cc = collections.Counter()
    rec_cc, _ = resolve_case(row_ns, coco_ns, cats, st_cc, geometry="center_crop")
    assert rec_cc is not None and rec_cc["target_id"] == "12", (rec_cc, dict(st_cc))
    assert rec_cc["target_iou"] > 0.9, rec_cc["target_iou"]

    st_st = collections.Counter()
    rec_st, _ = resolve_case(row_ns, coco_ns, cats, st_st, geometry="stretch")
    assert rec_st is None or rec_st["target_iou"] < rec_cc["target_iou"], (
        "stretch must not beat center_crop on centre-cropped data")

    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magicbrush", default="/data/datasets/MagicBrush")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="write the case JSON here")
    p.add_argument("--build", action="store_true",
                   help="also hand records to instance_bench.build()")
    p.add_argument("--overlays", type=int, default=0,
                   help="write N target/sibling overlay PNGs")
    p.add_argument("--overlay-dir", default="runs/gate_c/overlays")
    p.add_argument("--geometry", default="auto",
                   choices=("auto",) + GEOMETRIES,
                   help="how the COCO frame maps into the MagicBrush frame; "
                        "'auto' probes and picks the winner")
    p.add_argument("--frames", default=None,
                   help="per-image crop rectangles from frame_fit.py")
    p.add_argument("--min-ncc", type=float, default=0.95,
                   help="drop cases whose frame did not register this well")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--min-margin", type=float, default=MIN_MARGIN)
    p.add_argument("--probe-coverage", action="store_true",
                   help="report the coverage/margin distribution and exit")
    p.add_argument("--probe-geometry", action="store_true",
                   help="probe the frame mapping and exit without building")
    p.add_argument("--probe-limit", type=int, default=120)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
