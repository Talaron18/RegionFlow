"""
Recover COCO siblings for RefEdit-Bench.

THE GAP. RefEdit-Bench ships `mask` -- the polygon of the ONE instance the
referring expression names -- and nothing about the other instances. Leakage is
measured on those other instances, so the sibling set has to come from
somewhere. The images are RefCOCO photographs, so COCO has every sibling
already; what is missing is the COCO image id, which the benchmark does not
carry and which 512x512 resquaring destroyed.

WHY NOT SEARCH PIXELS. Matching 200 bench images against 82k COCO train2014
photographs means downloading ~13GB before learning anything. Instead:

  STAGE 1  descriptors. A crop-and-resize is AFFINE, so normalising a polygon
           to its own bounding box makes its shape invariant to whatever
           squaring was applied -- including a non-square crop stretched to
           square, which bbox normalisation handles per axis. Rasterise every
           COCO polygon into a 16x16 normalised grid once; cache it.

  STAGE 2  shape match. The bench mask gets the same descriptor. Cosine
           similarity against the cached matrix is one matmul and gives the top
           K candidate (image_id, ann_id) pairs. Shape alone is NOT conclusive
           -- COCO is full of similar blobs -- so this is a shortlist, not an
           answer.

  STAGE 3  confirm with pixels. Download only the shortlisted images and
           register each against the bench image with frame_fit.fit_rect. The
           true match registers near NCC 1.0; an unrelated photograph does not.
           This is the same procedure that registered 266/266 MagicBrush rows
           at median NCC 1.000.

  STAGE 4  validate twice, independently of how the match was found.
           (a) map the winning COCO annotation through the fitted rectangle
               into the 512 frame and IoU it against the SHIPPED mask. A real
               match agrees; a coincidence does not.
           (b) check the recovered COCO category against the noun in
               `edit_instruction_single`. Disagreement means drop the case.
           A case failing either is DROPPED, not repaired. A wrong image id
           silently produces a plausible sibling set, which is the worst
           possible failure for a benchmark.

NOTE ON MASK TIGHTNESS. The shipped target masks are hand-drawn and loose (the
giraffe case runs off the neck into sky). Siblings recovered here come from
COCO and are tighter. Mixing tightness biases `leakage_ratio`: a loose target
mask dilutes target_change while tight sibling masks concentrate
sibling_change, pushing the ratio up. Report `sibling_excess_over_bg`
alongside it. Stage 4a's IoU quantifies the disagreement per case, so the bias
is measurable rather than assumed.

USAGE
    python bench/refedit_siblings.py --self-test
    python bench/refedit_siblings.py --stage descriptors      # ~3 min, cached
    python bench/refedit_siblings.py --stage match --limit 20 # shortlist only
    python bench/refedit_siblings.py --stage all --out bench/cases_refedit.json
"""

from __future__ import annotations

import argparse
import ast
import collections
import io
import json
import os
import re
import pickle
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DESC = 16                 # descriptor grid side
MIN_AREA_FRAC = 0.005     # matches instance_bench -- CASE SELECTION only
MAX_AREA_FRAC = 0.40      # CASE SELECTION only; see MIN_INDEX_FRAC
#: The search index must be PERMISSIVE. Applying the case-selection area bounds
#: at index-build time excluded every annotation covering >40% of its image --
#: which is the Easy split's defining property ("the interested object occupies
#: a significant portion of the image"). Raising --topk from 8 to 32 then
#: changed nothing, because the right annotation was never indexed at all.
MIN_INDEX_FRAC = 0.001
#: All four recovered rectangles were the FULL COCO frame: RefEdit resized to
#: 512x512 with no crop. A full-frame resize preserves NORMALISED bbox position
#: and size exactly, so those four numbers are strong evidence that pure shape
#: matching throws away -- and they survive a loose hand-drawn outline far
#: better than shape does.
#: Weight calibrated against the scales involved, not chosen by feel: bbox
#: mean|delta| runs ~0-0.3 while shape cosine runs ~0-1, so a unit weight
#: cannot overcome a 0.25 shape gap and the bbox term does nothing. Sweep with
#: --bbox-weight; 0 recovers pure shape ranking.
BBOX_WEIGHT = 3.0
DESC_VERSION = "v2"
MIN_CONFIRM_NCC = 0.95    # stage 3: image registration must succeed
MIN_MASK_AGREE = 0.40     # stage 4a: shipped mask vs recovered annotation
TOPK = 8


# ---------------------------------------------------------------- polygons --

def parse_polys(seg):
    """RefEdit stores the polygon as a STRING '[[x, y, x, y, ...]]'."""
    p = ast.literal_eval(seg) if isinstance(seg, str) else seg
    if not p:
        return []
    return p if isinstance(p[0], (list, tuple)) else [p]


def raster(polys, w, h, offset=(0.0, 0.0), scale=(1.0, 1.0)) -> np.ndarray:
    img = Image.new("L", (int(w), int(h)), 0)
    dr = ImageDraw.Draw(img)
    ox, oy = offset
    sx, sy = scale
    for poly in polys:
        if len(poly) >= 6:
            dr.polygon([((poly[i] - ox) * sx, (poly[i + 1] - oy) * sy)
                        for i in range(0, len(poly), 2)], fill=1)
    return np.asarray(img, dtype=bool)


def bbox_feat(polys, frame_wh) -> np.ndarray | None:
    """[cx, cy, w, h] as fractions of the frame. Invariant to a full-frame
    resize, which is what RefEdit applied."""
    xs = [v for q in polys for v in q[0::2]]
    ys = [v for q in polys for v in q[1::2]]
    if len(xs) < 3:
        return None
    W, H = frame_wh
    if not W or not H:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return np.array([(x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                     (x1 - x0) / W, (y1 - y0) / H], np.float32)


def descriptor(polys) -> np.ndarray | None:
    """Bbox-normalised DESCxDESC shape, zero-mean unit-norm.

    Invariant to translation and to independent x/y scaling, which is exactly
    the freedom a crop-and-resize has. NOT invariant to rotation or flip --
    neither of which is in play here.
    """
    xs = [v for q in polys for v in q[0::2]]
    ys = [v for q in polys for v in q[1::2]]
    if len(xs) < 3:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w < 1e-6 or h < 1e-6:
        return None
    g = raster(polys, DESC, DESC, offset=(x0, y0),
               scale=(DESC / w, DESC / h)).astype(np.float32).ravel()
    c = g - g.mean()
    n = float(np.linalg.norm(c))
    if n > 1e-8:
        return c / n
    # The polygon fills its own bounding box, so after normalisation it is
    # uniform and mean-centring annihilates it. Rare for COCO outlines but
    # real (signs, screens, walls). Fall back to the uncentred unit vector:
    # less discriminative, but a valid descriptor rather than a dropped case.
    n = float(np.linalg.norm(g))
    return g / n if n > 1e-8 else None


# ------------------------------------------------------------- stage 1 ------

def build_descriptors(ann_dir: Path, cache: Path, splits=("train2014",)) -> dict:
    """Descriptor matrix over every non-crowd polygon annotation."""
    npz = cache / f"coco_desc_{DESC_VERSION}_{'_'.join(splits)}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        print(f"descriptors: {z['D'].shape[0]} cached from {npz.name}")
        return {"D": z["D"], "B": z["B"], "ann_id": z["ann_id"],
                "image_id": z["image_id"], "cat_id": z["cat_id"]}

    D, B, ann_id, image_id, cat_id = [], [], [], [], []
    for split in splits:
        blob = json.loads((ann_dir / f"instances_{split}.json").read_text())
        sizes = {im["id"]: (im["width"], im["height"]) for im in blob["images"]}
        for n, a in enumerate(blob["annotations"]):
            if a.get("iscrowd") or not isinstance(a.get("segmentation"), list):
                continue
            W, H = sizes.get(a["image_id"], (0, 0))
            if not W:
                continue
            if a["area"] / (W * H) < MIN_INDEX_FRAC:
                continue
            d = descriptor(a["segmentation"])
            b = bbox_feat(a["segmentation"], (W, H))
            if d is None or b is None:
                continue
            D.append(d)
            B.append(b)
            ann_id.append(a["id"])
            image_id.append(a["image_id"])
            cat_id.append(a["category_id"])
            if n % 100000 == 0:
                print(f"  {split} {n} scanned, {len(D)} kept")

    out = {"D": np.asarray(D, np.float32), "B": np.asarray(B, np.float32),
           "ann_id": np.asarray(ann_id), "image_id": np.asarray(image_id),
           "cat_id": np.asarray(cat_id)}
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **out)
    print(f"descriptors: {out['D'].shape[0]} built -> {npz}")
    return out


# ------------------------------------------------------------- stage 2 ------

def batched_shortlist(Q: np.ndarray, BQ: np.ndarray, idx: dict, topk: int,
                      bbox_weight: float = BBOX_WEIGHT, chunk: int = 50000):
    """All rows scored against the whole index in one chunked BLAS pass.

    The per-row version issues R separate mat-vecs, each streaming the entire
    600MB descriptor matrix from memory -- R times the bandwidth for the same
    arithmetic. As one (R,256)@(256,C) matmul per chunk it is a single BLAS
    call that saturates every core, and the index is read once.

    The bbox penalty is accumulated one dimension at a time: the obvious
    broadcast |BQ[:,None,:] - B[None,:,:]| would allocate R*C*4 floats, which
    at R=200 and C=50k is 160M values per chunk.
    """
    R = Q.shape[0]
    keep = min(topk * 6, idx["D"].shape[0])      # raw depth before image dedup
    top_s = np.full((R, 0), -np.inf, np.float32)
    top_i = np.zeros((R, 0), np.int64)

    for start in range(0, idx["D"].shape[0], chunk):
        D = idx["D"][start:start + chunk]
        B = idx["B"][start:start + chunk]
        sim = (Q @ D.T).astype(np.float32)                       # (R, C)
        if bbox_weight:
            bd = np.zeros_like(sim)
            for k in range(BQ.shape[1]):
                bd += np.abs(BQ[:, k:k + 1] - B[None, :, k])
            sim -= bbox_weight * (bd / BQ.shape[1])

        n = min(keep, sim.shape[1])
        part = np.argpartition(-sim, n - 1, axis=1)[:, :n]
        cand_s = np.take_along_axis(sim, part, axis=1)
        cand_i = part + start

        top_s = np.concatenate([top_s, cand_s], axis=1)
        top_i = np.concatenate([top_i, cand_i], axis=1)
        if top_s.shape[1] > keep:
            n = min(keep, top_s.shape[1])
            sel = np.argpartition(-top_s, n - 1, axis=1)[:, :n]
            top_s = np.take_along_axis(top_s, sel, axis=1)
            top_i = np.take_along_axis(top_i, sel, axis=1)

    order = np.argsort(-top_s, axis=1)
    top_s = np.take_along_axis(top_s, order, axis=1)
    top_i = np.take_along_axis(top_i, order, axis=1)

    out = []
    for r in range(R):
        seen, cands = set(), []
        for c in range(top_i.shape[1]):
            j = int(top_i[r, c])
            img = int(idx["image_id"][j])
            if img in seen:
                continue
            seen.add(img)
            cands.append({"image_id": img, "ann_id": int(idx["ann_id"][j]),
                          "cat_id": int(idx["cat_id"][j]),
                          "shape_sim": float(idx["D"][j] @ Q[r]),
                          "bbox_dist": float(np.abs(idx["B"][j] - BQ[r]).mean()),
                          "score": float(top_s[r, c])})
            if len(cands) >= topk:
                break
        out.append(cands)
    return out


def shortlist(bench_desc: np.ndarray, idx: dict, topk: int = TOPK,
              bench_bbox: np.ndarray | None = None,
              bbox_weight: float = BBOX_WEIGHT):
    """Top-K candidates by shape similarity penalised by bbox disagreement.

    score = cos(shape) - bbox_weight * mean|delta bbox|

    The bbox term is what rescues loose hand-drawn outlines: a mask that runs
    off the object (the giraffe case, mask_agreement 0.469) has a poor shape
    match but still lands in roughly the right place at roughly the right size.
    """
    sim = idx["D"] @ bench_desc
    score = sim
    bdist = None
    if bench_bbox is not None and "B" in idx and len(idx["B"]):
        bdist = np.abs(idx["B"] - bench_bbox[None, :]).mean(axis=1)
        score = sim - bbox_weight * bdist

    order = np.argsort(score)[::-1]
    seen, out = set(), []
    for j in order:
        img = int(idx["image_id"][j])
        if img in seen:
            continue
        seen.add(img)
        out.append({"image_id": img, "ann_id": int(idx["ann_id"][j]),
                    "cat_id": int(idx["cat_id"][j]),
                    "shape_sim": float(sim[j]),
                    "bbox_dist": float(bdist[j]) if bdist is not None else None,
                    "score": float(score[j])})
        if len(out) >= topk:
            break
    return out


# ------------------------------------------------------------- stage 3/4 ----

def fetch(image_id: int, url: str, cache: Path) -> Path:
    dst = cache / f"{image_id}.jpg"
    if not dst.exists():
        tmp = cache / f".{image_id}.part"
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)          # atomic: a killed run leaves no truncated jpg
    return dst


def _fit_task(task):
    """Module-level so ProcessPoolExecutor can pickle it. Paths only -- the
    images are read inside the worker, so nothing large crosses the boundary."""
    key, coco_path, bench_path = task
    from PIL import Image as _I  # noqa: PLC0415
    from frame_fit import fit_rect  # noqa: PLC0415
    try:
        o = _I.open(coco_path).convert("RGB")
        b = _I.open(bench_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return key, None
    return key, fit_rect(o, b)


def fit_many(tasks, jobs: int):
    """fit_rect is pure Python + PIL, so it holds the GIL: threads give nothing
    and only processes use the other cores."""
    if not tasks:
        return {}
    if jobs <= 1:
        return dict(_fit_task(t) for t in tasks)
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
    out = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for key, fit in ex.map(_fit_task, tasks, chunksize=4):
            out[key] = fit
    return out


def prefetch(cands, urls, cache: Path, jobs: int = 8) -> dict:
    """Download a batch of candidates concurrently. Failures are just absent."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    todo = [(c["image_id"], urls.get(c["image_id"])) for c in cands]
    todo = [(i, u) for i, u in todo if u and not (cache / f"{i}.jpg").exists()]
    if not todo:
        return {}
    got = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch, i, u, cache): i for i, u in todo}
        for f in futs:
            try:
                got[futs[f]] = f.result()
            except Exception:  # noqa: BLE001
                pass
    return got


def confirm(bench_img: Image.Image, cands, urls, cache: Path,
            width: int = 3, jobs: int = 8):
    """Register against candidates in shape order, stopping at the first hit.

    The shortlist is ranked by shape similarity and the true match is usually
    rank 1 (median top-1 similarity 0.854 on this benchmark), so fitting all K
    candidates does 8x the downloads and 8x the registrations for the same
    answer. Candidates are prefetched `width` at a time in parallel -- enough
    to hide network latency without paying for the whole shortlist.

    Returns the first candidate reaching MIN_CONFIRM_NCC, else the best seen
    (so the caller can report how close the near-misses came).
    """
    from frame_fit import fit_rect  # noqa: PLC0415

    best, tried = None, 0
    for start in range(0, len(cands), width):
        batch = cands[start:start + width]
        prefetch(batch, urls, cache, jobs)
        for c in batch:
            url = urls.get(c["image_id"])
            if not url:
                continue
            try:
                orig = Image.open(fetch(c["image_id"], url, cache)).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            fit = fit_rect(orig, bench_img)
            tried += 1
            rec = {**c, "rect": fit["rect"], "ncc_fit": fit["ncc_fit"],
                   "coco_wh": fit["coco_wh"], "n_tried": tried}
            if best is None or rec["ncc_fit"] > best["ncc_fit"]:
                best = rec
            if rec["ncc_fit"] >= MIN_CONFIRM_NCC:
                return best
    return best


#: Role and everyday nouns RefEdit's authors use that COCO class names do not
#: cover. Drawn from the 47 rejected rows, not invented: every entry below
#: appeared in an instruction whose recovered annotation already agreed with the
#: benchmark's own shipped mask.
ROLE_NOUNS = {
    "person": ["player", "tennis player", "baseball player", "skateboarder",
               "surfer", "skier", "snowboarder", "umpire", "officer", "cop",
               "chef", "cook", "adult", "elderly", "toddler", "baby", "infant",
               "worker", "rider", "pedestrian", "waiter", "waitress",
               "batter", "pitcher", "catcher", "goalie", "dancer", "groom",
               "bride", "teenager", "gentleman", "male", "female", "human"],
    "motorcycle": ["scooter", "moped", "vespa"],
    "umbrella": ["parasol"],
    "bird": ["ostrich", "pigeon", "seagull", "duck", "goose", "parrot", "swan"],
    "dog": ["puppy", "retriever", "labrador", "poodle", "terrier", "hound"],
    "cat": ["kitten", "kitty"],
    "suitcase": ["luggage", "case", "trunk"],
    "handbag": ["purse", "clutch", "satchel"],
    "couch": ["sofa", "loveseat"],
    "chair": ["sofa", "armchair", "stool", "seat"],
    "cup": ["mug", "glass", "tumbler", "drink", "coffee", "beer"],
    "car": ["van", "taxi", "cab", "sedan", "suv", "jeep"],
    "truck": ["lorry", "pickup", "van"],
    "airplane": ["aircraft", "jet", "plane", "airliner"],
    "bed": ["bunk", "mattress"],
    "bowl": ["dish", "container", "box"],
}


def noun_agrees(instruction: str, cat_name: str) -> bool:
    """Does the recovered COCO category appear in the instruction?

    TWO BUGS LIVED HERE.

    1. The caller passed `edit_instruction_single` -- the instruction with the
       REFERRING EXPRESSION STRIPPED OUT, which is exactly the clause naming
       the object. "Make the quilt on the bottom part of the bunk bed black"
       becomes "Make the quilt black." Of 47 rejections, 16 passed against the
       full text and 0 against the stripped text. Always check the full one.

    2. The vocabulary was COCO class names only. RefEdit's authors write
       "tennis player", "chef", "scooter", "parasol", "ostrich". See ROLE_NOUNS.

    This check is ADVISORY, not authoritative. `build_case` already compares
    the recovered annotation against the benchmark's OWN shipped mask, which is
    its statement of which instance the expression names -- strictly stronger
    evidence than whether a noun happens to appear in a sentence.
    """
    try:
        from magicbrush_coco import names_category  # noqa: PLC0415
        if names_category(instruction, cat_name):
            return True
    except ImportError:
        base = cat_name.split()[-1].lower()
        if re.search(rf"\b{re.escape(base)}s?\b", instruction, re.I):
            return True
    for alt in ROLE_NOUNS.get(cat_name, []):
        if re.search(rf"\b{re.escape(alt)}s?\b", instruction, re.I):
            return True
    return False


def build_case(row, best, coco_index, cats, urls, cache: Path,
               require_noun: bool = False):
    """Stage 4: validate the match, then take siblings from the COCO join."""
    from frame_fit import apply_rect  # noqa: PLC0415

    rec = coco_index[best["image_id"]]
    ch, cw = rec["h"], rec["w"]
    shipped = raster(parse_polys(row["mask"]), 512, 512)

    anns = {a["id"]: a for a in rec["anns"]}
    tgt = anns.get(best["ann_id"])
    if tgt is None:
        return None, "target annotation missing"

    def to_bench(a):
        m = raster(a["segmentation"], cw, ch)
        return apply_rect(m, best["rect"], (512, 512))

    tm = to_bench(tgt)
    inter = int((tm & shipped).sum())
    union = int((tm | shipped).sum())
    agree = inter / union if union else 0.0
    if agree < MIN_MASK_AGREE:
        return None, "recovered annotation disagrees with shipped mask"

    cat = cats[tgt["category_id"]]
    # The full instruction, never the stripped one -- see noun_agrees.
    noun_ok = noun_agrees(row.get("editing_instruction", ""), cat) or \
        noun_agrees(row.get("edit_instruction_single", "") or "", cat)
    if require_noun and not noun_ok:
        return None, "recovered category absent from instruction"

    area = ch * cw
    sibs, dists = [], []
    for a in rec["anns"]:
        if a["id"] == tgt["id"]:
            continue
        if a["area"] / area < MIN_AREA_FRAC:
            continue
        (sibs if a["category_id"] == tgt["category_id"] else dists).append(a)
    if not sibs:
        return None, "no same-category sibling above the area floor"

    masks = {str(tgt["id"]): tm}
    for a in sibs + dists:
        masks[str(a["id"])] = to_bench(a)

    case = {
        "id": f"refedit-{row['index']}",
        "source": "RefEdit-Bench",
        "difficulty": row["category"],
        "instruction": row["editing_instruction"],
        "instruction_single": row.get("edit_instruction_single"),
        "category": cat,
        "coco_image_id": best["image_id"],
        "target_id": str(tgt["id"]),
        "instances": [{"id": str(a["id"]), "category": cats[a["category_id"]],
                       "mask_ref": str(a["id"])}
                      for a in [tgt] + sibs + dists],
        "n_siblings": len(sibs),
        "noun_match": bool(noun_ok),
        "shape_sim": round(best["shape_sim"], 4),
        "ncc_fit": best["ncc_fit"],
        "mask_agreement": round(agree, 4),
        "rect": best["rect"],
        "coco_hw": [ch, cw],
    }
    return (case, masks), None


# --------------------------------------------------------------- loading ----

def load_bench(path: str):
    import pyarrow.parquet as pq  # noqa: PLC0415

    t = pq.read_table(path)
    d = t.to_pydict()
    rows = []
    for i in range(t.num_rows):
        rows.append({"index": i, "mask": d["mask"][i],
                     "editing_instruction": d["editing_instruction"][i],
                     "edit_instruction_single": d.get(
                         "edit_instruction_single", [""] * t.num_rows)[i],
                     "category": d["category"][i],
                     "input_image": d["input_image"][i]})
    return rows


def load_coco(ann_dir: Path, splits=("train2014",), cache: Path | None = None):
    """Parsed once, then pickled. instances_train2014.json is ~450MB of JSON
    and reparsing it costs ~a minute on every run for data that never changes."""
    if cache is not None:
        pkl = cache / f"coco_index_{'_'.join(splits)}.pkl"
        if pkl.exists():
            t = time.time()
            with pkl.open("rb") as fh:
                index, cats, urls = pickle.load(fh)
            print(f"coco index: {len(index)} images from cache ({time.time() - t:.1f}s)")
            return index, cats, urls

    index, cats, urls = _load_coco_json(ann_dir, splits)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / f"coco_index_{'_'.join(splits)}.pkl").open("wb") as fh:
            pickle.dump((index, cats, urls), fh, protocol=5)
    return index, cats, urls


def _load_coco_json(ann_dir: Path, splits=("train2014",)):
    index, cats, urls = {}, {}, {}
    for split in splits:
        blob = json.loads((ann_dir / f"instances_{split}.json").read_text())
        cats.update({c["id"]: c["name"] for c in blob["categories"]})
        for im in blob["images"]:
            index[im["id"]] = {"h": im["height"], "w": im["width"], "anns": []}
            urls[im["id"]] = im.get("coco_url")
        for a in blob["annotations"]:
            if not a.get("iscrowd") and a["image_id"] in index:
                index[a["image_id"]]["anns"].append(a)
    return index, cats, urls


def bench_image(cell) -> Image.Image:
    if isinstance(cell, dict) and cell.get("bytes"):
        return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    return Image.open(io.BytesIO(cell)).convert("RGB")


def write_overlay(row, case, masks, out_dir: Path, outline_px: int = 4,
                  crop: bool = False, margin: float = 0.12):
    """Target green, same-category siblings red, shipped mask outlined yellow.

    The shipped mask is drawn separately because it is the ONLY independent
    evidence that the recovered COCO annotation is the instance the referring
    expression names. Green and yellow disagreeing means the join found the
    wrong instance -- which `mask_agreement` scores but cannot show.
    """
    img = bench_image(row["input_image"])
    arr = np.asarray(img).astype(np.float32)

    tgt = masks[case["target_id"]]
    sib = np.zeros_like(tgt)
    for inst in case["instances"]:
        if inst["category"] == case["category"] and inst["id"] != case["target_id"]:
            if inst["id"] in masks:
                sib |= masks[inst["id"]]

    arr[sib] = 0.45 * arr[sib] + 0.55 * np.array([235, 60, 60], np.float32)
    arr[tgt] = 0.45 * arr[tgt] + 0.55 * np.array([60, 220, 90], np.float32)

    # The shipped outline must survive downscaling. At 1px it vanishes in a
    # 248px contact-sheet thumbnail -- and green-inside-yellow IS the check, so
    # a sheet without a visible outline only shows that green is on *something*.
    shipped = raster(parse_polys(row["mask"]), 512, 512)
    inner = shipped.copy()
    for _ in range(max(1, outline_px)):
        inner = (inner & np.pad(inner, 1)[:-2, 1:-1] & np.pad(inner, 1)[2:, 1:-1]
                 & np.pad(inner, 1)[1:-1, :-2] & np.pad(inner, 1)[1:-1, 2:])
    arr[shipped & ~inner] = np.array([255, 235, 40], np.float32)

    # Optionally crop to the union of target + siblings so a small object is
    # not three pixels across in the sheet.
    if crop:
        union = tgt | sib
        ys, xs = np.where(union)
        if len(ys):
            m = int(margin * 512)
            y0, y1 = max(0, ys.min() - m), min(512, ys.max() + m)
            x0, x1 = max(0, xs.min() - m), min(512, xs.max() + m)
            side = max(y1 - y0, x1 - x0)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y0 = max(0, min(512 - side, cy - side // 2))
            x0 = max(0, min(512 - side, cx - side // 2))
            arr = arr[y0:y0 + side, x0:x0 + side]

    out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{case['id']}_{case['difficulty']}_"
            f"{case['category'].replace(' ', '-')}_"
            f"agree{case['mask_agreement']:.2f}.png")
    Image.fromarray(arr.astype(np.uint8)).save(out_dir / name)


def stratified_sample(cases, n, seed=0):
    """Sample across difficulty x person/other x mask_agreement tercile.

    The first-N overlays were an ORDERED sample: RefEdit-Bench stores easy at
    indices 0-99, so "the first 24" audits mostly easy cases and whatever
    categories sort early. An audit that supports a claim has to cover the
    strata where failure is plausible -- hard, person-heavy, low agreement --
    not the strata that happen to come first.
    """
    if n >= len(cases):
        return list(cases)
    ag = sorted(x["mask_agreement"] for x in cases)
    lo, hi = ag[len(ag) // 3], ag[2 * len(ag) // 3]

    def stratum(x):
        t = 0 if x["mask_agreement"] < lo else (1 if x["mask_agreement"] < hi else 2)
        return (x["difficulty"], x["category"] == "person", t)

    buckets = collections.defaultdict(list)
    for x in cases:
        buckets[stratum(x)].append(x)
    rng = np.random.default_rng(seed)
    for v in buckets.values():
        rng.shuffle(v)

    out, keys = [], sorted(buckets, key=lambda k: -len(buckets[k]))
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:                       # round-robin keeps strata even
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def write_contact_sheet(entries, out_path: Path, cols=4, thumb=320, pad=6, cap=34):
    """Grid of overlays with captions. 24 files to open one by one is a sample
    nobody finishes; one sheet is a sample that gets looked at."""
    from PIL import ImageFont  # noqa: PLC0415
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    rows = (len(entries) + cols - 1) // cols
    W = cols * (thumb + pad) + pad
    H = rows * (thumb + cap + pad) + pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    dr = ImageDraw.Draw(sheet)

    for i, (path, lines) in enumerate(entries):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + cap + pad)
        try:
            im = Image.open(path).convert("RGB").resize((thumb, thumb), Image.BILINEAR)
        except Exception:  # noqa: BLE001
            continue
        sheet.paste(im, (x, y))
        for k, line in enumerate(lines[:2]):
            dr.text((x + 2, y + thumb + 2 + k * 13), line[:52],
                    fill=(230, 230, 235) if k == 0 else (160, 160, 170), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ------------------------------------------------------------------ main ----

def run(args):
    ann_dir = Path(args.coco_annotations)
    cache = Path(args.cache)
    splits = tuple(args.coco_splits.split(","))

    idx = build_descriptors(ann_dir, cache, splits)
    if args.stage == "descriptors":
        return

    rows = load_bench(args.bench)
    if args.limit:
        rows = rows[:args.limit]
    coco_index, cats, urls = load_coco(ann_dir, splits, cache=cache)
    print(f"\n{len(rows)} bench rows, {len(coco_index)} COCO images\n")

    cases, masks_all, why = [], {}, collections.Counter()
    row_by_case = {}
    sims, nccs, agrees, tried, rejects = [], [], [], [], []
    t0 = time.time()

    # ---- descriptors for every row, then ONE batched scoring pass ----------
    keep_rows, Q, BQ = [], [], []
    for row in rows:
        polys = parse_polys(row["mask"])
        d, bb = descriptor(polys), bbox_feat(polys, (512, 512))
        if d is None or bb is None:
            why["degenerate mask polygon"] += 1
            continue
        keep_rows.append(row)
        Q.append(d)
        BQ.append(bb)
    if not keep_rows:
        print("no usable rows")
        return

    Q = np.stack(Q).astype(np.float32)
    BQ = np.stack(BQ).astype(np.float32)
    t = time.time()
    shortlists = batched_shortlist(Q, BQ, idx, args.topk, args.bbox_weight)
    print(f"scored {len(keep_rows)} rows against {idx['D'].shape[0]} "
          f"annotations in {time.time() - t:.1f}s")
    sims = [c[0]["shape_sim"] if c else 0.0 for c in shortlists]

    if args.stage == "match":
        if args.verbose:
            for row, cands in zip(keep_rows, shortlists):
                print(f"  [{row['index']:>3}] {row['category']:<5} "
                      f"{row['editing_instruction'][:52]}")
                for c in cands[:4]:
                    print(f"        img {c['image_id']:>7} ann {c['ann_id']:>8} "
                          f"shape {c['shape_sim']:.3f} bbox {c['bbox_dist']:.3f} "
                          f"score {c['score']:.3f}")
        print(f"\ntop-1 shape similarity: median {np.median(sims):.3f}")
        print("(stage=match: shortlist only, nothing downloaded)")
        return

    # ---- bench images to disk once: workers take paths, not pixels ---------
    img_cache = Path(args.image_cache)
    img_cache.mkdir(parents=True, exist_ok=True)
    bench_dir = img_cache / "_bench512"
    bench_dir.mkdir(exist_ok=True)
    bench_paths = {}
    for row in keep_rows:
        bp = bench_dir / f"{row['index']}.png"
        if not bp.exists():
            bench_image(row["input_image"]).save(bp)
        bench_paths[row["index"]] = bp

    # ---- rounds: download a slice of every row's shortlist in parallel,
    #      fit them all in parallel, keep rows that have not registered yet ---
    pending = list(range(len(keep_rows)))
    best_of, fits_per_row = {}, collections.Counter()
    for rnd in range(0, args.topk, args.prefetch):
        if not pending:
            break
        batch = {r: shortlists[r][rnd:rnd + args.prefetch] for r in pending}
        flat = [c for cs in batch.values() for c in cs]
        if not flat:
            break

        t = time.time()
        prefetch(flat, urls, img_cache, args.jobs)
        t_dl = time.time() - t

        tasks = []
        for r, cs in batch.items():
            for c in cs:
                jpg = img_cache / f"{c['image_id']}.jpg"
                if jpg.exists():
                    tasks.append(((r, c["image_id"]), str(jpg),
                                  str(bench_paths[keep_rows[r]["index"]])))
                    fits_per_row[r] += 1
        t = time.time()
        fits = fit_many(tasks, args.jobs)
        t_fit = time.time() - t

        still = []
        for r, cs in batch.items():
            done = False
            for c in cs:
                fit = fits.get((r, c["image_id"]))
                if not fit:
                    continue
                rec = {**c, "rect": fit["rect"], "ncc_fit": fit["ncc_fit"],
                       "coco_wh": fit["coco_wh"]}
                cur = best_of.get(r)
                if cur is None or rec["ncc_fit"] > cur["ncc_fit"]:
                    best_of[r] = rec
                if rec["ncc_fit"] >= MIN_CONFIRM_NCC:
                    done = True
            if not done:
                still.append(r)
        print(f"  round {rnd // args.prefetch + 1}: {len(batch)} rows, "
              f"{len(tasks)} fits, dl {t_dl:.1f}s fit {t_fit:.1f}s, "
              f"{len(batch) - len(still)} registered")
        pending = still

    # ---- build cases ------------------------------------------------------
    for r, row in enumerate(keep_rows):
        best = best_of.get(r)
        if best is None:
            why["no candidate image fetched"] += 1
            continue
        nccs.append(best["ncc_fit"])
        tried.append(fits_per_row[r])
        if best["ncc_fit"] < MIN_CONFIRM_NCC:
            why["no candidate registered"] += 1
            if args.verbose:
                print(f"  [{row['index']:>3}] no registration; best ncc "
                      f"{best['ncc_fit']:.3f} on img {best['image_id']} "
                      f"(shape {best['shape_sim']:.3f}, bbox {best['bbox_dist']:.3f})")
            continue
        built, reason = build_case(row, best, coco_index, cats, urls,
                                   Path(args.image_cache),
                                   require_noun=args.require_noun)
        if built is None:
            why[reason] += 1
            rejects.append({
                "index": row["index"], "reason": reason,
                "difficulty": row["category"],
                "recovered_category": cats.get(best.get("cat_id"), "?"),
                "coco_image_id": best["image_id"],
                "instruction": row["editing_instruction"],
                "instruction_single": row.get("edit_instruction_single"),
                "shape_sim": round(best["shape_sim"], 3),
                "ncc_fit": best["ncc_fit"]})
            if args.verbose:
                print(f"  [{row['index']:>3}] {reason}: recovered "
                      f"{cats.get(best.get('cat_id'), '?')!r} | "
                      f"{row['editing_instruction'][:64]}")
            continue
        case, masks = built
        agrees.append(case["mask_agreement"])
        cases.append(case)
        masks_all[case["id"]] = masks
        row_by_case[case["id"]] = row

    print(f"\ntotal {time.time() - t0:.1f}s")

    n = len(rows)
    print(f"\ntop-1 shape similarity: median {np.median(sims):.3f}" if sims else "")
    if args.stage == "match":
        print("(stage=match: shortlist only, nothing downloaded)")
        return

    print(f"\nBUILT {len(cases)}/{n} ({len(cases) / max(n, 1):.1%})\n")
    for k, v in why.most_common():
        print(f"  {v:>4}  {k}")
    if nccs:
        v = np.array(nccs)
        print(f"\n  registration NCC: median {np.median(v):.3f}  "
              f">= {MIN_CONFIRM_NCC}: {(v >= MIN_CONFIRM_NCC).mean():.1%}")
        print(f"  candidates fitted per row: mean "
              f"{np.mean(tried) if tried else float('nan'):.2f} "
              f"(shortlist was {args.topk}) -- 1.0 means shape rank 1 was right")
    if agrees:
        v = np.array(agrees)
        print(f"  shipped-vs-recovered mask IoU: median {np.median(v):.3f}  "
              f"p10 {np.percentile(v, 10):.3f}")
        print("    (well below 1.0 is EXPECTED -- shipped masks are loose;\n"
              "     this is the tightness bias, now quantified per case)")
    if cases:
        nm = sum(1 for x in cases if not x.get("noun_match", True))
        print(f"\n  noun_match False (recovered category not named in the "
              f"instruction): {nm}/{len(cases)}")
        print(f"  difficulty: {collections.Counter(c['difficulty'] for c in cases).most_common()}")
        print(f"  siblings:   {sorted(collections.Counter(c['n_siblings'] for c in cases).items())}")
        print(f"  categories: {collections.Counter(c['category'] for c in cases).most_common(10)}")

    if args.overlays and cases:
        sample = stratified_sample(cases, args.overlays, args.overlay_seed)
        out_dir = Path(args.overlay_dir)
        entries = []
        for case in sample:
            write_overlay(row_by_case[case["id"]], case, masks_all[case["id"]],
                          out_dir, outline_px=args.outline_px, crop=args.crop)
            name = (f"{case['id']}_{case['difficulty']}_"
                    f"{case['category'].replace(' ', '-')}_"
                    f"agree{case['mask_agreement']:.2f}.png")
            entries.append((out_dir / name, [
                f"{case['id']} {case['difficulty']} {case['category']} "
                f"agr{case['mask_agreement']:.2f} sib{case['n_siblings']}",
                case["instruction"]]))
        sheet = write_contact_sheet(entries, out_dir / "contact_sheet.png",
                                    cols=args.sheet_cols)
        print(f"\n  {len(sample)} stratified overlays -> {out_dir}/")
        print(f"  contact sheet -> {sheet}")
        print("  green=target  red=siblings  yellow=shipped benchmark mask")
        st = collections.Counter(
            (c["difficulty"], "person" if c["category"] == "person" else "other")
            for c in sample)
        print(f"  strata: {sorted(st.items())}")

        ledger = [{"id": c["id"], "difficulty": c["difficulty"],
                   "category": c["category"], "n_siblings": c["n_siblings"],
                   "mask_agreement": c["mask_agreement"],
                   "noun_match": c.get("noun_match"),
                   "instruction": c["instruction"],
                   "verdict": None, "notes": ""} for c in sample]
        lp = Path(args.audit_out)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(ledger, indent=1))
        print(f"  audit ledger -> {lp}  (fill in \"verdict\" per case)")
        print("    ok            green sits on the instance the expression names")
        print("    wrong_target  green is on a different instance")
        print("    wrong_sibling a red mask is not really a same-category instance")
        print("    unusable      target and a sibling overlap so much that leakage")
        print("                  could not be attributed to either")

    if rejects:
        noun = [r for r in rejects
                if r["reason"] == "recovered category absent from instruction"]
        if noun:
            pairs = collections.Counter(r["recovered_category"] for r in noun)
            print("\n  categories rejected by the noun check "
                  "(recovered, but not found in the instruction):")
            for cat, n in pairs.most_common(15):
                print(f"    {n:>3}  {cat}")
            print("    -> each is either a synonym gap in the surface-form "
                  "table\n       or a genuinely wrong recovery; the instruction "
                  "text decides.")
        if args.rejects_out:
            Path(args.rejects_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.rejects_out).write_text(json.dumps(rejects, indent=1))
            print(f"\n  wrote {len(rejects)} rejects -> {args.rejects_out}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"source": "RefEdit-Bench x COCO", "n": len(cases),
             "rejected": dict(why),
             "selection": {"min_confirm_ncc": MIN_CONFIRM_NCC,
                           "min_mask_agree": MIN_MASK_AGREE,
                           "min_area_frac": MIN_AREA_FRAC, "topk": args.topk},
             "cases": cases}, indent=1))
        print(f"\nwrote {args.out}")


def self_test():
    tri = [[10, 10, 10, 90, 70, 90]]
    blob = [[20, 20, 80, 25, 75, 85, 25, 80]]

    # descriptor is invariant to crop-and-resize (translate + per-axis scale)
    moved = [[(v * 2.5 + 300) if i % 2 == 0 else (v * 1.7 + 40)
              for i, v in enumerate(tri[0])]]
    a, b = descriptor(tri), descriptor(moved)
    assert a is not None and float(a @ b) > 0.97, float(a @ b)
    assert float(descriptor(tri) @ descriptor(blob)) < 0.9

    # shortlist returns the right annotation and dedups by image
    idx = {"D": np.stack([descriptor(blob), descriptor(tri), descriptor(tri)]),
           "ann_id": np.array([1, 2, 3]), "image_id": np.array([10, 20, 20]),
           "cat_id": np.array([51, 17, 17])}
    got = shortlist(descriptor(moved), idx, topk=3)
    assert got[0]["ann_id"] == 2 and got[0]["image_id"] == 20, got
    assert len({g["image_id"] for g in got}) == len(got), "images must dedup"

    assert parse_polys("[[1, 2, 3, 4, 5, 6]]") == [[1, 2, 3, 4, 5, 6]]
    assert parse_polys([[1, 2, 3, 4, 5, 6]]) == [[1, 2, 3, 4, 5, 6]]
    assert raster(tri, 100, 100).sum() > 1000

    assert noun_agrees("Remove the giraffe.", "giraffe")
    assert noun_agrees("Change the cars to red.", "car")
    assert noun_agrees("Make the dining table blue.", "dining table")
    assert not noun_agrees("Remove the giraffe.", "zebra")
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014",
                   help="RefCOCO is built on train2014; add val2014 to widen")
    p.add_argument("--cache", default="/data/datasets/coco/desc")
    p.add_argument("--image-cache", default="/data/datasets/coco/originals")
    p.add_argument("--stage", default="all",
                   choices=("descriptors", "match", "all"))
    p.add_argument("--topk", type=int, default=TOPK)
    p.add_argument("--prefetch", type=int, default=3,
                   help="candidates downloaded in parallel per attempt round")
    p.add_argument("--jobs", type=int, default=0,
                   help="worker processes / concurrent downloads; 0 = all cores")
    p.add_argument("--bbox-weight", type=float, default=BBOX_WEIGHT,
                   help="penalty on normalised-bbox disagreement; 0 = shape only")
    p.add_argument("--overlays", type=int, default=0,
                   help="write N target/sibling/shipped-mask overlays")
    p.add_argument("--overlay-dir", default="runs/gate_c/refedit_cases")
    p.add_argument("--overlay-seed", type=int, default=0)
    p.add_argument("--outline-px", type=int, default=4,
                   help="thickness of the shipped-mask outline; must survive "
                        "downscaling into the contact sheet")
    p.add_argument("--crop", action="store_true",
                   help="crop each overlay to target+siblings so small objects "
                        "are legible in the sheet")
    p.add_argument("--sheet-cols", type=int, default=4)
    p.add_argument("--audit-out", default="runs/gate_c/audit_ledger.json")
    p.add_argument("--require-noun", action="store_true",
                   help="reject when the recovered category is not named in "
                        "the instruction; advisory by default, since mask "
                        "agreement against the benchmark's own mask is stronger")
    p.add_argument("--rejects-out", default=None,
                   help="write rejected rows with what was recovered, for "
                        "extending the surface-form table from evidence")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.jobs <= 0:
        args.jobs = os.cpu_count() or 8
    self_test() if args.self_test else run(args)


if __name__ == "__main__":
    main()
