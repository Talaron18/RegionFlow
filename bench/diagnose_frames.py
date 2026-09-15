"""
Settle two questions the mask-based probe could not.

WHY THE EARLIER PROBE FAILED. `probe_geometry` scored each candidate transform
by how well a COCO instance mask aligned with the MagicBrush EDIT mask. That
only discriminates if the edit mask is a tight object mask. If it is loose, or
inverted, or nearly the whole frame, every transform scores badly and the probe
reports a narrow margin -- which reads as "the transforms are equivalent" when
it actually means "the instrument cannot see".

This module asks both questions without using any annotation.

Q1 -- IS THE ALPHA DECODE RIGHT, AND HOW TIGHT IS THE MASK?
    Pure measurement of the edit mask's area fraction under both polarities.
        ~0.90  -> the polarity is inverted; `alpha < 128` is wrong for this
                  release and every IoU computed so far is meaningless
        ~0.05-0.35 -> polarity right, and the value tells you how loose the
                  hand-drawn region is. MagicBrush masks are crowdworker
                  DALL-E 2 inpainting regions, not object boundaries, so a
                  correctly aligned tight COCO instance can still only reach
                  IoU ~= area(inst)/area(edit).
    If Q1 says "loose", IoU is the wrong statistic for filter 3 and coverage
    (|edit & inst| / |inst|) is the right one.

Q2 -- HOW WAS THE COCO FRAME MADE SQUARE?
    COCO images are 640x480 etc; MagicBrush `source_img` is 500x500. Download
    the COCO original via `coco_url`, apply each candidate transform, and
    compare to `source_img` DIRECTLY, in pixels. No mask is involved, so
    nothing about mask quality can confound the answer. The winner should be
    unambiguous -- a wrong transform misplaces image content, which a
    correlation on downsampled grayscale sees immediately.

    If NO transform scores well, the framing assumption itself is wrong: the
    source may not be derived from the COCO original by a global resample at
    all, and the whole COCO-join plan needs rethinking rather than patching.

USAGE
    python bench/diagnose_frames.py --split dev            # Q1 only, offline
    python bench/diagnose_frames.py --split dev --fetch 16 # Q1 + Q2
    python bench/diagnose_frames.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SQUARE_MODES = ("stretch", "center_crop", "pad")


# ------------------------------------------------------------------- utils --

def cell_bytes(cell):
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unhandled image cell {type(cell).__name__}")


def to_square(img: Image.Image, side: int, mode: str) -> Image.Image:
    w, h = img.size
    if mode == "stretch":
        out = img
    elif mode == "center_crop":
        s = min(w, h)
        out = img.crop(((w - s) // 2, (h - s) // 2,
                        (w - s) // 2 + s, (h - s) // 2 + s))
    elif mode == "pad":
        s = max(w, h)
        out = Image.new("RGB", (s, s), (0, 0, 0))
        out.paste(img.convert("RGB"), ((s - w) // 2, (s - h) // 2))
    else:
        raise ValueError(mode)
    return out.convert("RGB").resize((side, side), Image.BILINEAR)


def ncc(a: Image.Image, b: Image.Image, side: int = 128) -> float:
    """Normalised cross-correlation on downsampled grayscale. 1.0 == identical.

    Downsampling is deliberate: it makes the score insensitive to the DALL-E
    resample and JPEG artefacts while staying very sensitive to CONTENT BEING
    IN THE WRONG PLACE, which is the only thing being asked about.
    """
    x = np.asarray(a.convert("L").resize((side, side), Image.BILINEAR), np.float64)
    y = np.asarray(b.convert("L").resize((side, side), Image.BILINEAR), np.float64)
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else 0.0


# ---------------------------------------------------------------------- Q1 --

def q1_mask_polarity(rows, out_dir: Path | None = None, overlays: int = 8):
    below, above, areas = [], [], []
    for row in rows:
        img = Image.open(io.BytesIO(cell_bytes(row["mask_img"])))
        mode = img.mode
        if mode != "RGBA":
            img = img.convert("RGBA")
        alpha = np.asarray(img)[..., 3]
        n = alpha.size
        below.append(float((alpha < 128).sum() / n))
        above.append(float((alpha >= 128).sum() / n))
        areas.append(mode)

    b, a = np.array(below), np.array(above)
    print("=" * 68)
    print("Q1  edit-mask area fraction")
    print("=" * 68)
    print(f"  source alpha modes seen: {collections.Counter(areas).most_common()}")
    for name, v in (("alpha <  128  (assumed edit region)", b),
                    ("alpha >= 128  (the complement)", a)):
        print(f"  {name}\n"
              f"      median {np.median(v):.3f}  mean {v.mean():.3f}  "
              f"p10 {np.percentile(v, 10):.3f}  p90 {np.percentile(v, 90):.3f}")
    if np.median(b) > 0.6:
        print("\n  -> POLARITY LOOKS INVERTED. `alpha < 128` selects most of the\n"
              "     frame. Every IoU computed so far is against a near-full mask.")
    elif np.median(b) < 0.001:
        print("\n  -> MASK IS EMPTY under this polarity. The alpha channel may be\n"
              "     fully opaque; the edit region is encoded some other way.")
    else:
        print(f"\n  -> polarity plausible. Median edit region is "
              f"{np.median(b):.1%} of frame.\n"
              "     A TIGHT instance inside a loose mask caps IoU at roughly\n"
              "     area(inst)/area(edit) even when perfectly aligned, so use\n"
              "     coverage = |edit & inst| / |inst| for filter 3, not IoU.")

    if out_dir and overlays:
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in rows[:overlays]:
            src = Image.open(io.BytesIO(cell_bytes(row["source_img"]))).convert("RGB")
            m = np.asarray(Image.open(io.BytesIO(cell_bytes(row["mask_img"])))
                           .convert("RGBA"))[..., 3] < 128
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                           .resize(src.size, Image.NEAREST), bool)
            arr = np.asarray(src).astype(np.float32)
            arr[m] = 0.5 * arr[m] + 0.5 * np.array([255, 230, 40], np.float32)
            Image.fromarray(arr.astype(np.uint8)).save(
                out_dir / f"editmask_{row['img_id']}.png")
        print(f"\n  {min(overlays, len(rows))} edit-mask overlays -> {out_dir}/")
    return {"below": b, "above": a}


# ---------------------------------------------------------------------- Q2 --

def q2_frame_transform(rows, coco_url, cache: Path, fetch: int):
    cache.mkdir(parents=True, exist_ok=True)
    scores = {m: [] for m in SQUARE_MODES}
    per_image = []

    done = 0
    for row in rows:
        if done >= fetch:
            break
        url = coco_url.get(row["coco_id"])
        if not url:
            continue
        dst = cache / f"{row['coco_id']}.jpg"
        if not dst.exists():
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as exc:  # noqa: BLE001
                print(f"  fetch failed {row['coco_id']}: {exc}")
                continue
        try:
            orig = Image.open(dst).convert("RGB")
            src = Image.open(io.BytesIO(cell_bytes(row["source_img"]))).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  decode failed {row['coco_id']}: {exc}")
            continue

        side = src.size[0]
        row_scores = {m: ncc(to_square(orig, side, m), src) for m in SQUARE_MODES}
        for m, v in row_scores.items():
            scores[m].append(v)
        per_image.append((row["coco_id"], orig.size, row_scores))
        done += 1

    if not per_image:
        print("\nQ2  no images fetched; skipping")
        return {}

    print("\n" + "=" * 68)
    print(f"Q2  COCO original vs source_img, {len(per_image)} images, NCC")
    print("=" * 68)
    print(f"  {'coco_id':>10} {'orig':>11}  " +
          "  ".join(f"{m:>12}" for m in SQUARE_MODES))
    for cid, size, s in per_image[:20]:
        star = max(s, key=s.get)
        print(f"  {cid:>10} {str(size):>11}  " +
              "  ".join(f"{s[m]:>11.3f}{'*' if m == star else ' '}"
                        for m in SQUARE_MODES))

    print()
    means = {}
    for m in SQUARE_MODES:
        v = np.array(scores[m])
        means[m] = float(v.mean())
        wins = sum(1 for _, _, s in per_image if max(s, key=s.get) == m)
        print(f"  {m:<13} mean NCC {v.mean():>6.3f}   median {np.median(v):>6.3f}"
              f"   wins {wins}/{len(per_image)}")

    win = max(means, key=means.get)
    runner = sorted(means.values())[-2]
    print(f"\n  winner: {win} (mean {means[win]:.3f}, {means[win] - runner:+.3f} over next)")
    if means[win] > 0.9:
        print("  -> decisive. Use this transform.")
    elif means[win] > 0.7:
        print("  -> probably right but not clean; check a few side by side.")
    else:
        print("  -> NO transform reproduces source_img from the COCO original.\n"
              "     source_img is not a global resample of the COCO image, so the\n"
              "     COCO join cannot be made geometrically exact by choosing a\n"
              "     transform. Rethink the plan rather than patch it.")
    return means


# -------------------------------------------------------------------- main --

def load_rows(mb_root: str, split: str, coco_ann: str):
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted(glob.glob(str(Path(mb_root) / "data" / f"{split}-*.parquet")))
    assert files, f"no {split} shards under {mb_root}/data"
    d = pq.read_table(files).to_pydict()

    norm = lambda x: int("".join(c for c in str(x) if c.isdigit()))  # noqa: E731
    rows = []
    for i in range(len(d["img_id"])):
        if d["turn_index"][i] != 1:
            continue
        rows.append({"img_id": d["img_id"][i], "coco_id": norm(d["img_id"][i]),
                     "source_img": d["source_img"][i], "mask_img": d["mask_img"][i]})

    urls = {}
    for s in ("train2014", "val2014"):
        p = Path(coco_ann) / f"instances_{s}.json"
        for im in json.loads(p.read_text())["images"]:
            urls[im["id"]] = im.get("coco_url")
    return rows, urls


def self_test():
    a = Image.new("RGB", (640, 480), (20, 20, 20))
    for x in range(0, 640, 80):
        Image.Image.paste(a, Image.new("RGB", (40, 480), (230, 90, 40)), (x, 0))
    assert to_square(a, 500, "stretch").size == (500, 500)
    assert to_square(a, 500, "center_crop").size == (500, 500)
    assert to_square(a, 500, "pad").size == (500, 500)

    ref = to_square(a, 500, "center_crop")
    s = {m: ncc(to_square(a, 500, m), ref) for m in SQUARE_MODES}
    assert s["center_crop"] > 0.999, s
    assert s["center_crop"] > s["stretch"], s
    assert ncc(ref, ref) > 0.999

    buf = io.BytesIO()
    al = np.full((64, 64, 4), 255, np.uint8)
    al[8:24, 8:24, 3] = 0
    Image.fromarray(al, "RGBA").save(buf, format="PNG")
    out = q1_mask_polarity([{"mask_img": {"bytes": buf.getvalue()},
                             "source_img": None, "img_id": "x"}], None, 0)
    assert abs(out["below"][0] - 256 / 4096) < 1e-9, out["below"][0]
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magicbrush", default="/data/datasets/MagicBrush")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--split", default="dev")
    p.add_argument("--fetch", type=int, default=0,
                   help="download N COCO originals and run Q2 (~150KB each)")
    p.add_argument("--cache", default="/data/datasets/coco/originals")
    p.add_argument("--overlays", type=int, default=8)
    p.add_argument("--overlay-dir", default="runs/gate_c/editmasks")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    rows, urls = load_rows(args.magicbrush, args.split, args.coco_annotations)
    print(f"{len(rows)} turn-1 rows\n")
    q1_mask_polarity(rows, Path(args.overlay_dir), args.overlays)
    if args.fetch:
        q2_frame_transform(rows, urls, Path(args.cache), args.fetch)


if __name__ == "__main__":
    main()
