"""
Fit the COCO -> MagicBrush crop rectangle PER IMAGE.

WHY. `diagnose_frames.py` Q2 showed that no single global transform reproduces
`source_img` from the COCO original -- but it also showed individual images
registering at NCC 0.996, 1.000, 0.879. An exact match on some images and a
0.2-0.7 partial match on the rest is not "different images". It is the same
image under a DIFFERENT RECTANGLE each time.

That is consistent with how MagicBrush was built: DALL-E 2's platform takes
square input, and crowdworkers supplied a squared version of each COCO photo.
Nothing forced that square to be centred, so there is no convention to guess.

WHAT THIS DOES. For each row, search the rectangle (x0, y0, w, h) in COCO
coordinates whose resample best matches `source_img`, scored by normalised
cross-correlation on downsampled grayscale. Coarse square-crop grid, then a
local refinement that lets w and h separate (so a stretch, or a crop plus a
slight aspect change, is reachable). Full-frame stretch is always evaluated as
a candidate so the degenerate case cannot be missed.

WHAT THE OUTPUT MEANS. `ncc_fit` per image is a SELF-VALIDATING gate:

    >= 0.95   registered. The rectangle is right; map COCO masks through it.
    0.80-0.95 probably right, visually check before trusting.
    <  0.80   did not register. Drop the case. Do NOT map masks through a
              rectangle that did not fit -- that is how a plausible-looking
              wrong answer gets into a benchmark.

The fraction reaching 0.95 is the real ceiling on how many Gate C cases this
join can yield, and it is measured rather than assumed.

IF ALMOST NOTHING REGISTERS at 4 free parameters, the hypothesis is dead --
`source_img` is then genuinely not a crop-and-resample of the COCO original --
and the COCO join has to be abandoned rather than patched further.

OUTPUT
    frames_<split>.json   {img_id: {rect, ncc_fit, coco_wh, src_wh}}
    consumed by magicbrush_coco.py --frames

USAGE
    python bench/frame_fit.py --self-test
    python bench/frame_fit.py --split dev --limit 40      # ~1 min, check first
    python bench/frame_fit.py --split dev --out bench/frames_dev.json
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

FIT_SIDE = 48          # grayscale resolution the NCC is computed at
WORK_MAX = 256         # COCO original is downsampled to this before searching


# ------------------------------------------------------------------- score --

def _gray(img: Image.Image, side: int = FIT_SIDE) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((side, side), Image.BILINEAR), np.float64)
    a -= a.mean()
    n = np.sqrt((a * a).sum())
    return a / n if n else a


def ncc_pre(a: np.ndarray, b: np.ndarray) -> float:
    """Both inputs already zero-mean and unit-norm from _gray."""
    return float((a * b).sum())


# --------------------------------------------------------------------- fit --

def _crop_score(work: Image.Image, ref: np.ndarray, rect) -> float:
    x0, y0, w, h = rect
    if w < 8 or h < 8:
        return -1.0
    patch = work.crop((int(x0), int(y0), int(x0 + w), int(y0 + h)))
    return ncc_pre(_gray(patch), ref)


def fit_rect(orig: Image.Image, src: Image.Image,
             coarse_steps: int = 9, scales: int = 7) -> dict:
    """Search the COCO rectangle that best reproduces `src`.

    Returns {rect (in ORIGINAL coordinates), ncc_fit, ncc_stretch}.
    """
    W, H = orig.size
    scale = min(1.0, WORK_MAX / max(W, H))
    work = orig.convert("L").resize((max(1, int(W * scale)), max(1, int(H * scale))),
                                    Image.BILINEAR)
    ww, wh = work.size
    ref = _gray(src)

    # always evaluate the degenerate candidate
    best = (0.0, 0.0, float(ww), float(wh))
    best_s = ncc_stretch = _crop_score(work, ref, best)

    # Coarse grid. Square crops are the likely case, but the grid also sweeps a
    # few aspect ratios: an aspect-preserving crop later stretched to square is
    # reachable only if the search starts near it -- local refinement cannot
    # walk from a square seed to a 4:3 rectangle.
    smin = min(ww, wh)
    for k in range(scales):
        side = smin * (1.0 - 0.45 * k / max(1, scales - 1))
        if side < 16:
            continue
        for aspect in (1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16):
            w = side * np.sqrt(aspect)
            h = side / np.sqrt(aspect)
            if w > ww or h > wh or w < 16 or h < 16:
                continue
            for i in range(coarse_steps):
                for j in range(coarse_steps):
                    x0 = (ww - w) * i / max(1, coarse_steps - 1)
                    y0 = (wh - h) * j / max(1, coarse_steps - 1)
                    s = _crop_score(work, ref, (x0, y0, w, h))
                    if s > best_s:
                        best_s, best = s, (x0, y0, w, h)

    # Refine. The neighbour set must be able to move ONE EDGE at a time
    # (x0 += d with w -= d), not just translate or resize the whole window --
    # without those the search stalls a few pixels short of the optimum and
    # reports ~0.87 where the true rectangle scores ~1.00.
    def neighbours(d):
        return (
            (d, 0, 0, 0), (-d, 0, 0, 0), (0, d, 0, 0), (0, -d, 0, 0),   # translate
            (0, 0, d, 0), (0, 0, -d, 0), (0, 0, 0, d), (0, 0, 0, -d),   # right/bottom edge
            (d, 0, -d, 0), (-d, 0, d, 0), (0, d, 0, -d), (0, -d, 0, d),  # left/top edge
            (0, 0, d, d), (0, 0, -d, -d),                                # scale corner
            (-d, -d, 2 * d, 2 * d), (d, d, -2 * d, -2 * d),              # scale centred
        )

    step = max(ww, wh) * 0.08
    for _ in range(40):
        improved = False
        x0, y0, w, h = best
        for dx, dy, dw, dh in neighbours(step):
            cx, cy, cw_, ch_ = x0 + dx, y0 + dy, w + dw, h + dh
            if cx < 0 or cy < 0 or cw_ < 8 or ch_ < 8:
                continue
            if cx + cw_ > ww or cy + ch_ > wh:
                continue
            s = _crop_score(work, ref, (cx, cy, cw_, ch_))
            if s > best_s:
                best_s, best, improved = s, (cx, cy, cw_, ch_), True
        if not improved:
            step *= 0.5
        if step < 0.25:
            break

    inv = 1.0 / scale
    rect = [round(float(v) * inv, 2) for v in best]
    return {"rect": rect, "ncc_fit": round(float(best_s), 4),
            "ncc_stretch": round(float(ncc_stretch), 4),
            "coco_wh": [W, H], "src_wh": list(src.size)}


def apply_rect(mask: np.ndarray, rect, out_hw) -> np.ndarray:
    """Map a COCO-frame boolean mask through a fitted rectangle."""
    x0, y0, w, h = [int(round(v)) for v in rect]
    H, Wd = mask.shape
    x0, y0 = max(0, min(x0, Wd - 1)), max(0, min(y0, H - 1))
    x1, y1 = min(Wd, x0 + max(1, w)), min(H, y0 + max(1, h))
    sub = mask[y0:y1, x0:x1]
    oh, ow = out_hw
    img = Image.fromarray(sub.astype(np.uint8) * 255)
    return np.asarray(img.resize((ow, oh), Image.NEAREST), dtype=bool)


# --------------------------------------------------------------------- io --

def load_rows(mb_root, split, coco_ann):
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted(glob.glob(str(Path(mb_root) / "data" / f"{split}-*.parquet")))
    assert files, f"no {split} shards under {mb_root}/data"
    d = pq.read_table(files).to_pydict()
    norm = lambda x: int("".join(c for c in str(x) if c.isdigit()))  # noqa: E731
    rows = [{"img_id": d["img_id"][i], "coco_id": norm(d["img_id"][i]),
             "source_img": d["source_img"][i]}
            for i in range(len(d["img_id"])) if d["turn_index"][i] == 1]
    urls = {}
    for s in ("train2014", "val2014"):
        for im in json.loads((Path(coco_ann) / f"instances_{s}.json").read_text())["images"]:
            urls[im["id"]] = im.get("coco_url")
    return rows, urls


def cell_bytes(cell):
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path"):
            return Path(cell["path"]).read_bytes()
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(type(cell).__name__)


def fetch(cid, url, cache: Path):
    dst = cache / f"{cid}.jpg"
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
    return dst


# ------------------------------------------------------------------- main --

def run(args):
    rows, urls = load_rows(args.magicbrush, args.split, args.coco_annotations)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.limit:
        rows = rows[:args.limit]
    print(f"fitting {len(rows)} rows\n")

    out, fails = {}, 0
    for n, row in enumerate(rows, 1):
        url = urls.get(row["coco_id"])
        if not url:
            fails += 1
            continue
        try:
            orig = Image.open(fetch(row["coco_id"], url, cache)).convert("RGB")
            src = Image.open(io.BytesIO(cell_bytes(row["source_img"]))).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  {row['coco_id']}: {type(exc).__name__} {exc}")
            fails += 1
            continue
        r = fit_rect(orig, src)
        out[str(row["img_id"])] = r
        if n % 25 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}")

    v = np.array([r["ncc_fit"] for r in out.values()]) if out else np.zeros(1)
    st = np.array([r["ncc_stretch"] for r in out.values()]) if out else np.zeros(1)
    print(f"\nfitted {len(out)} ({fails} failed)\n")
    print(f"  ncc_fit      median {np.median(v):.3f}  mean {v.mean():.3f}")
    print(f"  ncc_stretch  median {np.median(st):.3f}  (the global baseline)")
    print(f"  >= 0.95 registered   {(v >= 0.95).mean():>6.1%}  ({(v >= 0.95).sum()})")
    print(f"  0.80-0.95 check      {((v >= 0.80) & (v < 0.95)).mean():>6.1%}")
    print(f"  <  0.80 drop         {(v < 0.80).mean():>6.1%}")

    if (v >= 0.95).mean() > 0.5:
        print("\n  -> the varying-rectangle hypothesis holds. Use --frames "
              "downstream\n     and gate cases on ncc_fit >= 0.95.")
    elif (v >= 0.80).mean() > 0.5:
        print("\n  -> partial. Registration helps but is not exact; inspect "
              "overlays\n     before building anything on it.")
    else:
        print("\n  -> DID NOT REGISTER at 4 free parameters. source_img is not a\n"
              "     crop-and-resample of the COCO original. Abandon the COCO "
              "join.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return out


def self_test():
    # A photo-like reference: real images are dominated by low frequencies, and
    # NCC at FIT_SIDE is unforgiving of high-frequency content under a
    # sub-pixel offset. Unblurred noise would understate the fit quality that
    # the same code achieves on photographs.
    rng = np.random.default_rng(0)
    from PIL import ImageFilter  # noqa: PLC0415
    a = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    orig = Image.fromarray(a).resize((640, 480), Image.BICUBIC) \
                             .filter(ImageFilter.GaussianBlur(3))

    for truth in ((0, 0, 480, 480), (160, 0, 480, 480), (90, 30, 400, 400)):
        x0, y0, w, h = truth
        src = orig.crop((x0, y0, x0 + w, y0 + h)).resize((500, 500), Image.BILINEAR)
        r = fit_rect(orig, src)
        assert r["ncc_fit"] > 0.9, (truth, r)
        err = max(abs(r["rect"][i] - truth[i]) for i in range(4))
        assert err < 0.10 * max(w, h), (truth, r["rect"], err)

    # a full-frame stretch must be recoverable too
    src = orig.resize((500, 500), Image.BILINEAR)
    r = fit_rect(orig, src)
    assert r["ncc_fit"] > 0.9, r
    assert r["ncc_stretch"] > 0.9, r

    # an unrelated image must NOT register
    other = Image.fromarray(rng.integers(0, 255, (500, 500, 3), dtype=np.uint8))
    assert fit_rect(orig, other)["ncc_fit"] < 0.6, "noise should not register"

    # apply_rect places a mask where the rectangle says
    m = np.zeros((480, 640), bool)
    m[100:200, 300:400] = True
    got = apply_rect(m, [160, 0, 480, 480], (500, 500))
    assert got.any() and not got[:50].any(), got.sum()
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magicbrush", default="/data/datasets/MagicBrush")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--cache", default="/data/datasets/coco/originals")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    self_test() if args.self_test else run(args)


if __name__ == "__main__":
    main()
