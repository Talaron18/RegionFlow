"""
Does the PIE-Bench++ mirror actually contain masks?

Every decoded mask came out full-image and 171 failed the sum check. Before
touching decode_rle, establish what the strings contain. Three outcomes:

  A. Most strings are "0 N" placeholders -> the mirror has NO usable masks.
     PIE-Bench++ was built for multi-aspect prompt evaluation, and may simply
     have dropped the spatial annotations. If so, fill the Google Form and get
     the original PIE-Bench; no amount of decoder fixing will help.
  B. Strings have many runs but sums that mismatch the image -> a resolution
     convention problem. Fixable: decode at the encoded size, then resize.
  C. Strings have many runs and matching sums -> only the run polarity is
     wrong. Fixable in one line.

    python scripts/diagnose_masks.py /data/datasets
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/datasets")
    files = sorted(root.rglob("*.parquet"))
    if not files:
        print(f"no parquet under {root}")
        sys.exit(1)

    print(f"{len(files)} parquet files")
    for f in files[:12]:
        print(f"  {f.relative_to(root)}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\n{len(df)} rows total\n")

    # ---- how many runs does each mask string have? -----------------------
    runs = df["mask"].astype(str).str.split().apply(len)
    print("runs per mask string:")
    print(f"  min {runs.min()}  median {int(runs.median())}  max {runs.max()}")
    print(f"  exactly 2 runs (i.e. a '0 N' placeholder): {int((runs == 2).sum())}"
          f" / {len(runs)}")

    print("\nmost common mask strings:")
    for val, n in Counter(df["mask"].astype(str)).most_common(5):
        print(f"  {n:>4}x  {val[:70]}{'...' if len(val) > 70 else ''}")

    # ---- run sums vs actual image sizes ----------------------------------
    print("\nrun sum vs image size (first 12 rows):")
    print(f"  {'id':<16} {'img w x h':<14} {'w*h':>10} {'run sum':>10} {'runs':>6}")
    sums, sizes = [], []
    for _, row in df.head(12).iterrows():
        img = row["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        w, h = img.size
        s = int(sum(int(x) for x in str(row["mask"]).split()))
        n = len(str(row["mask"]).split())
        print(f"  {str(row['id'])[:15]:<16} {f'{w} x {h}':<14} {w*h:>10} {s:>10} {n:>6}")
        sums.append(s)
        sizes.append(w * h)

    all_sums = df["mask"].astype(str).apply(lambda s: sum(int(x) for x in s.split()))
    print("\ndistinct run sums across the whole set:")
    for val, n in Counter(all_sums).most_common(6):
        side = np.sqrt(val)
        note = f"  = {side:.0f}^2" if abs(side - round(side)) < 1e-6 else ""
        print(f"  {n:>4}x  {val}{note}")

    # ---- verdict ---------------------------------------------------------
    placeholder_frac = float((runs <= 2).mean())
    print("\n" + "=" * 70)
    if placeholder_frac > 0.8:
        print(
            "VERDICT A: the mirror has NO usable masks.\n"
            f"  {placeholder_frac:.0%} of entries are 2-run placeholders carrying no\n"
            "  spatial information. PIE-Bench++ targets multi-aspect prompt\n"
            "  evaluation and appears to have dropped the mask annotations.\n\n"
            "  Get the ORIGINAL PIE-Bench: the form linked at line 78 of\n"
            "  PnPInversion/README.md -> https://forms.gle/hVMkTABb4uvZVjme9\n"
            "  Its mapping_file.json has real per-image masks, and load_pie_bench\n"
            "  already targets that layout."
        )
    elif len(set(all_sums)) <= 3 and len(set(sizes)) > 1:
        print(
            "VERDICT B: masks are encoded at a FIXED resolution while images\n"
            "  vary. Decode at the encoded size, then resize to the image with\n"
            "  nearest-neighbour. Relax the sum check in decode_rle accordingly."
        )
    else:
        print(
            "VERDICT C: runs look real. Only the polarity is in question —\n"
            "  compare a decoded mask against the image to pick the direction."
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
