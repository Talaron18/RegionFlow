"""
Find and validate whatever benchmark data is on disk.

Two possible layouts depending on which route you took:

  A. original PIE-Bench   -- mapping_file.json + data/annotation_images/
  B. PIE-Bench++ mirror   -- parquet files with embedded image bytes

This detects which is present, prints the real schema, loads it, and runs the
validator. Run it before any sweep.

    python scripts/inspect_data.py
    python scripts/inspect_data.py /some/other/path
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere, with or without PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEARCH_ROOTS = ["/data", "/workspace"]


def find_layouts(roots):
    json_hits, parquet_hits = [], []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for f in p.rglob("mapping_file*.json"):
            if "ti2i" not in f.name:          # skip the TI2I benchmark file
                json_hits.append(f)
        # a directory containing parquet is the unit, not each file
        seen = set()
        for f in p.rglob("*.parquet"):
            top = f.parent
            while top.parent != Path(root) and top.parent != Path("/"):
                top = top.parent
            if top not in seen:
                seen.add(top)
                parquet_hits.append(top)
    return json_hits, parquet_hits


def main():
    roots = [sys.argv[1]] if len(sys.argv) > 1 else SEARCH_ROOTS
    json_hits, parquet_hits = find_layouts(roots)

    print("=" * 78)
    print(f"searched: {', '.join(roots)}")
    print(f"  mapping_file.json : {[str(p) for p in json_hits] or 'none'}")
    print(f"  parquet dirs      : {[str(p) for p in parquet_hits] or 'none'}")
    print("=" * 78)

    if not json_hits and not parquet_hits:
        print(
            "\nNo benchmark data found.\n"
            "  original PIE-Bench: fill the form linked in PnPInversion/README.md\n"
            "  mirror:  hf download UB-CVML-Group/PIE_Bench_pp --repo-type dataset \\\n"
            "             --local-dir /data/datasets/PIE_Bench_pp"
        )
        sys.exit(1)

    from bench.data import validate

    if json_hits:
        from bench.data import inspect_schema, load_pie_bench

        mapping = json_hits[0]
        data_root = mapping.parent
        print(f"\n--- schema: {mapping} ---")
        print(inspect_schema(mapping, n=1))

        imgs = data_root / "annotation_images"
        if not imgs.exists():
            print(
                f"\n!! {imgs} missing. The mapping file alone cannot drive a "
                "sweep — the 700 images are a separate download."
            )
            sys.exit(1)

        import numpy as np
        from PIL import Image

        cases = load_pie_bench(
            data_root,
            json_name=mapping.name,
            load_image=lambda p: np.array(Image.open(p).convert("RGB")),
        )
    else:
        from bench.data import describe_parquet, load_pie_bench_pp

        root = parquet_hits[0]
        print(f"\n--- schema: {root} ---")
        print(describe_parquet(root, n=1))
        cases = load_pie_bench_pp(root)

    print("\n--- validation ---")
    print(validate(cases))

    print("\n--- mask sanity (the assumption most likely to be wrong) ---")
    import numpy as np

    fracs = np.array([float((c.target_mask > 0).mean()) for c in cases])
    print(f"  mask covers <1% of image : {int((fracs < 0.01).sum())} cases")
    print(f"  mask covers >90%         : {int((fracs > 0.90).sum())} cases")
    if (fracs > 0.90).mean() > 0.5:
        print(
            "\n  !! Most masks cover almost the whole image. The RLE runs are\n"
            "     probably inverted — flip the starting value in decode_rle."
        )
    elif (fracs < 0.01).mean() > 0.5:
        print("\n  !! Most masks are near-empty. Same suspicion, other direction.")
    else:
        print("  distribution looks plausible")

    print(f"\n{len(cases)} cases ready")


if __name__ == "__main__":
    main()
