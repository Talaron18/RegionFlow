"""
Which shards do we actually have?

The Gate A cache was built from 14 of MagicBrush's 51 train shards, because
that is what had downloaded. Nothing in the pipeline noticed -- the loader
globs whatever is there and reports a plausible count. 1235 triplets looked
fine, and the run trained on roughly a quarter of the data.

Shard filenames carry their own denominator (train-00006-of-00051-<hash>),
so completeness is checkable without touching the network.

    python scripts/check_shards.py --root /data/datasets/MagicBrush
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"(?P<split>[a-z]+)-(?P<idx>\d+)-of-(?P<total>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        print(f"no parquet under {root}")
        return 1

    by_split: dict = defaultdict(lambda: {"have": set(), "total": 0, "bytes": 0})
    for p in shards:
        m = PAT.search(p.name)
        if not m:
            print(f"  unparsed filename: {p.name}")
            continue
        rec = by_split[m["split"]]
        rec["have"].add(int(m["idx"]))
        rec["total"] = int(m["total"])
        rec["bytes"] += p.stat().st_size

    incomplete = False
    for split, rec in sorted(by_split.items()):
        total, have = rec["total"], rec["have"]
        missing = sorted(set(range(total)) - have)
        pct = 100 * len(have) / total
        print(f"\n{split}: {len(have)}/{total} shards ({pct:.0f}%), "
              f"{rec['bytes']/2**30:.1f} GB")
        if missing:
            incomplete = True
            shown = ", ".join(f"{i:05d}" for i in missing[:12])
            more = f" ... (+{len(missing)-12} more)" if len(missing) > 12 else ""
            print(f"  MISSING {len(missing)}: {shown}{more}")
            print(f"  Expect roughly {len(have)}/{total:.0f} of the full "
                  "triplet count from this split.")
        else:
            print("  complete")

    if incomplete:
        print("\nRe-pull with:")
        print("  huggingface-cli download osunlp/MagicBrush --repo-type dataset \\")
        print(f"      --local-dir {root}")
        print("It skips files already present, so this only fetches the gaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
