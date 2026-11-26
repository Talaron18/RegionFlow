#!/usr/bin/env python3
"""Finish the throughput edits: the DataLoader line, plus a state report.

    python fix_loader.py                 # report + apply the loader edit
    python fix_loader.py --report-only   # just tell me what is in the file

Your file already has the checkpointing edit, the CLI flags and the bench
routing; only the DataLoader edit is missing, and its anchor did not match
verbatim. This finds the DataLoader call structurally (by balancing
parentheses from `loader = DataLoader(`) instead of matching its text, so the
argument order and line breaks inside it do not matter.

It also reports how many times each throughput symbol appears, because a
half-applied patch can leave a second copy of run_bench(), which Python
accepts silently -- the later definition just wins.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

NEW_LOADER = '''    # THROUGHPUT. batch x accum is the optimiser batch and is unchanged by
    # how it is split: the loss is a per-sample mean, so batch=8/accum=2 and
    # batch=2/accum=8 give the same gradient. Bigger micro-batches are
    # strictly better GPU utilisation IF they fit -- the 5k run peaked at
    # 9.5 of 24 GB with batch=2, so there is headroom. Measure, do not assume:
    # --bench-steps times a few steps and exits.
    #
    # persistent_workers keeps the loader workers alive across epochs. The
    # trainer re-iterates the loader on StopIteration, which respawns them
    # every epoch otherwise; at batch=2 an epoch is short.
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    print(f"[data] {len(ds)} cached samples, batch {args.batch} x accum "
          f"{args.accum} = {args.batch * args.accum} per step, "
          f"{args.num_workers} workers")
'''

SYMBOLS = [
    ("def run_bench", 1, "the bench function"),
    ("return run_bench(", 1, "bench routed before Trainer"),
    ('"--bench-steps"', 1, "--bench-steps flag"),
    ('"--num-workers"', 1, "--num-workers flag"),
    ('"--no-grad-checkpointing"', 1, "--no-grad-checkpointing flag"),
    ("if args.grad_checkpointing:", 1, "checkpointing made optional"),
    ("persistent_workers", 1, "loader edit (this script)"),
    ("import tempfile", None, "tempfile import (module or local, either is ok)"),
]


def find_loader_span(src: str):
    """Return (start, end) of the whole `loader = DataLoader(...)` statement."""
    m = re.search(r"^[ \t]*loader\s*=\s*DataLoader\(", src, re.M)
    if not m:
        return None
    i = src.index("(", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    end = src.index("\n", j) + 1
    return m.start(), end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="scripts/gate_b_meanflow.py")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}  (run from the repo root, or pass --path)")
        return 1
    src = path.read_text()
    print(f"{path}  sha256 {hashlib.sha256(src.encode()).hexdigest()[:12]}\n")

    print("state:")
    problems = []
    for needle, want, label in SYMBOLS:
        n = src.count(needle)
        flag = ""
        if want is not None and n > want:
            flag = f"  <-- {n} copies, expected {want}"
            problems.append((needle, n))
        print(f"  {n:>2}x  {label}{flag}")
    if problems:
        print("\nDuplicates above mean a patch was applied twice. Python keeps "
              "the LAST definition, so this runs but is confusing; delete the "
              "earlier copy by hand.")

    span = find_loader_span(src)
    if span is None:
        print("\nNo `loader = DataLoader(` statement found at all. Paste the "
              "lines around it and the edit can be re-cut.")
        return 1
    start, end = span
    current = src[start:end]
    print("\nfound this DataLoader statement:")
    for line in current.rstrip("\n").splitlines():
        print(f"  |{line}")

    if "persistent_workers" in current:
        print("\nalready has persistent_workers; nothing to do")
        return 0
    # argparse defines args.num_workers from the flag string, so check the
    # FLAG, not the attribute -- the attribute only appears in the new code
    if '"--num-workers"' not in src:
        print("\n--num-workers flag is missing, so the new loader would fail. "
              "Apply the CLI-flags edit first.")
        return 1
    if args.report_only:
        print("\nreport only: nothing written")
        return 0

    out = src[:start] + NEW_LOADER + src[end:]
    backup = path.with_suffix(path.suffix + ".bak2")
    backup.write_text(src)
    path.write_text(out)
    print(f"\nreplaced it; wrote {path}, backup at {backup}")
    print("now run:  python -m py_compile", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
