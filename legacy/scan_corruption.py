#!/usr/bin/env python3
"""Scope size-preserved NUL corruption on /dev/md0 before refetching anything.

Two independent checks, because they cover different files:

  verify-hf   HF cache blob names ARE the content hash -- sha256 for LFS
              files (64 hex chars), git blob sha1 for small ones (40).
              This is ground truth and covers the safetensors, which the
              JSON scan does not touch.

  scan        Everything with no reference hash: the latent cache, the
              checkpoints, Gate C outputs, the repo itself. Reports files
              that are entirely NUL and files with a long interior NUL run.

USAGE
    python scan_corruption.py verify-hf /root/autodl-tmp/regionflow-data/hf
    python scan_corruption.py scan /root/autodl-tmp/regionflow-data/<cache_dir>
    python scan_corruption.py scan /root/RegionFlow --ext .py

READING THE OUTPUT. A long NUL run is evidence, not a verdict. Adapter
checkpoints contain zero-initialised LoRA B matrices BY DESIGN -- that is
the thing you have previously gone looking for on purpose. Judge a .pt by
`inspect-ckpt` below, not by its zero fraction. A NUL run inside a latent
cache shard has no benign explanation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

CHUNK = 8 << 20


def git_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as fh:
        while (b := fh.read(CHUNK)):
            h.update(b)
    return h.hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while (b := fh.read(CHUNK)):
            h.update(b)
    return h.hexdigest()


def cmd_verify_hf(args) -> int:
    root = Path(args.root)
    blobs = sorted(root.rglob("blobs/*"))
    if not blobs:
        print(f"no blobs under {root} -- is this the HF_HOME root?")
        return 2
    bad, skipped, ok = [], 0, 0
    for b in blobs:
        if b.is_symlink() or not b.is_file():
            continue
        name = b.name
        if b.suffix == ".incomplete":
            bad.append((b, "incomplete download"))
            continue
        if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
            got = sha256(b)
        elif len(name) == 40 and all(c in "0123456789abcdef" for c in name):
            got = git_sha1(b)
        else:
            skipped += 1
            continue
        if got != name:
            bad.append((b, f"hash {got[:12]} != name {name[:12]}"))
        else:
            ok += 1
        print(f"\r  checked {ok + len(bad)}/{len(blobs)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    for b, why in bad:
        sz = b.stat().st_size
        zero = is_all_zero(b)
        print(f"CORRUPT {b}  ({sz} B, {why}{', ALL NUL' if zero else ''})")
    print(f"\n{ok} verified, {len(bad)} corrupt, {skipped} unhashable "
          f"(name is not an etag)")
    return 1 if bad else 0


def is_all_zero(path: Path) -> bool:
    with path.open("rb") as fh:
        while (b := fh.read(CHUNK)):
            if b.count(0) != len(b):
                return False
    return True


def zero_profile(path: Path):
    """(total, n_zero, longest_run). One pass, no whole-file buffering."""
    total = n_zero = longest = run = 0
    with path.open("rb") as fh:
        while (b := fh.read(CHUNK)):
            total += len(b)
            n_zero += b.count(0)
            for byte in b:
                if byte == 0:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
    return total, n_zero, longest


def cmd_scan(args) -> int:
    root = Path(args.root)
    thresh = args.min_run
    hits = []
    files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
    if args.ext:
        files = [p for p in files if p.suffix in args.ext]
    for i, p in enumerate(files, 1):
        print(f"\r  scanned {i}/{len(files)}", end="", file=sys.stderr)
        try:
            sz = p.stat().st_size
            if sz == 0:
                hits.append((p, 0, 0, 0, "ZERO LENGTH"))
                continue
            total, nz, longest = zero_profile(p)
            if nz == total:
                hits.append((p, total, nz, longest, "ALL NUL"))
            elif longest >= thresh:
                hits.append((p, total, nz, longest,
                             f"NUL run {longest} B"))
        except OSError as exc:
            hits.append((p, -1, -1, -1, f"UNREADABLE {exc}"))
    print(file=sys.stderr)
    for p, total, nz, longest, why in hits:
        frac = f"{nz / total:.1%}" if total > 0 else "-"
        print(f"{why:<24} {frac:>7} zero  {total:>14,} B  {p}")
    print(f"\n{len(files)} files, {len(hits)} flagged "
          f"(min interior run {thresh} B)")
    return 1 if hits else 0


def cmd_inspect_ckpt(args) -> int:
    """Judge a .pt by its tensors, not its zero fraction."""
    import torch
    sd = torch.load(args.path, map_location="cpu", weights_only=False)
    print(f"step={sd.get('step')} best_val={sd.get('best_val')}")
    tensors = sd.get("adapter", {})
    if not tensors:
        print("no 'adapter' key -- wrong checkpoint layout?")
        return 2
    n_zero = n_nan = 0
    for k, v in tensors.items():
        if not hasattr(v, "numel"):
            continue
        f = v.float()
        allz = bool((f == 0).all())
        nan = bool(torch.isnan(f).any() or torch.isinf(f).any())
        n_zero += allz
        n_nan += nan
        if nan:
            print(f"  NaN/Inf  {k}  {tuple(v.shape)}")
    print(f"{len(tensors)} tensors, {n_zero} all-zero, {n_nan} with NaN/Inf")
    print("all-zero is EXPECTED for LoRA B at init and for an untrained "
          "interval embedding. NaN/Inf is not, and neither is an all-zero "
          "tensor in a checkpoint that trained.")
    if "opt" in sd:
        print("optimiser state present -- resumable")
    return 1 if n_nan else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify-hf")
    v.add_argument("root")
    v.set_defaults(fn=cmd_verify_hf)

    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--min-run", type=int, default=1 << 20,
                   help="flag interior NUL runs at or above this (default 1 MiB)")
    s.add_argument("--ext", nargs="*", default=None,
                   help="limit to these suffixes, e.g. --ext .pt .safetensors")
    s.set_defaults(fn=cmd_scan)

    c = sub.add_parser("inspect-ckpt")
    c.add_argument("path")
    c.set_defaults(fn=cmd_inspect_ckpt)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
