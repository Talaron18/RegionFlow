#!/usr/bin/env python
"""Diff the CONFIG of two runs, read from the checkpoints themselves.

Why not `diff -r` the two run directories: the sidecar files are unreliable.
`train_args.json` only exists for runs launched after the provenance patch,
and a run directory's other files (stdout, val csv) do not contain the
effective cap at all. Every checkpoint, however, has carried

    torch.save({"step": ..., "config": asdict(self.cfg), ...})

since transport/train.py:212, and that copy is the config the run ACTUALLY
used rather than a re-typed one. So this is the authoritative diff.

    python scripts/diff_run_config.py runs/curr_a0_control_s1/ckpt_final.pt \
                                      runs/curr_a2_anchor_s1/ckpt_final.pt

Exit status is 1 when any non-trivial key differs, so it can gate a script.
Keys that necessarily differ between two run directories (`out`, `device`)
are reported separately as "expected" rather than counted as differences.

    python scripts/diff_run_config.py --selftest    # no torch, no checkpoints
"""
import json
import sys
from pathlib import Path

# differ by construction (different directories / hardware), not by design
EXPECTED = ("out", "out_dir", "device")

# the gate_b CLI keys worth surfacing when a train_args.json exists
CAP_KEYS = ("loss_path", "delta_cap_effective", "max_delta", "curriculum",
            "cap_start", "cap_end", "warmup_frac", "anchor_mode",
            "anchor_width", "data_proportion", "faithful", "custom",
            "seed", "steps")

# Worth printing even when IDENTICAL: the "same" list shows only key NAMES, and
# these are the keys whose VALUE changes how a run must be read. Concrete case:
# grad_clip_frac == 0 alongside grad_norm > 1e4 looks like a broken metric until
# you see the threshold -- train.py:285 compares the PRE-clip norm, so it only
# means grad_clip itself is > 1e4 and clipping never binds.
ALWAYS_SHOW = ("steps", "lr", "grad_clip", "grad_accum", "batch_size", "seed",
               "val_every", "delta_start", "delta_end", "delta_warmup_frac",
               "weight_decay", "ema_decay", "keep_last", "ckpt_every")


def _fmt(v, width=40):
    s = str(v)
    return s if len(s) <= width else s[:width - 3] + "..."


def compare(ca, cb):
    """-> (same_keys, expected_keys, diffs) with diffs = [(k, va, vb), ...]"""
    same, expected, diffs = [], [], []
    for k in sorted(set(ca) | set(cb)):
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        if va == vb:
            same.append(k)
        elif k in EXPECTED:
            expected.append((k, va, vb))
        else:
            diffs.append((k, va, vb))
    return same, expected, diffs


def report(label, ca, cb):
    """Print one comparison block. Returns the number of real differences."""
    same, expected, diffs = compare(ca, cb)
    print(f"\n--- {label} ---")
    print(f"same ({len(same)}): " + ", ".join(same) if same else "same (0)")
    if not diffs:
        print("no differing keys")
    else:
        w = max(len(k) for k, _, _ in diffs)
        print("DIFFERS:")
        for k, va, vb in diffs:
            print(f"  {k:<{w}}  A={_fmt(va):<42} B={_fmt(vb)}")
    if expected:
        w = max(len(k) for k, _, _ in expected)
        print("expected (run dir / hardware, not a design difference):")
        for k, va, vb in expected:
            print(f"  {k:<{w}}  A={_fmt(va):<42} B={_fmt(vb)}")
    return len(diffs)


def sidecar(ckpt_path):
    p = Path(ckpt_path).parent / "train_args.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:                      # noqa: BLE001
        return {"<unreadable>": f"{exc}"}


def load(ckpt_path):
    import torch                                 # noqa: PLC0415
    # PLAIN LOAD, ON PURPOSE. `mmap=True` was tried here and HUNG on the box's
    # torch 2.8: children pinned it at ~500 MB RSS for 8-14 minutes with no
    # output, and `tests/test_diff_run_config.py` (no subprocess timeout) then
    # hung with them. It passed on torch 2.12 locally, which is exactly why it
    # shipped. The machine has 1 TB of RAM (`free -g`: 1007 total, 759
    # available), so reading a checkpoint that also carries `opt` and `ema`
    # (transport/train.py:207-221) is a non-issue. Do NOT re-add mmap without a
    # timeout-guarded test on the TARGET torch version.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in sd:
        raise SystemExit(f"{ckpt_path}: no 'config' in the checkpoint -- it "
                         f"predates transport/train.py:212, fall back to "
                         f"diffing the run directory by hand")
    return sd


def selftest():
    a = {"steps": 5000, "seed": 0, "lr": 1e-4, "out_dir": "x", "gate": "b"}
    b = {"steps": 5000, "seed": 1, "lr": 1e-4, "out_dir": "y", "gate": "b"}
    same, expected, diffs = compare(a, b)
    assert same == ["gate", "lr", "steps"], same
    assert [k for k, _, _ in expected] == ["out_dir"], expected
    assert [k for k, _, _ in diffs] == ["seed"], diffs
    # a key present in only one config must read as a difference, not vanish
    same, expected, diffs = compare({"a": 1}, {"b": 2})
    # A's value first, B's second; a key missing on one side must show as
    # "<absent>" on that side rather than being silently dropped
    assert diffs == [("a", 1, "<absent>"), ("b", "<absent>", 2)], diffs
    # equal NaN-free floats compare equal; ints and floats must not be equal
    assert compare({"x": 1}, {"x": 1.0})[2] == [], "1 == 1.0 is intended"
    # the report must not crash on an empty diff
    assert report("selftest", {"a": 1}, {"a": 1}) == 0
    print("selftest ok")
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    pos = [a for a in argv[1:] if not a.startswith("-")]
    if len(pos) != 2:
        print(__doc__)
        return 2
    pa, pb = pos
    for p in (pa, pb):
        if not Path(p).exists():
            raise SystemExit(f"missing checkpoint: {p}")

    n = 0
    # Size first: the checkpoint carries `opt` + `ema` as well as the adapter
    # (transport/train.py:207-221), so "how big is this file" is a real question
    # and answering it is cheaper than guessing.
    for tag, p in (("A", pa), ("B", pb)):
        print(f"{tag}: {p}   file {Path(p).stat().st_size / 2 ** 20:.0f} MB")
    sda, sdb = load(pa), load(pb)
    print(f"A: saved step {sda.get('step')}   B: saved step {sdb.get('step')}")

    print("\nvalues  A | B  (printed even when equal -- see ALWAYS_SHOW)")
    w = max(len(k) for k in ALWAYS_SHOW)
    ca, cb = sda["config"], sdb["config"]
    for k in ALWAYS_SHOW:
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        mark = "   <-- DIFFERS" if va != vb else ""
        print(f"  {k:<{w}}  {_fmt(va, 26):<28} | {_fmt(vb, 26)}{mark}")

    n += report("TrainConfig (embedded in the checkpoint)", ca, cb)

    aa, ab = sidecar(pa), sidecar(pb)
    print("\n--- gate_b CLI (train_args.json sidecar) ---")
    for tag, args, path in (("A", aa, pa), ("B", ab, pb)):
        if args is None:
            print(f"{tag}: absent -- this run predates the provenance patch, "
                  f"so the trained Delta cap is NOT recoverable from "
                  f"{Path(path).parent}/. Get it from the training log, or "
                  f"treat the cap as unknown.")
    if aa is not None and ab is not None:
        keep = {k: {kk: v.get(kk, "<absent>") for kk in CAP_KEYS}
                for k, v in (("A", aa), ("B", ab))}
        for k, v in keep.items():
            print(f"{k}: " + "  ".join(f"{kk}={_fmt(v[kk], 24)}"
                                       for kk in CAP_KEYS if v[kk] != "<absent>"))
        n += report("gate_b CLI", aa, ab)
    elif aa is not None or ab is not None:
        # can't diff, but the surviving sidecar is still worth printing
        tag, args = ("A", aa) if aa is not None else ("B", ab)
        print(f"{tag}: " + "  ".join(f"{kk}={_fmt(args.get(kk), 24)}"
                                     for kk in CAP_KEYS if kk in args))
        print("only one side has a sidecar -- the CLI-level comparison needs "
              "either both sidecars or both training logs (grep the launch "
              "line for --curriculum / --max-delta / --anchor-mode)")
    print(f"\n{n} real difference(s)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
