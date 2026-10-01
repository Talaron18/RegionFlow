"""CPU test for scripts/diff_run_config.py -- no real checkpoints needed.

Builds throwaway checkpoints with the same {"step", "config"} shape that
transport/train.py:209-212 saves, then runs the script as a subprocess and
asserts on stdout and the exit status. This covers what `--selftest` cannot:
the value table, the ALWAYS_SHOW block, all three sidecar branches, and the
exit-code contract (1 = a real config difference, 0 = none).

    python tests/test_diff_run_config.py
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "diff_run_config.py"
WORK = ROOT / "tmp" / "diffcfg"

BASE = {"steps": 5000, "lr": 1e-4, "grad_clip": 100000.0, "grad_accum": 8,
        "batch_size": 2, "seed": 0, "val_every": 1000, "out_dir": "runs/a",
        "gate": "b", "delta_start": 0.0, "delta_end": 0.5,
        "delta_warmup_frac": 0.5}


def write(tag, config, args=None):
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"step": config.get("steps", 5000), "config": config,
                "adapter": {}}, d / "ckpt.pt")
    if args is not None:
        (d / "train_args.json").write_text(json.dumps(args, indent=1))
    else:
        (d / "train_args.json").unlink(missing_ok=True)
    return str(d / "ckpt.pt")


def run(pa, pb, timeout=120):
    """Never hang. A hung child is exactly how this test pinned 12 processes on
    the box for up to 14 minutes (a `mmap=True` load that stalls under torch 2.8
    while passing under 2.12 locally). A timeout turns that into a loud failure
    instead of a mystery that costs an hour of idle GPU.
    """
    try:
        r = subprocess.run([sys.executable, str(SCRIPT), pa, pb],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s -- child hung"
    return r.returncode, r.stdout + r.stderr


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}"
          + (f"  -- {detail}" if detail and not cond else ""))
    return bool(cond)


ok = True
# --- case 1: a real difference, no sidecars ------------------------------
b = dict(BASE, seed=1, batch_size=16, grad_accum=1, val_every=500,
         out_dir="runs/b")
pa, pb = write("a", BASE), write("b", b)
rc, out = run(pa, pb)
ok &= check("case1 exit 1 on a real difference", rc == 1, f"rc={rc}")
ok &= check("case1 names the differing keys",
            all(k in out for k in ("seed", "batch_size", "grad_accum")))
ok &= check("case1 ALWAYS_SHOW prints values even when equal",
            "values  A | B" in out and "grad_clip" in out
            and "100000.0" in out)
ok &= check("case1 out_dir demoted to 'expected'",
            "expected (run dir / hardware" in out)
ok &= check("case1 sidecar-absent branch fires twice",
            out.count("predates the provenance patch") == 2)

# --- case 2: identical configs -> exit 0 ---------------------------------
pa2 = write("a2", BASE)
rc, out = run(pa2, pa2)
ok &= check("case2 exit 0 when nothing differs", rc == 0, f"rc={rc}")
ok &= check("case2 says so", "no differing keys" in out)

# --- case 3: both sidecars present --------------------------------------
side_a = {"loss_path": "ref", "delta_cap_effective": "static hard max 0.5",
          "curriculum": False, "max_delta": 0.5, "anchor_mode": "exact"}
side_b = dict(side_a, anchor_mode="small_delta", max_delta=0.75,
              delta_cap_effective="static hard max 0.75")
pa3 = write("a3", BASE, side_a)
pb3 = write("b3", dict(BASE, seed=1), side_b)
rc, out = run(pa3, pb3)
ok &= check("case3 exit 1 with sidecars", rc == 1, f"rc={rc}")
ok &= check("case3 prints the cap keys",
            "delta_cap_effective=" in out and "anchor_mode=" in out)
ok &= check("case3 diffs the CLI (anchor_mode + max_delta)",
            "gate_b CLI" in out and "small_delta" in out and "0.75" in out)

# --- case 4: exactly one sidecar -> cannot diff, must say so -------------
pa4 = write("a4", BASE, side_a)
pb4 = write("b4", dict(BASE, seed=1))
rc, out = run(pa4, pb4)
ok &= check("case4 warns instead of pretending", 
            "only one side has a sidecar" in out)
ok &= check("case4 still exits 1 on the config difference", rc == 1,
            f"rc={rc}")

# --- case 5: the built-in selftest --------------------------------------
try:
    r = subprocess.run([sys.executable, str(SCRIPT), "--selftest"],
                       capture_output=True, text=True, cwd=str(ROOT),
                       timeout=120)
except subprocess.TimeoutExpired:
    r = None
ok &= check("case5 --selftest passes",
            r is not None and r.returncode == 0 and "selftest ok" in r.stdout)

print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
sys.exit(0 if ok else 1)
