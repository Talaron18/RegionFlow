"""
Synthetic controls for the decoupling metric. Run before any GPU time.

    python scripts/test_decoupling_metric.py --image "visual compare/source.png"

Two constructed edits with known ground truth:

  DESTROY    blend the masked region toward noise. Structure genuinely dies.
             The metric MUST say CONFIRMED (Stage 0 coupling reproduced).
  RECOLOUR   shift chroma in L*a*b* with L* untouched. Structure is preserved
             by construction. The metric MUST say REFUTED.

WHY THIS FILE EXISTS. The first version of the Stage 1 metric used
corr(dE, struct) with a threshold at |corr| < 0.5, mirroring Stage 0's
headline. Run against RECOLOUR on a real photograph it returned corr = -0.92
and would have declared the coupling CONFIRMED -- i.e. it would have killed the
project on a model that was working perfectly. Pearson correlation saturates
near -1 for any monotone decline regardless of magnitude, so it cannot see
effect size, which is the entire question.

The fix was to make struct-at-fixed-dE the primary statistic. That fix was
found by running this control, on the CPU, in about a minute. Stage 0's own
lesson was that automated verdicts were wrong every time they fired; a verdict
function is itself an instrument and needs its own calibration.

This also quantifies a second confound worth reporting in the paper: part of
any struct decline is GAMUT CLIPPING rather than model behaviour. On the test
photograph, a structure-preserving recolour clips ~33% of in-mask pixels at
dE ~ 74, and clipping compresses local contrast. The in-mask clipped fraction
is printed alongside so the two causes can be separated.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_metrics_without_torch():
    """Exec only the metric functions out of tau_sweep.py.

    tau_sweep imports torch and diffusers at module level, which this control
    does not need. Extracting the function bodies keeps the control runnable on
    a laptop and, more importantly, keeps it running the SAME code -- a
    reimplementation here would defeat the purpose.
    """
    path = Path(__file__).resolve().parent / "tau_sweep.py"
    tree = ast.parse(path.read_text())
    ns = {"np": np}
    want = {"to_lab", "lab_deltas", "struct_corr", "luma"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    missing = want - set(ns)
    if missing:
        raise RuntimeError(f"could not extract {missing} from tau_sweep.py")
    return ns


def lab_to_rgb(lab: np.ndarray) -> tuple:
    """Inverse of tau_sweep.to_lab. Returns (uint8 image, clipped fraction)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29

    def f(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))

    x, y, z = f(fx) * 0.95047, f(fy), f(fz) * 1.08883
    lin = np.stack([
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ], axis=-1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.maximum(lin, 0) ** (1 / 2.4) - 0.055) * 255
    clipped = (srgb < 0) | (srgb > 255)
    return np.clip(srgb, 0, 255).astype(np.uint8), clipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="a real photograph; falls back to synthetic noise")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    args = ap.parse_args()

    ns = load_metrics_without_torch()
    import bench.decoupling as D
    D._METRICS = (ns["lab_deltas"], ns["struct_corr"])

    if args.image:
        from PIL import Image
        base = np.array(Image.open(args.image).convert("RGB"))
    else:
        rng = np.random.default_rng(0)
        base = (rng.random((256, 256, 3)) * 255).astype(np.uint8)

    h, w = base.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[int(0.45 * h):int(0.65 * h), int(0.10 * w):int(0.45 * w)] = True
    rng = np.random.default_rng(0)

    def destroy(a):
        ed = base.astype(float)
        n = rng.random(base.shape) * 255
        ed[mask] = ed[mask] * (1 - a / 1.5) + n[mask] * (a / 1.5)
        return np.clip(ed, 0, 255).astype(np.uint8), 0.0

    def recolour(a):
        lab = ns["to_lab"](base).copy()
        lab[mask, 1] += 45 * a
        lab[mask, 2] += 45 * a
        img, clipped = lab_to_rgb(lab)
        return img, float(clipped[mask].mean())

    failures = []
    for label, fn, expect in (("DESTROY", destroy, "CONFIRMED"),
                              ("RECOLOUR", recolour, "REFUTED")):
        pts, clips = {}, []
        for case in ("ball", "sheep", "phone_booth"):
            row = []
            for a in args.strengths:
                ed, cf = fn(a)
                clips.append(cf)
                row.append(D.evaluate_point(case, a, base, ed, mask))
            pts[case] = row
        rep = D.decoupling_report(pts)
        ok = rep["verdict"] == expect
        print(f"\n===== {label}  (expect {expect}) -> {rep['verdict']} "
              f"{'OK' if ok else 'MISMATCH'} =====")
        print(f"  mean in-mask clipped fraction: {100*np.mean(clips):.1f}%")
        print(D.format_report(rep).split("Pearson correlation")[0])
        if not ok:
            failures.append(f"{label}: expected {expect}, got {rep['verdict']}")

    if failures:
        print("\nCONTROL FAILED -- do not trust this metric on real runs:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nBoth controls passed. The metric separates structure-preserving "
          "from structure-destroying edits at matched dE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
