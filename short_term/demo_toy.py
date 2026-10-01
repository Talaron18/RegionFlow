#!/usr/bin/env python3
"""
The conceptual experiment, on CPU.

RUNTIME: measured at 52 s on the CPU-only dev box (torch import included); the
"about a second" this docstring used to claim was never true here, and a cold
first torch import can push it past two minutes, which is long enough to look
like a hang. It is deterministic, so nothing is lost by waiting.

WHAT IT SETTLES. "The distribution is multimodal, so add a latent and let the
model learn the multimodality." Three claims get separated here, on the
smallest object that can carry all three:

  1. A DETERMINISTIC one-step regression onto the target reproduces Gate A's
     failure exactly: its MSE optimum is the conditional MEAN, which for two
     equally likely modes is the midpoint -- "no visible change straight to a
     smear", in closed form.
  2. A NOISE-SIDE field with the same conditional mean still recovers the
     distribution, because the mode is carried by the initial noise, not by the
     regression head. No latent needs to be added; the noise IS the latent, and
     it is in the STATE, which is strictly stronger than in the conditioning.
  3. INFERENCE-TIME GUIDANCE selects the mode without any change to the
     conditioning, and does so LOCALLY: a mask on the left region pins the left
     attribute while the right region keeps sampling freely. That is tier D.

The control that makes (3) mean something is the weight=0 row: with no guidance
the two regions are independent draws, and the right region is not
incidentally determined by guiding the left.

    python short_term/demo_toy.py
    python short_term/demo_toy.py --draws 2000 --steps 20

HONEST SCOPE. This is an instrument for the mechanism, not evidence about a
real editor. The toy's regions are independent, its targets are piecewise
constant, and its field is exact rather than learned. It can tell you whether a
mechanism is even capable of doing what you want; it cannot tell you whether a
trained model will.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same partial-sync guard as edit.py, for the same reason: without
# __init__.py a directory is still importable as a namespace package, so a
# missing sibling shows up as a bare ModuleNotFoundError naming the wrong
# thing.
_SIBLINGS = ("__init__.py", "guidance.py", "sampler.py", "backends.py")
_missing = [n for n in _SIBLINGS
            if not (Path(__file__).resolve().parent / n).is_file()]
if _missing:
    _have = sorted(p.name for p in Path(__file__).resolve().parent.glob("*.py"))
    raise SystemExit(
        "short_term/ is incompletely synced (a copy problem, not a bug).\n"
        f"  directory : {Path(__file__).resolve().parent}\n"
        f"  missing   : {', '.join(_missing)}\n"
        f"  present   : {', '.join(_have) if _have else '(no .py files)'}\n"
        "Copy the whole short_term/ directory. Verify with:\n"
        "  python short_term/edit.py --doctor")

from short_term.backends import ToyMixtureField  # noqa: E402
from short_term.guidance import GuidanceTerm  # noqa: E402
from short_term.sampler import sample, sigma_grid  # noqa: E402


def _draw(field, z_src, sigmas, terms, pin, draws, steps, seed, kappa,
          init_scale=1.0):
    """Sample `draws` times, count the (left, right) mode combination.

    `sigmas` and `steps` are both the grid: `sigmas` is the tensor the sampler
    consumes, `steps` is only used for reporting, so the two are passed
    separately rather than recomputed.
    """
    torch.manual_seed(seed)
    counts = Counter()
    for _ in range(draws):
        z = torch.randn_like(z_src) * init_scale
        z_out, _ = sample(field, z, sigmas, terms,
                          direct_kappa=kappa, pin=pin)
        mode = field.decode(z_out)
        half = z_src.shape[-1] // 2
        left = float(mode[0, 0, :, :half].mean())
        right = float(mode[0, 0, :, half:].mean())
        counts[(round(left), round(right))] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-scale", type=float, default=1.0,
                    help="std of the initial latent. This is the noise-side "
                         "analogue of 'sampling strength': it is what carries "
                         "the mode.")
    args = ap.parse_args()

    g = args.grid
    dt = torch.float64
    field = ToyMixtureField.halves(g, g, dtype=dt)
    z_src = torch.zeros(1, 1, g, g, dtype=dt)
    sigmas = sigma_grid(args.steps)
    n = args.draws

    half = args.grid // 2
    left_mask = torch.zeros(1, 1, g, g, dtype=dt)
    left_mask[:, :, :, :half] = 1.0

    print("=" * 74)
    print("1. DETERMINISTIC REGRESSION -- the MSE optimum is the midpoint")
    print("=" * 74)
    torch.manual_seed(args.seed)
    targets = torch.where(torch.rand(n, 1, 1, 1, dtype=dt) < 0.5, 1.0, -1.0)
    targets = targets.expand(n, 1, g, g).clone()
    # The pairflow analogue: v = target - source, regressed with MSE from a
    # fixed source. The least-squares constant predictor IS the sample mean.
    v_target = targets - z_src
    mse_pred = v_target.mean(dim=0)
    print(f"  source z_src = 0.0 (the neutral image), targets are +/-1 per region")
    print(f"  least-squares v        = {float(mse_pred.mean()):+.4f}"
          f"   (targets mean {float(targets.mean()):+.4f})")
    print(f"  one-step readout       = {float((z_src + mse_pred).mean()):+.4f}"
          "   <- the midpoint of the two modes")
    print("  This is Gate A's failure reproduced in one line: the MSE optimum")
    print("  is the conditional mean, and the conditional mean of {-1,+1} is 0.")
    print("  Note what it is NOT evidence about: nothing here is multimodal")
    print("  'given the conditioning' in a way that a latent would fix. The")
    print("  information is missing because the SOURCE carries none of it.")

    print()
    print("=" * 74)
    print("2. UNGUIDED NOISE-SIDE SAMPLING -- the distribution is recovered")
    print("=" * 74)
    counts_u = _draw(field, z_src, sigmas, (), None, n, args.steps,
                     args.seed, None, args.init_scale)
    _report(counts_u, n, "unguided")

    print()
    print("=" * 74)
    print("3. GUIDANCE ON THE LEFT REGION ONLY -- mode selected, locally")
    print("=" * 74)
    term = GuidanceTerm(beta=5.0, weight=left_mask,
                        target=torch.ones_like(z_src), name="left:=+1")
    kappa = None
    counts_g = _draw(field, z_src, sigmas, (term,), None, n, args.steps,
                     args.seed, kappa, args.init_scale)
    _report(counts_g, n, "guided  (W = left half, Y = +1)")

    print()
    print("=" * 74)
    print("4. CONTROL: SAME TERM, WEIGHT ZERO")
    print("=" * 74)
    inert = GuidanceTerm(beta=5.0, weight=torch.zeros_like(left_mask),
                         target=torch.ones_like(z_src), name="inert")
    counts_i = _draw(field, z_src, sigmas, (inert,), None, n, args.steps,
                     args.seed, None, args.init_scale)
    _report(counts_i, n, "W = 0")

    print()
    print("=" * 74)
    print("5. CONTROL: HARD PIN OF THE RIGHT REGION (the Stage 0 idiom)")
    print("=" * 74)
    # pin=(z_pin, keep): the sampler does z <- keep*z + (1-keep)*z_pin, so
    # `keep` is 1 where the sample is FREE. keep = left_mask therefore pins the
    # RIGHT region to the source and leaves the left free.
    counts_p = _draw(field, z_src, sigmas, (term,), (z_src, left_mask), n,
                     args.steps, args.seed, None, args.init_scale)
    _report(counts_p, n, "guided left + pinned right to z_src")

    print()
    print("=" * 74)
    print("READING")
    print("=" * 74)
    print("  row 2  both regions ~50/50 and independent: the field carries the")
    print("         whole distribution. No latent was added to get this.")
    print("  row 3  the LEFT is always +1 while the RIGHT stays ~50/50. The")
    print("         conditioning never changed -- an ambiguous instruction was")
    print("         selected by a guidance signal plus a mask, which is the")
    print("         answer to 'can we avoid needing a very precise prompt'.")
    print("  row 4  weight 0 reproduces row 2, so row 3's left/right split is")
    print("         the guidance and not an artifact of the region structure.")
    print("  row 5  pinning freezes the right region at the wrong value (0),")
    print("         which is not a plausible sample -- the cost of hard pinning")
    print("         that guidance does not pay.")
    return 0


def _report(counts: Counter, n: int, label: str) -> None:
    total = sum(counts.values())
    print(f"  {label:<38} " + "  ".join(
        f"({k[0]:+d},{k[1]:+d}) {100 * v / max(total, 1):4.1f}%"
        for k, v in sorted(counts.items())))
    left_pos = sum(v for k, v in counts.items() if k[0] > 0) / max(total, 1)
    right_pos = sum(v for k, v in counts.items() if k[1] > 0) / max(total, 1)
    print(f"  {'':<38} P(left=+1) = {left_pos:.3f}   "
          f"P(right=+1) = {right_pos:.3f}   n={total}")


if __name__ == "__main__":
    raise SystemExit(main())
