"""Does the interval embedding ripple with Delta? CPU only, minutes.

    python scripts/probe_delta_ripple.py runs/fd_frozen_p75_5k/ckpt_final.pt

WHY. The cap probe found the step-5000 model's D_r ratio oscillating across
Delta -- 0.84, 1.28, 0.98, 1.22, 0.86, 0.66 at Delta 0.3 to 0.6 -- while
cos_correction declined smoothly (0.73 -> 0.62) and every corr_err spike was
accounted for by that Delta's magnitude. So the learned correction points a
consistent direction and its SCALE wobbles by ~30% between Deltas 0.05 apart.

The suspect is the input basis. `sinusoidal` gives the Delta half 128
components; 68 of them complete more than one cycle over Delta in [0, 0.8],
so most of the input's Delta variation is faster than anything the true
average velocity does. An aliased basis rippling in amplitude while its
direction stays smooth is this signature.

This probe tests that WITHOUT the model: it measures ||d_pool(t, Delta)|| on
a dense Delta grid straight from the checkpoint's embedding weights, removes
a smooth trend, and reports the residual ripple and its dominant period. If
the embedding is smooth in Delta, the ripple is small and the mechanism is
somewhere else (the LoRA, or the eval itself).

COLUMNS
    ripple%      RMS of the detrended amplitude, as a % of the mean amplitude
    period       Delta-period of the largest detrended Fourier component
    ripple%_in   same, restricted to the TRAINED range (Delta <= --max-delta)

A same-norm random readout is measured alongside: it shows how much ripple
this basis produces with NO training, which is the null this must clear.

READING (pre-registered)
  - ripple >= ~10% with a period near 0.05-0.15: consistent with the eval's
    oscillation; P5 (low-passing the Delta basis) is the indicated fix and
    predicts both disappearing.
  - ripple a few % and smooth: the embedding is not the source. Look at the
    LoRA's Delta response or at eval sampling noise -- run the dense eval
    with a second seed before concluding anything.
  - trained ripple ~= random-readout ripple: the ripple is the basis's own
    geometry, present from init, not something training produced. Still
    consistent with P5, but it means the run never escaped it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transport.conditioning import (  # noqa: E402
    interval_kind_from_keys, make_interval_embedding,
)


def ripple(delta: np.ndarray, amp: np.ndarray, trend_order: int = 3):
    """Detrend with a low-order polynomial, return (ripple%, dominant period)."""
    coef = np.polyfit(delta, amp, trend_order)
    resid = amp - np.polyval(coef, delta)
    pct = 100 * resid.std() / max(amp.mean(), 1e-12)
    # rfft on the residual; grid is uniform, so bin k is k cycles over the span
    spec = np.abs(np.fft.rfft(resid - resid.mean()))
    k = int(spec[1:].argmax()) + 1
    span = delta[-1] - delta[0]
    return pct, span / k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--t", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--max-delta", type=float, default=0.5,
                    help="trained range, for the ripple%_in column")
    ap.add_argument("--grid", type=int, default=241,
                    help="Delta samples from 0 to --delta-max-plot")
    ap.add_argument("--delta-max-plot", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=5,
                    help="random-readout controls to average")
    args = ap.parse_args()

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        kind = interval_kind_from_keys(ck["adapter"].keys())
        if kind != "centered":
            print(f"{path}: interval kind {kind}, this probe expects "
                  "'centered' -- skipped")
            continue
        sd = {k.split("interval.", 1)[1]: v.float()
              for k, v in ck["adapter"].items() if k.startswith("interval.")}
        pooled_dim = sd["t_branch.4.weight"].shape[0]
        emb = make_interval_embedding("centered", pooled_dim)
        emb.load_state_dict(sd)

        d = torch.linspace(0, args.delta_max_plot, args.grid)
        dn = d.numpy()
        inside = dn <= args.max_delta + 1e-9
        print(f"\n{path}  step {ck['step']}  grid {args.grid} points "
              f"over Delta [0, {args.delta_max_plot}]")

        W = sd["delta_branch.4.weight"]
        for t in args.t:
            tt = torch.full_like(d, t)
            with torch.no_grad():
                amp = emb.delta_part(tt, tt - d).norm(dim=1).numpy()
            pct, per = ripple(dn, amp)
            pct_in, per_in = ripple(dn[inside], amp[inside])

            rnd = []
            for seed in range(args.seeds):
                g = torch.Generator().manual_seed(seed)
                with torch.no_grad():
                    emb.delta_branch[4].weight.copy_(
                        torch.randn(W.shape, generator=g) * W.std())
                    a_r = emb.delta_part(tt, tt - d).norm(dim=1).numpy()
                rnd.append(ripple(dn, a_r)[0])
            with torch.no_grad():
                emb.delta_branch[4].weight.copy_(W)

            print(f"  t={t:<4} ripple {pct:5.1f}%  period {per:.3f}   "
                  f"(trained range: {pct_in:5.1f}%  period {per_in:.3f})   "
                  f"random readout {np.mean(rnd):5.1f}%")

        # amplitude at the Deltas the cap probe measured, for direct comparison
        probe_d = torch.tensor([0.3, 0.4, 0.45, 0.5, 0.55, 0.6])
        tt = torch.full_like(probe_d, 0.5)
        with torch.no_grad():
            a = emb.delta_part(tt, tt - probe_d).norm(dim=1)
        sm = np.polyval(np.polyfit(probe_d.numpy(), a.numpy(), 1),
                        probe_d.numpy())
        print("  at the cap-probe Deltas (t=0.5), amplitude vs a linear fit:")
        print("    " + "  ".join(f"d={x:g}:{100 * (y / s - 1):+5.1f}%"
                                 for x, y, s in zip(probe_d.tolist(),
                                                    a.tolist(), sm)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
