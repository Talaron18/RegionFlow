"""Is the overshoot one scalar, or is it structured? GPU, ~1 min per cell.

    python scripts/probe_correction_scale.py --cache /data/cache/<hash> \\
        --ckpt runs/fd_frozen_p75_5k/ckpt_final.pt

WHY. Across a 5x2 (t_center, Delta) sweep the step-5000 model's D_r ratio was
above 1 at EVERY point (mean 1.14 at Delta=0.4, 1.25 at Delta=0.5) while
cos_correction declined smoothly with t (0.76 at t=0.65 to 0.68 at t=0.80)
with no feature at any Delta. So direction is well behaved and the magnitude
is systematically too large. Neither t nor Delta alone explained the size of
the overshoot: two (t_center, Delta) pairs landing on the SAME t disagreed
(1.279 vs 1.321 at t=0.700), and each Delta's ratio ranged widely across
t_centers.

This fits the scale directly. For the learned correction
c = u_hat(r) - u_hat(r=t) and the true one c* = u_true - v, the least-squares
scalar per sample is

    a = <c, c*> / <c, c>      (a < 1 means the model overshoots)

and applying it changes the correction error from

    ||c - c*||^2 / ||c*||^2   to   1 - cos^2,

which is the corr_err_best column eval_interval_field.py already reports. So
this probe does not invent a new metric; it reports WHICH scalar gets there
and whether one scalar does it everywhere.

COLUMNS, per (t_center, Delta) cell
    a_opt      mean per-sample optimal scale (its own best value per sample)
    a_pool     one scalar for the whole cell: sum<c,c*> / sum<c,c>, which is
               what an inference-time calibration could actually apply
    a_spread   std of per-sample a, over the mean: how much the right scale
               varies BETWEEN samples in one cell. Large means no fixed
               scalar helps much even within a cell.
    cos        as in the eval
    err        corr_err now
    err_pool   corr_err after applying a_pool to every sample in the cell
    err_best   1 - cos^2, the floor at this direction

and finally one GLOBAL scalar fitted across every cell, with each cell's
error under it. The comparison that matters:

    err_pool (per cell) vs err_global (one number everywhere)

READING (pre-registered, before any result)
  - err_global close to err_pool everywhere, a_pool varying little across
    cells: the overshoot is ONE number. A fixed inference-time scale is an
    honest post-hoc calibration, reportable as such, and the residual error
    is all direction.
  - a_pool varying strongly and systematically with t or Delta: the overshoot
    is structured, no single scalar fixes it, and the magnitude error has to
    be addressed in training (loss weighting, or the Delta parameterisation).
  - a_spread large (say > 0.4) even within a cell: the right scale differs
    per IMAGE, so no (t, Delta)-indexed calibration helps either; that would
    point at the correction's magnitude being sample-dependent in a way the
    conditioning cannot express.

Samples, noise seed and integrator match eval_interval_field.py exactly, so
D_r_true and cos reproduce that script's numbers cell for cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_interval_field import true_average_velocity  # noqa: E402
from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.conditioning import (  # noqa: E402
    interval_kind_from_keys, make_interval_embedding,
)
from transport.data import TripletCache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--t-centers", type=float, nargs="+",
                    default=[0.45, 0.5, 0.55])
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--substeps", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--out", default="runs/correction_scale.json")
    args = ap.parse_args()

    from diffusers import SD3Transformer2DModel

    cache = TripletCache(args.cache)
    n = min(args.samples, len(cache))
    idx = np.random.default_rng(0).choice(len(cache), size=n, replace=False)
    x0 = torch.stack([torch.from_numpy(cache[int(j)]["z_tgt"]) for j in idx])
    pe = torch.stack([torch.from_numpy(cache[int(j)]["prompt_embeds"])
                      for j in idx])
    po = torch.stack([torch.from_numpy(cache[int(j)]["pooled"]) for j in idx])

    dt = getattr(torch, args.dtype)
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dt
    ).to(args.device).eval().requires_grad_(False)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank)).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    adapter.eval()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    kind = interval_kind_from_keys(sd["adapter"].keys())
    if kind is not None and kind != adapter.cfg.interval_kind:
        adapter.interval = make_interval_embedding(
            kind, int(tr.config.pooled_projection_dim)).to(args.device)
        adapter.cfg.interval_kind = kind
    missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys {unexpected[:5]}")
    lost = [k for k in missing if k.startswith("interval.")]
    if lost:
        raise RuntimeError(f"interval weights absent: {lost[:4]}")

    x0 = x0.to(args.device).float()
    pe = pe.to(args.device, adapter.model_dtype)
    po = po.to(args.device, adapter.model_dtype)
    g = torch.Generator(device=args.device).manual_seed(1234)
    eps = torch.randn(x0.shape, device=args.device, generator=g)

    print(f"[scale] {args.ckpt} step {sd.get('step')}  {n} samples  "
          f"{len(args.t_centers)}x{len(args.deltas)} cells")
    print(f"{'t_c':>6}{'delta':>7}{'t':>7}{'a_opt':>8}{'a_pool':>8}"
          f"{'a_spread':>10}{'cos':>7}{'err':>8}{'err_pool':>10}"
          f"{'err_best':>10}")

    cells, num_all, den_all = [], 0.0, 0.0
    for tc in args.t_centers:
        for d in args.deltas:
            t = torch.full((n,), tc + d / 2, device=args.device)
            r = torch.full((n,), tc - d / 2, device=args.device)
            tt = t.reshape(-1, 1, 1, 1)
            z_t = (1 - tt) * x0 + tt * eps

            def base_fn(z_, s_):
                return adapter.base_velocity(z_, s_, pe, po)

            u_true = true_average_velocity(base_fn, z_t, t, r, args.substeps)
            with torch.no_grad():
                u_hat = adapter(z_t, t, r, pe, po)
                v_inst = adapter.base_velocity(z_t, t, pe, po)
                u_hat_r0 = adapter(z_t, t, t.clone(), pe, po)

            c = (u_hat - u_hat_r0).flatten(1).double()
            ct = (u_true - v_inst).flatten(1).double()
            num = (c * ct).sum(1)
            den = c.pow(2).sum(1).clamp(min=1e-24)
            nt2 = ct.pow(2).sum(1).clamp(min=1e-24)

            a = num / den                       # per-sample optimal scale
            a_pool = float(num.sum() / den.sum())
            cos = torch.nn.functional.cosine_similarity(c, ct, dim=1)
            err = float(((c - ct).pow(2).sum(1) / nt2).mean())
            err_pool = float(((a_pool * c - ct).pow(2).sum(1) / nt2).mean())
            err_best = float((1 - cos ** 2).mean())
            spread = float(a.std() / a.mean().abs().clamp(min=1e-12))

            num_all += float(num.sum())
            den_all += float(den.sum())
            cells.append({
                "t_center": tc, "delta": d, "t": tc + d / 2,
                "a_opt": float(a.mean()), "a_pool": a_pool,
                "a_spread": spread, "cos": float(cos.mean()),
                "err": err, "err_pool": err_pool, "err_best": err_best,
                "_num": float(num.sum()), "_den": float(den.sum()),
                "_c": c.cpu(), "_ct": ct.cpu(),
            })
            k = cells[-1]
            print(f"{tc:>6}{d:>7}{tc + d / 2:>7.3f}{k['a_opt']:>8.3f}"
                  f"{a_pool:>8.3f}{spread:>10.3f}{k['cos']:>7.3f}"
                  f"{err:>8.3f}{err_pool:>10.3f}{err_best:>10.3f}")

    a_global = num_all / den_all
    print(f"\nglobal pooled scale a = {a_global:.3f}")
    print(f"{'t_c':>6}{'delta':>7}{'err':>8}{'err_global':>12}"
          f"{'err_pool':>10}{'err_best':>10}")
    rows = []
    for k in cells:
        c, ct = k.pop("_c"), k.pop("_ct")
        nt2 = ct.pow(2).sum(1).clamp(min=1e-24)
        k["err_global"] = float(((a_global * c - ct).pow(2).sum(1)
                                 / nt2).mean())
        print(f"{k['t_center']:>6}{k['delta']:>7}{k['err']:>8.3f}"
              f"{k['err_global']:>12.3f}{k['err_pool']:>10.3f}"
              f"{k['err_best']:>10.3f}")
        rows.append(k)

    ap_arr = np.array([k["a_pool"] for k in rows])
    print(f"\na_pool across cells: mean {ap_arr.mean():.3f}  "
          f"min {ap_arr.min():.3f}  max {ap_arr.max():.3f}  "
          f"std {ap_arr.std():.3f}")
    print(f"mean err {np.mean([k['err'] for k in rows]):.3f}  "
          f"-> global {np.mean([k['err_global'] for k in rows]):.3f}  "
          f"-> per-cell {np.mean([k['err_pool'] for k in rows]):.3f}  "
          f"(floor {np.mean([k['err_best'] for k in rows]):.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "ckpt_step": sd.get("step"),
         "a_global": a_global, "cells": rows}, indent=1, default=str))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
