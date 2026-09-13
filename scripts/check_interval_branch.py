"""Is the interval branch learning, or random-walking? CPU only, seconds.

    python scripts/check_interval_branch.py runs/gate_b_p10/ckpt_*.pt \\
        runs/centered/ckpt_*.pt --cache /data/cache/<hash>

For each checkpoint it reports, for every zero-initialised output layer of the
interval embedding:

    rms          RMS of the weights = total displacement from init
    noise        what Adam's random walk would give over the actual lr
                 schedule if every gradient were direction-free:
                 sqrt(sum_i lr_i^2)
    full         what a perfectly consistent gradient would give: sum_i lr_i
    rms/noise    ~1.0 means the layer has only random-walked. The concat
                 embedding measured 1.04 at step 200 and 1.75 at step 1000.
    drift%       sqrt(rms^2 - noise^2) / full: rough fraction of the maximum
                 possible consistent movement. Assumes iid gradient noise.

and, with --cache, the size of what the embedding writes into pooled_projections,
as a fraction of ||pooled|| itself:

    r-signal     || d_pool(t, Delta) - d_pool(t, 0) ||   (the part that can
                 carry interval information)
    offset       || d_pool(t, 0) ||                       (the part that cannot)

Concat and centered checkpoints are handled alike; the kind is read from the
parameter names. This says whether the r-path MOVED. Whether it moved the
RIGHT way is eval_interval_field.py's cos_correction, and needs the GPU.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.conditioning import (  # noqa: E402
    interval_kind_from_keys, make_interval_embedding,
)

READOUTS = {
    "concat": {"net.4": "single MLP (t and Delta mixed)"},
    "centered": {"delta_branch.4": "f: Delta branch",
                 "t_branch.4": "g: t branch"},
}
DELTAS = (0.05, 0.1, 0.2, 0.4, 0.8)


def lr_at(step: int, c: dict) -> float:
    """Mirror of transport/train.py Trainer._lr."""
    if step < c["warmup"]:
        return c["lr"] * (step + 1) / c["warmup"]
    p = (step - c["warmup"]) / max(1, c["steps"] - c["warmup"])
    return c["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--cache", default=None,
                    help="TripletCache dir; enables the r-signal/offset "
                         "columns relative to ||pooled||")
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()

    pooled_norm = None
    if args.cache:
        from transport.data import TripletCache
        cache = TripletCache(args.cache)
        n = min(args.samples, len(cache))
        pooled_norm = float(torch.stack([
            torch.from_numpy(cache[i]["pooled"]).float().norm()
            for i in range(n)]).mean())
        print(f"mean ||pooled|| over {n} cached samples: {pooled_norm:.3f}\n")

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c, step = ck["config"], ck["step"]
        kind = interval_kind_from_keys(ck["adapter"].keys())
        if kind is None:
            print(f"{path}: no interval weights, skipped")
            continue
        sd = {k.split("interval.", 1)[1]: v.float()
              for k, v in ck["adapter"].items() if k.startswith("interval.")}
        lrs = [lr_at(s, c) for s in range(step)]
        noise = math.sqrt(sum(x * x for x in lrs))
        full = sum(lrs)

        print(f"{path}  step {step}  kind {kind}  (noise {noise:.2e}, "
              f"full {full:.2e})")
        for prefix, label in READOUTS[kind].items():
            w = sd[f"{prefix}.weight"]
            rms = w.pow(2).mean().sqrt().item()
            drift = math.sqrt(max(rms ** 2 - noise ** 2, 0.0))
            print(f"    {label:<32} rms {rms:.2e}  rms/noise "
                  f"{rms / max(noise, 1e-30):5.2f}  drift {100 * drift / full:5.1f}%")

        if pooled_norm is not None:
            emb = make_interval_embedding(kind, pooled_dim=sd[
                next(k for k in sd if k.endswith("4.weight"))].shape[0])
            emb.load_state_dict(sd)
            with torch.no_grad():
                for t in (0.5, 0.9):
                    d = torch.tensor([x for x in DELTAS if x <= t])
                    tt = torch.full_like(d, t)
                    base = emb(tt, tt.clone())
                    sig = (emb(tt, tt - d) - base).norm(dim=1)
                    cells = "  ".join(f"d={x:<4} {100 * s / pooled_norm:6.3f}%"
                                      for x, s in zip(d.tolist(), sig.tolist()))
                    print(f"    t={t}: offset {100 * base[0].norm() / pooled_norm:6.3f}%"
                          f"   r-signal  {cells}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
