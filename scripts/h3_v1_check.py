"""H3 V1, part 1 -- the compatibility contract on REAL checkpoints (one GPU
load, minutes). H3_SIGNED_DELTA_DESIGN.md Sec.3 / Sec.6 V1 / risk Sec.7.3.

For each checkpoint, on the same inputs (n rows of (z, t, r) with r <= t, 25%
exact anchors, real prompt embeddings from the cache):

  CONTROL   unsigned vs unsigned, twice.   Must be torch.equal. If not, the
            box is nondeterministic and the next line cannot be read; the
            script says so and exits 2 rather than reporting a false failure.
  CONTRACT  signed vs unsigned, r <= t.    Must be torch.equal (not allclose).
            The unsigned path IS the pre-H3 expression (clamp(min=0)), so this
            is the pre-H3 vs post-H3 comparison without a second checkout.
  BEFORE    unsigned, r > t  vs  r == t.   Must be torch.equal: today's
            checkpoints read the reverse direction as the anchor.
  AFTER     signed,   r > t  vs  r == t.   Reported (relative difference).
            Not a pass/fail: an untrained reverse side is extrapolation.

Exit 0 only if CONTROL, CONTRACT and BEFORE all hold for every checkpoint.

    PYTORCH_JIT=0 python scripts/h3_v1_check.py --cache /data/cache/<id> \\
        --ckpt runs/A0/ckpt_final.pt runs/A2/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.conditioning import (  # noqa: E402
    interval_kind_from_keys, make_interval_embedding)
from transport.data import TripletCache  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from diffusers import SD3Transformer2DModel

    cache = TripletCache(args.cache)
    idx = [i % len(cache) for i in range(args.n)]
    x0 = torch.stack([torch.from_numpy(cache[i]["z_tgt"]) for i in idx])
    pe = torch.stack([torch.from_numpy(cache[i]["prompt_embeds"]) for i in idx])
    po = torch.stack([torch.from_numpy(cache[i]["pooled"]) for i in idx])

    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer",
        torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    tr.requires_grad_(False)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank)).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    adapter.eval()

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    a, b = torch.rand(args.n, generator=g), torch.rand(args.n, generator=g)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    anc = torch.rand(args.n, generator=g) < 0.25
    r = torch.where(anc, t, r)
    eps = torch.randn(x0.shape, generator=g)
    tt = t.reshape(-1, 1, 1, 1)
    z = ((1 - tt) * x0.float() + tt * eps)

    dev, mdt = args.device, adapter.model_dtype

    def run(t_, r_, signed):
        adapter.interval.signed = signed
        outs = []
        with torch.no_grad():
            for s in range(0, args.n, args.chunk):
                sl = slice(s, s + args.chunk)
                outs.append(adapter(z[sl].to(dev), t_[sl].to(dev), r_[sl].to(dev),
                                    pe[sl].to(dev, mdt), po[sl].to(dev, mdt)).cpu())
        return torch.cat(outs)

    ok = True
    for ck in args.ckpt:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        kind = interval_kind_from_keys(sd["adapter"].keys())
        if kind is not None and kind != adapter.cfg.interval_kind:
            adapter.interval = make_interval_embedding(
                kind, int(tr.config.pooled_projection_dim)).to(dev)
            adapter.cfg.interval_kind = kind
        missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
        if unexpected or [k for k in missing if k.startswith("interval.")]:
            raise RuntimeError(f"{ck}: bad keys {unexpected[:3]} / missing interval")
        print(f"\n[{ck}] step {sd.get('step')} interval={adapter.cfg.interval_kind}")

        u0 = run(t, r, False)
        u0b = run(t, r, False)
        control = torch.equal(u0, u0b)
        print(f"  CONTROL  unsigned twice equal: {control}")
        if not control:
            print("  nondeterministic forward on this box -- CONTRACT unreadable. "
                  f"max|diff| {float((u0 - u0b).abs().max()):.3e}")
            return 2
        contract = torch.equal(u0, run(t, r, True))
        print(f"  CONTRACT signed == unsigned on r<=t (n={args.n}, "
              f"{int(anc.sum())} anchors): {contract}")

        # reverse queries: swap the non-anchor rows
        tr_, rr_ = torch.where(anc, t, r), torch.where(anc, r, t)
        fwd_rows = ~anc
        u_anchor_uns = run(tr_, tr_.clone(), False)
        before = torch.equal(run(tr_, rr_, False)[fwd_rows], u_anchor_uns[fwd_rows])
        print(f"  BEFORE   unsigned r>t == r==t: {before}")
        u_rev = run(tr_, rr_, True)
        u_anc = run(tr_, tr_.clone(), True)
        rel = ((u_rev - u_anc)[fwd_rows].flatten(1).norm(dim=1)
               / u_anc[fwd_rows].flatten(1).norm(dim=1).clamp(min=1e-12))
        print(f"  AFTER    signed r>t vs r==t: rel diff median {float(rel.median()):.3%}"
              f" max {float(rel.max()):.3%} (untrained side: extrapolation)")
        ok = ok and contract and before

    print("\nV1 compatibility:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
