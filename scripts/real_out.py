"""Extract u, v, mask for a real-cache index and save to npz, so the
same decomposition scripts (synthetic_inside_outside.py,
synthetic_spectrum.py) can run on it unchanged.

Usage:
    python scripts/extract_real_u_v.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --idx 0 \
        --out runs/real_u_v_idx0.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


def _ensure_batch(t, ndim):
    while t.dim() < ndim:
        t = t.unsqueeze(0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from diffusers import SD3Transformer2DModel
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(args.device)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="pair", rank=args.rank,
                          mask_encoder_kind=args.mask_encoder_kind)
    ).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _m, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")
    adapter.eval()
    print(f"loaded {args.ckpt}  step {sd.get('step', '?')}")

    cache = TripletCache(args.cache)
    rec = cache[args.idx]
    print(f"idx {args.idx}: {rec['meta'].get('id')}  "
          f"instr={rec['meta'].get('instruction')!r}  "
          f"mask_frac={rec['meta'].get('mask_frac')}")

    z_src = _ensure_batch(torch.from_numpy(rec["z_src"]).float(), 4).to(args.device)
    z_tgt = _ensure_batch(torch.from_numpy(rec["z_tgt"]).float(), 4).to(args.device)
    mask = _ensure_batch(torch.from_numpy(rec["mask"]).float(), 4).to(args.device)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    pe = _ensure_batch(torch.from_numpy(rec["prompt_embeds"]), 3).to(args.device)
    pooled = _ensure_batch(torch.from_numpy(rec["pooled"]), 2).to(args.device)

    with torch.no_grad():
        s0 = torch.zeros(z_src.shape[0], device=args.device)
        u = adapter(z_src, s0, s0, pe, pooled, mask=mask)
        v = z_tgt - z_src

    np.savez(
        args.out,
        u=u[0].cpu().numpy(),
        v=v[0].cpu().numpy(),
        mask=mask[0, 0].cpu().numpy().astype(bool),
        z_src=z_src[0].cpu().numpy(),
        z_tgt=z_tgt[0].cpu().numpy(),
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()