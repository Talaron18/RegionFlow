"""Round-trip on a CACHED sample: use the instruction the model was
actually trained on, from the cache itself. This removes the prompt-OOD
confound.

Usage:
    python scripts/invert_one_cached.py \
        --ckpt runs/gate_b/ckpt_final.pt \
        --cache /data/cache/13061d51aee4 \
        --idx 0 \
        --out runs/inversion_cached
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


def _ensure(t, nd):
    while t.dim() < nd:
        t = t.unsqueeze(0)
    return t


@torch.no_grad()
def invert(adapter, z_data, pe, pooled, steps, device):
    z = z_data.clone()
    b = z.shape[0]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t_val = float(ts[i]); r_val = float(ts[i + 1])
        s = torch.full((b,), t_val, device=z.device)
        u = adapter(z, s, s, pe, pooled)
        z = z + (r_val - t_val) * u
    return z


@torch.no_grad()
def generate(adapter, z_noise, pe, pooled, steps, device):
    z = z_noise.clone()
    b = z.shape[0]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_val = float(ts[i]); r_val = float(ts[i + 1])
        s = torch.full((b,), t_val, device=z.device)
        r = torch.full((b,), r_val, device=z.device)
        u = adapter(z, s, r, pe, pooled)
        z = z - (t_val - r_val) * u
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--which", default="z_src",
                    choices=["z_src", "z_tgt"],
                    help="z_tgt is IN-distribution for Gate B; z_src is not")
    ap.add_argument("--interval-kind", default="centered",
                    choices=("concat", "centered"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
    from bench.inversion import SD3EditPipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer",
        torch_dtype=torch.float32).to(args.device)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank,
                          interval_kind=args.interval_kind)
    ).to(args.device)
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _m, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")
    adapter.eval()

    cache = TripletCache(args.cache)
    rec = cache[args.idx]
    meta = rec["meta"]
    print(f"idx {args.idx}: {meta['id']}  "
          f"instruction={meta['instruction']!r}")

    z_src = _ensure(torch.from_numpy(rec["z_src"]).float(), 4).to(args.device)
    z_tgt = _ensure(torch.from_numpy(rec["z_tgt"]).float(), 4).to(args.device)
    pe = _ensure(torch.from_numpy(rec["prompt_embeds"]), 3).to(args.device)
    pooled = _ensure(torch.from_numpy(rec["pooled"]), 2).to(args.device)

    z_ref = z_tgt if args.which == "z_tgt" else z_src
    print(f"round-trip target: {args.which}")

    z_noise = invert(adapter, z_ref, pe, pooled, args.steps, args.device)
    z_recon = generate(adapter, z_noise, pe, pooled, args.steps, args.device)

    def _n(x): return float(x.pow(2).mean().sqrt())
    def _cos(a, b):
        na, nb = a.norm(), b.norm()
        return float((a * b).sum() / (na * nb + 1e-8))

    print(f"\n||z_ref||    = {_n(z_ref):.4f}")
    print(f"||z_noise||  = {_n(z_noise):.4f}")
    print(f"||z_recon||  = {_n(z_recon):.4f}")
    print(f"\ncos(z_recon, z_ref) = {_cos(z_recon, z_ref):+.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_ref = helper.decode_latent(z_ref)
    img_recon = helper.decode_latent(z_recon)
    img_noise = helper.decode_latent(z_noise)
    ref_arr = np.asarray(Image.fromarray(img_ref).resize((512, 512)))
    rec_arr = np.asarray(Image.fromarray(img_recon).resize((512, 512)))
    nz_arr  = np.asarray(Image.fromarray(img_noise).resize((512, 512)))
    panel = np.concatenate([ref_arr, rec_arr, nz_arr], axis=1)
    Image.fromarray(panel).save(out / "roundtrip.png")
    print(f"saved {out / 'roundtrip.png'}  ({args.which} | recon | noise)")


if __name__ == "__main__":
    main()
