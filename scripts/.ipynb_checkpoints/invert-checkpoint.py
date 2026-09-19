"""Invert Gate B: data latent -> noise -> data (round-trip).

Convention (confirmed from transport/sampling.py):
  ts = linspace(1, 0)   generation; 1 = noise, 0 = data
  t = ts[i], r = ts[i+1], t > r
  u = model(z, t, r, ...)      signature (z, t, r)
  z = z - (t - r) * u          direction: toward data
  u (mathematically) = (z_t - z_r)/(t - r), which points toward noise.

Inversion is the mirror: t = 0 -> 1, z = z + dt * u_instantaneous.
Query with r = t (instantaneous velocity), which is what 75% of Gate B's
training data saw (the anchors).

Usage:
    python scripts/invert_meanflow.py \
        --ckpt runs/gate_b/ckpt_final.pt \
        --src <a real image> \
        --prompt "a photo of a cat" \
        --out runs/inversion_check
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


def build_adapter(args):
    from diffusers import SD3Transformer2DModel
    tr = SD3Transformer2DModel.from_pretrained(
        args.model, subfolder="transformer",
        torch_dtype=torch.float32).to(args.device)
    ad = TransportAdapter(
        tr, AdapterConfig(mode="meanflow", rank=args.rank)
    ).to(args.device)
    for p in ad.trainable_parameters():
        p.data = p.data.float()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _m, unexpected = ad.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")
    ad.eval()
    print(f"loaded {args.ckpt} step {sd.get('step', '?')}")
    return ad


@torch.no_grad()
def invert_instant(adapter, z_data, pe, pooled, steps, device):
    """t = 0 (data) -> t = 1 (noise). Uses r = t (instantaneous velocity)."""
    z = z_data.clone()
    b = z.shape[0]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t_val = float(ts[i])
        r_val = float(ts[i + 1])
        s = torch.full((b,), t_val, device=z.device)
        # r = t: instantaneous velocity, the mode 75% of anchors trained on
        u = adapter(z, s, s, pe, pooled)
        z = z + (r_val - t_val) * u
    return z


@torch.no_grad()
def generate(adapter, z_noise, pe, pooled, steps, device):
    """t = 1 (noise) -> t = 0 (data). Matches sampling.py exactly."""
    z = z_noise.clone()
    b = z.shape[0]
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_val = float(ts[i])
        r_val = float(ts[i + 1])
        s = torch.full((b,), t_val, device=z.device)
        r = torch.full((b,), r_val, device=z.device)
        u = adapter(z, s, r, pe, pooled)
        z = z - (t_val - r_val) * u
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from diffusers import StableDiffusion3Pipeline
    from bench.inversion import SD3EditPipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)
    adapter = build_adapter(args)

    src = Image.open(args.src).convert("RGB").resize((512, 512))
    z_src = helper.encode_image(src).float()
    pe, _, pooled, _ = helper.encode_prompt(args.prompt)

    # ---- 1. data -> noise ----
    z_noise = invert_instant(adapter, z_src, pe, pooled, args.steps, args.device)

    # ---- 2. noise -> data (round-trip) ----
    z_recon = generate(adapter, z_noise, pe, pooled, args.steps, args.device)

    # ---- diagnostics ----
    def _norm(x):
        return float(x.pow(2).mean().sqrt())

    def _cos(a, b):
        na, nb = a.norm(), b.norm()
        return float((a * b).sum() / (na * nb + 1e-8))

    print(f"\nlatent norms:")
    print(f"  ||z_src||    = {_norm(z_src):.4f}")
    print(f"  ||z_noise||  = {_norm(z_noise):.4f}   (expect ~1.0)")
    print(f"  ||z_recon||  = {_norm(z_recon):.4f}")
    print(f"\ncos(z_recon, z_src) = {_cos(z_recon, z_src):+.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_recon = helper.decode_latent(z_recon)
    img_noise = helper.decode_latent(z_noise)

    src_arr = np.asarray(src)
    recon_arr = np.asarray(Image.fromarray(img_recon).resize((512, 512)))
    noise_arr = np.asarray(Image.fromarray(img_noise).resize((512, 512)))
    panel = np.concatenate([src_arr, recon_arr, noise_arr], axis=1)
    Image.fromarray(panel).save(out / "roundtrip.png")
    print(f"saved {out / 'roundtrip.png'}  (src | z_recon | z_noise)")


if __name__ == "__main__":
    main()