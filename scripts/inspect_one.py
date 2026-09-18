"""Inspect one or more cached items: model prediction vs ground truth.

Decodes four panels through the VAE:
    source | dataset target | oracle displacement | model prediction

The oracle panel applies the dataset's OWN z_tgt - z_src inside the mask.
The model panel applies the adapter's u at s=0, same mask pin. Comparing
the last two shows what the model lost: noise, smear, or a shifted version
of the right edit.

Usage:
    python scripts/inspect_one.py \
        --idx 0 2 \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --out runs/gate_a_sigma_0.0/inspect
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


@torch.no_grad()
def decode(z: torch.Tensor, vae, device: str) -> np.ndarray:
    """latent (1, C, H, W) fp32 -> uint8 (H*8, W*8, 3)."""
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    lat = z.to(device).float() / vae.config.scaling_factor + shift
    img = vae.decode(lat.to(vae.dtype)).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    return (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _ensure_batch(t: torch.Tensor, ndim: int) -> torch.Tensor:
    while t.dim() < ndim:
        t = t.unsqueeze(0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True,
                    help="output prefix; files are <out>_idx<N>.png")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    ap.add_argument("--no-pin", action="store_true",
                    help="do NOT restrict displacement to the mask")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--strength", type=float, nargs="+", default=[1.0],
                    help="one or more scale values; each produces its own "
                         "panel. e.g. --strength 1 2 3")
    args = ap.parse_args()

    from diffusers import AutoencoderKL, SD3Transformer2DModel

    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float16
    ).to(args.device).eval()

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

    out_pref = Path(args.out)
    out_pref.parent.mkdir(parents=True, exist_ok=True)

    for idx in args.idx:
        rec = cache[idx]
        meta = rec["meta"]
        print(f"\nidx {idx}: id={meta['id']}  mask_frac={meta['mask_frac']:.4f}")
        print(f"  instruction: {meta['instruction']!r}")

        z_src = _ensure_batch(torch.from_numpy(rec["z_src"]).float(), 4).to(args.device)
        z_tgt = _ensure_batch(torch.from_numpy(rec["z_tgt"]).float(), 4).to(args.device)
        mask = _ensure_batch(torch.from_numpy(rec["mask"]).float(), 4).to(args.device)
        # mask: ensure channel dim is 1
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        pe = _ensure_batch(torch.from_numpy(rec["prompt_embeds"]), 3).to(args.device)
        pooled = _ensure_batch(torch.from_numpy(rec["pooled"]), 2).to(args.device)

        with torch.no_grad():
            s0 = torch.zeros(z_src.shape[0], device=args.device)
            u = adapter(z_src, s0, s0, pe, pooled, mask=mask)

        m = torch.ones_like(mask) if args.no_pin else mask
        v_clean = z_tgt - z_src
        z_oracle = z_src + v_clean * m
        u_norm = float((u * m).pow(2).sum().sqrt())
        print(f"  ||u*m|| = {u_norm:.4f}")
        for s in args.strength:
            print(f"    a={s:g}  ||z_pred - z_src|| = {s * u_norm:.4f}")
        # per-panel cosines for the printout
        def _cos(a, b, msk):
            num = float((a * b * msk).sum())
            den = (float((a.pow(2) * msk).sum()) ** 0.5
                   * float((b.pow(2) * msk).sum()) ** 0.5)
            return num / den if den > 1e-8 else float("nan")

        print(f"  cos(u, v) inside mask:   {_cos(u, v_clean, mask):+.3f}")
        print(f"  cos(u, v) full frame:    {_cos(u, v_clean, torch.ones_like(mask)):+.3f}")

        panels = [
            ("source", z_src),
            ("dataset target", z_tgt),
            ("oracle  z_src + (z_tgt-z_src)*M", z_oracle),
        ]
        for s in args.strength:
            panels.append((f"model a={s:g}", z_src + s * u * m))
        ims = []
        labels = []
        for lab, z in panels:
            try:
                arr = decode(z, vae, args.device)
                ims.append(Image.fromarray(arr).resize((args.size, args.size),
                                                       Image.LANCZOS))
                labels.append(lab)
            except Exception as e:  # noqa: BLE001
                print(f"    decode failed for '{lab}': {e}")

        sheet = Image.new("RGB", (args.size * len(ims), args.size + 24),
                          (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for i, (im, lab) in enumerate(zip(ims, labels)):
            sheet.paste(im, (i * args.size, 24))
            d.text((i * args.size + 4, 6), lab, fill=(0, 0, 0))
        path = Path(f"{out_pref}_idx{idx}.png")
        sheet.save(path)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()