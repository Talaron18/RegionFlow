"""Discrimination check: does a given feature extractor distinguish
"blurry MSE optimum" from "sharp target" in feature space?

The MSE-vs-LPIPS check (scripts/mse_lpips_correlation.py) showed Pearson
r = 0.23 -- almost independent -- and yet LPIPS training did not produce
sharper images. That suggests LPIPS's feature space does not encode the
same thing as "looks sharp" for this task.

Before paying for training with a different feature extractor, this
script asks the cheaper prior question: given the current sigma0=0.0
checkpoint's prediction (blurry) and the target (sharp), how far apart
does the extractor score them, relative to the perfect-oracle prediction
which is by construction inside the mask equal to the target?

If the gap is small, training with this extractor reproduces LPIPS's
failure. If the gap is large, the extractor has leverage.

No training, no gradients. One forward per sample through
adapter + VAE + extractor.

Usage:
    python scripts/feature_discrimination.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --every-nth 40 \
        --out runs/gate_a_sigma_0.0/feat_disc_dinov2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache, torch_dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def decode_latent(z: torch.Tensor, vae) -> torch.Tensor:
    """z: (B, C, h, w) fp32. Returns (B, 3, H, W) float32 in [-1, 1]."""
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    scaling = float(vae.config.scaling_factor)
    z_norm = z / scaling + shift
    img = vae.decode(z_norm.to(next(vae.parameters()).dtype)).sample
    return img.float().clamp(-1, 1)


def dino_normalize(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W) in [-1, 1]. Returns imagenet-normalized, resized."""
    img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
    mean = torch.tensor(IMAGENET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img.device).view(1, 3, 1, 1)
    img = (img - mean) / std
    # DINOv2-base is patch 14, we feed 224 -> 16x16 = 256 patches
    img = F.interpolate(img, size=(224, 224), mode="bilinear",
                        align_corners=False)
    return img


@torch.no_grad()
def dino_patches(dino, img_neg1_1: torch.Tensor) -> torch.Tensor:
    """img_neg1_1: (B, 3, H, W) in [-1, 1].

    Returns L2-normalized patch features of shape (B, N, D), N = 256 for a
    224x224 input with patch size 14.
    """
    x = dino_normalize(img_neg1_1)
    out = dino(pixel_values=x)
    patches = out.last_hidden_state[:, 1:, :]  # drop CLS
    patches = patches / patches.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return patches


def mask_to_patch_grid(mask: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """mask: (B, 1, h, w) in {0, 1}. Returns (B, N) bool, N = grid*grid."""
    m = F.adaptive_avg_pool2d(mask, (grid, grid))
    return (m.view(mask.shape[0], -1) > 0.5)


def masked_patch_cosine_dist(a: torch.Tensor, b: torch.Tensor,
                             m: torch.Tensor) -> torch.Tensor:
    """a, b: (B, N, D) L2-normalized. m: (B, N) bool.
    Returns (B,) mean 1-cos inside mask."""
    cos = (a * b).sum(dim=-1)  # (B, N)
    d = (1.0 - cos) * m.float()
    return d.sum(dim=-1) / m.float().sum(dim=-1).clamp(min=1.0)


def masked_patch_l2sq(a: torch.Tensor, b: torch.Tensor,
                      m: torch.Tensor) -> torch.Tensor:
    """Same, but squared L2 per patch (before the per-patch mean)."""
    sq = (a - b).pow(2).sum(dim=-1)  # (B, N)
    return (sq * m.float()).sum(dim=-1) / m.float().sum(dim=-1).clamp(min=1.0)


def latent_mse_in_mask(a: torch.Tensor, b: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """a, b: (B, C, h, w). mask: (B, 1, h, w) in {0, 1}.
    Returns (B,) mean per-element squared error inside mask."""
    C = a.shape[1]
    n_px = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return ((a - b).pow(2) * mask).sum(dim=(1, 2, 3)) / (n_px * C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-nth", type=int, default=40)
    ap.add_argument("--n-max", type=int, default=5000)
    ap.add_argument("--extractor", default="dinov2",
                    choices=("dinov2",))
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    # ---- DINOv2 --------------------------------------------------------
    try:
        from transformers import AutoModel, Dinov2Model  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pip install transformers first, >= 4.30 for DINOv2 support:\n"
            "    pip install -U transformers")
    print(f"loading {args.extractor}...")
    try:
        dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
    except Exception:
        dino = AutoModel.from_pretrained("facebook/dinov2-base")
    dino = dino.to(args.device).eval()
    dino.requires_grad_(False)

    # ---- LPIPS (only for the side-by-side reference) -------------------
    lpips_fn = None
    try:
        import lpips as lpips_lib
        print("loading LPIPS (for comparison)...")
        lpips_fn = lpips_lib.LPIPS(net="alex").to(args.device).eval()
        lpips_fn.requires_grad_(False)
    except ImportError:
        print("  lpips not installed; skipping LPIPS column")

    # ---- VAE -----------------------------------------------------------
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    print("loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32
    ).to(args.device).eval()
    vae.requires_grad_(False)

    print("loading transformer...")
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

    # ---- data ----------------------------------------------------------
    cache = TripletCache(args.cache)
    ds = torch_dataset(cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2)

    records = []
    i0 = 0
    with torch.no_grad():
        for batch in dl:
            if i0 >= args.n_max:
                break
            z_src = batch["z_src"].to(args.device).float()
            z_tgt = batch["z_tgt"].to(args.device).float()
            mask = batch["mask"].to(args.device).float()
            pe = batch["prompt_embeds"].to(args.device)
            pooled = batch["pooled"].to(args.device)
            b = z_src.shape[0]

            s0 = torch.zeros(b, device=args.device)
            u = adapter(z_src, s0, s0, pe, pooled, mask=mask)

            v = z_tgt - z_src

            z_pred = z_src + u * mask       # model output, blurry
            z_orac = z_src + v * mask       # oracle, sharp (== z_tgt in mask)

            img_pred = decode_latent(z_pred, vae)
            img_orac = decode_latent(z_orac, vae)
            img_tgt = decode_latent(z_tgt, vae)

            # ---- latent MSE (for reference) ----------------------------
            d_mse_pred = latent_mse_in_mask(z_pred, z_tgt, mask)
            d_mse_orac = latent_mse_in_mask(z_orac, z_tgt, mask)

            # ---- LPIPS -------------------------------------------------
            if lpips_fn is not None:
                d_lp_pred = lpips_fn(img_pred, img_tgt).flatten()
                d_lp_orac = lpips_fn(img_orac, img_tgt).flatten()
            else:
                d_lp_pred = torch.full((b,), float("nan"))
                d_lp_orac = torch.full((b,), float("nan"))

            # ---- DINOv2 patch features ---------------------------------
            p_pred = dino_patches(dino, img_pred)
            p_orac = dino_patches(dino, img_orac)
            p_tgt = dino_patches(dino, img_tgt)

            m_patch = mask_to_patch_grid(mask, grid=16)

            d_dino_pred = masked_patch_cosine_dist(p_pred, p_tgt, m_patch)
            d_dino_orac = masked_patch_cosine_dist(p_orac, p_tgt, m_patch)

            d_dino_l2_pred = masked_patch_l2sq(p_pred, p_tgt, m_patch)
            d_dino_l2_orac = masked_patch_l2sq(p_orac, p_tgt, m_patch)

            for j in range(b):
                global_idx = i0 + j
                if global_idx % args.every_nth != 0:
                    continue
                if global_idx >= len(cache):
                    continue
                rec = cache[global_idx]
                meta = rec["meta"]
                records.append({
                    "idx": global_idx,
                    "id": meta.get("id"),
                    "instruction": meta.get("instruction"),
                    "mask_frac": float(meta.get("mask_frac", -1.0)),
                    "d_mse_pred": float(d_mse_pred[j]),
                    "d_mse_orac": float(d_mse_orac[j]),
                    "d_lpips_pred": float(d_lp_pred[j]),
                    "d_lpips_orac": float(d_lp_orac[j]),
                    "d_dino_cos_pred": float(d_dino_pred[j]),
                    "d_dino_cos_orac": float(d_dino_orac[j]),
                    "d_dino_l2_pred": float(d_dino_l2_pred[j]),
                    "d_dino_l2_orac": float(d_dino_l2_orac[j]),
                })
            i0 += b
            if records and len(records) % 25 == 0:
                print(f"  collected {len(records)}")

    if len(records) < 8:
        print(f"only {len(records)} samples; need at least 8")
        return

    # ---- aggregate ---------------------------------------------------
    def arr(key):
        return np.array([r[key] for r in records])

    def stat(name, pred, orac):
        valid = ~(np.isnan(pred) | np.isnan(orac))
        if valid.sum() == 0:
            print(f"  {name:22s}  (all NaN)")
            return
        pr = np.median(pred[valid])
        orc = np.median(orac[valid])
        ratio = pr / orc if orc > 1e-12 else float("inf")
        print(f"  {name:22s}  pred {pr:>10.5f}  oracle {orc:>10.5f}  "
              f"pred/oracle {ratio:>8.2f}")
        return ratio

    print(f"\n{'=' * 72}")
    print(f"n = {len(records)}")
    print(f"  Distances in the mask region. Oracle = z_src + (z_tgt-z_src)*M,")
    print(f"  which equals z_tgt inside the mask, so d_oracle is the floor.")
    print(f"  Higher pred/oracle ratio = the extractor separates the blurry")
    print(f"  prediction from the sharp target more cleanly.\n")

    r_mse = stat("latent MSE",
                 arr("d_mse_pred"), arr("d_mse_orac"))
    r_lp = None
    if lpips_fn is not None:
        r_lp = stat("LPIPS",
                    arr("d_lpips_pred"), arr("d_lpips_orac"))
    r_dino_cos = stat("DINOv2 patch cosine",
                      arr("d_dino_cos_pred"), arr("d_dino_cos_orac"))
    r_dino_l2 = stat("DINOv2 patch L2^2",
                     arr("d_dino_l2_pred"), arr("d_dino_l2_orac"))

    print("\n  Interpretation:")
    print("    The right question is NOT whether DINO gives a big number --")
    print("    it is whether the RATIO pred/oracle is larger than LPIPS's,")
    print("    which would mean DINO sees the difference LPIPS misses.")

    if r_lp is not None and r_dino_cos is not None:
        if r_dino_cos > 1.5 * r_lp:
            print(f"    DINO ratio {r_dino_cos:.2f} vs LPIPS {r_lp:.2f}  "
                  f"-> DINO has more leverage. Worth training.")
        elif r_dino_cos < 0.7 * r_lp:
            print(f"    DINO ratio {r_dino_cos:.2f} vs LPIPS {r_lp:.2f}  "
                  f"-> DINO is LESS discriminative here. Not the fix.")
        else:
            print(f"    DINO ratio {r_dino_cos:.2f} vs LPIPS {r_lp:.2f}  "
                  f"-> comparable. Inspect the per-sample table before "
                  f"committing.")

    # per-sample dump: worst cases where DINO should separate but doesn't
    print("\n  Samples where DINO is LEAST discriminative "
          "(d_dino_cos_pred small relative to oracle):")
    ratio_per = arr("d_dino_cos_pred") / np.clip(arr("d_dino_cos_orac"),
                                                 1e-12, None)
    order = np.argsort(ratio_per)
    print(f"  {'idx':>6}  {'pred':>10}  {'oracle':>10}  {'ratio':>7}  "
          f"instruction")
    for i in order[:8]:
        r = records[i]
        print(f"  {r['idx']:>6d}  {r['d_dino_cos_pred']:>10.5f}  "
              f"{r['d_dino_cos_orac']:>10.5f}  {ratio_per[i]:>7.2f}  "
              f"{r['instruction'][:48]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "extractor": args.extractor,
        "n": len(records),
        "median": {
            "mse": {"pred": float(np.nanmedian(arr("d_mse_pred"))),
                    "oracle": float(np.nanmedian(arr("d_mse_orac")))},
            "lpips": {"pred": float(np.nanmedian(arr("d_lpips_pred"))),
                      "oracle": float(np.nanmedian(arr("d_lpips_orac")))},
            "dino_cos": {"pred": float(np.median(arr("d_dino_cos_pred"))),
                         "oracle": float(np.median(arr("d_dino_cos_orac")))},
            "dino_l2": {"pred": float(np.median(arr("d_dino_l2_pred"))),
                        "oracle": float(np.median(arr("d_dino_l2_orac")))},
        },
        "per_sample": records,
    }, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()