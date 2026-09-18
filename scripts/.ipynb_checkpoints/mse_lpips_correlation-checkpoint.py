"""Correlation between latent MSE and image-space perceptual distance.

The question this answers, in one number: is MSE already capturing what
LPIPS would measure? If yes, adding a perceptual loss to the training
objective cannot change the optimum in a useful way, and the BiFlow-
inspired path is dead. If no, MSE is missing something LPIPS catches,
and the perceptual loss has traction.

Also dumps the largest rank-disagreement samples -- the cases where MSE
says "good" and LPIPS says "bad" -- which is exactly the failure mode the
whole investigation is chasing.

Requires:  pip install lpips
(LPIPS weights download on first use, ~30 MB, from a public CDN.)

Usage:
    python scripts/mse_lpips_correlation.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --every-nth 40 \
        --out runs/gate_a_sigma_0.0/mse_lpips.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache, torch_dataset


@torch.no_grad()
def decode_latent(z: torch.Tensor, vae) -> torch.Tensor:
    """z: (B, C, h, w) fp32. Returns (B, 3, H, W) float32 in [-1, 1]."""
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    z_norm = z / vae.config.scaling_factor + shift
    img = vae.decode(z_norm.to(vae.dtype)).sample
    return img.float().clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-nth", type=int, default=40,
                    help="sample every Nth cached item; 40 -> ~113 items "
                         "from a 4511-item cache")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    try:
        import lpips
    except ImportError:
        raise SystemExit(
            "pip install lpips first:\n"
            "    pip install lpips\n"
            "The LPIPS weights download on first use (~30 MB).")

    try:
        from scipy.stats import spearmanr
        have_scipy = True
    except ImportError:
        have_scipy = False

    print("loading LPIPS (alex)...")
    lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()

    from diffusers import AutoencoderKL, SD3Transformer2DModel

    print("loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float16
    ).to(args.device).eval()

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

    cache = TripletCache(args.cache)
    ds = torch_dataset(cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=2)

    records = []
    i0 = 0
    with torch.no_grad():
        for batch in dl:
            z_src = batch["z_src"].to(args.device).float()
            z_tgt = batch["z_tgt"].to(args.device).float()
            mask = batch["mask"].to(args.device).float()
            pe = batch["prompt_embeds"].to(args.device)
            pooled = batch["pooled"].to(args.device)
            b = z_src.shape[0]

            s0 = torch.zeros(b, device=args.device)
            u = adapter(z_src, s0, s0, pe, pooled, mask=mask)

            v = z_tgt - z_src
            m_b = mask
            C = u.shape[1]
            n_px = m_b.sum(dim=(1, 2, 3)).clamp(min=1.0)
            mse_in = ((u - v).pow(2) * m_b).sum(dim=(1, 2, 3)) / (n_px * C)

            z_pred = z_src + u * m_b
            img_pred = decode_latent(z_pred, vae)
            img_tgt = decode_latent(z_tgt, vae)
            lpips_val = lpips_fn(img_pred, img_tgt).flatten()

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
                    "mse_in_mask": float(mse_in[j].item()),
                    "lpips": float(lpips_val[j].item()),
                })
            i0 += b
            if len(records) and len(records) % 25 == 0:
                print(f"  collected {len(records)}")

    if len(records) < 8:
        print(f"only {len(records)} samples; need at least 8")
        return

    mse_arr = np.array([r["mse_in_mask"] for r in records])
    lpips_arr = np.array([r["lpips"] for r in records])

    pearson = float(np.corrcoef(mse_arr, lpips_arr)[0, 1])
    if have_scipy:
        spearman = float(spearmanr(mse_arr, lpips_arr).correlation)
    else:
        # rank correlation by hand
        r1 = mse_arr.argsort().argsort().astype(float)
        r2 = lpips_arr.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(r1, r2)[0, 1])

    n = len(records)
    rank_mse = mse_arr.argsort().argsort() / (n - 1)
    rank_lpips = lpips_arr.argsort().argsort() / (n - 1)
    rank_diff = np.abs(rank_mse - rank_lpips)

    mse_top_q = mse_arr >= np.percentile(mse_arr, 75)
    lpips_top_q = lpips_arr >= np.percentile(lpips_arr, 75)
    overlap = float((mse_top_q & lpips_top_q).mean())

    print(f"\n{'=' * 72}")
    print(f"n = {n}")
    print(f"  Pearson  r(MSE, LPIPS) = {pearson:+.4f}")
    print(f"  Spearman rho           = {spearman:+.4f}")
    print(f"  top-quartile overlap   = {overlap:.3f}  "
          f"(0.0625 if independent, 0.25 if perfectly aligned)")
    print(f"  rank-disagreement med  = {np.median(rank_diff):.3f}")

    def _dump(title, order, k=5):
        print(f"\n  -- {title} --")
        print(f"  {'idx':>6}  {'MSE_in':>10}  {'LPIPS':>8}  {'rank_diff':>9}  "
              f"instruction")
        for i in order[:k]:
            r = records[i]
            print(f"  {r['idx']:>6d}  {r['mse_in_mask']:>10.5f}  "
                  f"{r['lpips']:>8.4f}  {rank_diff[i]:>9.3f}  "
                  f"{r['instruction'][:45]}")

    _dump("worst by LPIPS", lpips_arr.argsort()[::-1])
    _dump("worst by MSE", mse_arr.argsort()[::-1])
    _dump("largest rank disagreement (MSE good, LPIPS bad or vice versa)",
          rank_diff.argsort()[::-1])

    print("\n  INTERPRETATION:")
    if pearson > 0.8:
        print(f"    Pearson r = {pearson:.3f} > 0.8.")
        print("    -> MSE and LPIPS are aligned. Adding a perceptual loss "
              "cannot change the optimum")
        print("       in a way MSE doesn't already reach.")
        print("    -> Do NOT train with perceptual loss. The BiFlow insight "
              "does not transfer.")
    elif pearson < 0.5:
        print(f"    Pearson r = {pearson:.3f} < 0.5.")
        print("    -> MSE and LPIPS measure different things.")
        print("    -> Perceptual loss has traction: the optimum would change.")
        print("    -> Proceed to train with perceptual loss "
              "(--perceptual-weight 0.1 to start).")
    else:
        print(f"    Pearson r = {pearson:.3f} (moderate).")
        print("    -> Partial overlap. Inspect the largest-rank-disagreement "
              "samples.")
        print("    -> If disagreement concentrates on the smearing failure "
              "mode, perceptual loss is motivated.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "n": n,
        "pearson": pearson,
        "spearman": spearman,
        "top_quartile_overlap": overlap,
        "rank_diff_median": float(np.median(rank_diff)),
        "per_sample": records,
    }, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()