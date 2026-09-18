"""Spatial concentration of the model's output u inside the mask.

If u's energy is concentrated in a few pixels, the smearing seen in
lowpass_cos is the average of multiple CONCENTRATED modes, and a
mixture head can separate them.

If u's energy is spread uniformly, each mode is itself diffuse -- a
mixture head inherits the diffusion and cannot fix it. Then the fix
has to be conditioning, not the objective.

Usage:
    python scripts/spatial_concentration.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --every-nth 20 \
        --out runs/gate_a_sigma_0.0/spatial_concentration.json
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


def concentration(u_abs2_in_mask: torch.Tensor) -> dict:
    """u_abs2_in_mask: (n_px,) -- |u|^2 summed over channels, only at
    mask pixels.

    Returns fraction of total energy in top 5/10/25/50% of pixels, and
    the effective support size = 1/sum(p^2) in pixel units, where p is
    the normalized distribution.
    """
    v = u_abs2_in_mask.flatten()
    v = v[v > 0]
    if v.numel() < 4:
        return {}
    p = v / v.sum()
    sorted_p = torch.sort(p, descending=True).values
    cum = torch.cumsum(sorted_p, 0)
    n = len(sorted_p)
    out = {}
    for frac in (0.05, 0.10, 0.25, 0.50):
        idx = max(0, int(frac * n) - 1)
        out[f"top{int(frac*100)}"] = float(cum[idx])
    out["eff_support_px"] = float(1.0 / p.pow(2).sum())
    out["n_px"] = int(n)
    out["eff_support_frac"] = out["eff_support_px"] / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-nth", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=5000)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    args = ap.parse_args()

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
    ds = torch_dataset(cache)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

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

            for j in range(b):
                global_idx = i0 + j
                if global_idx % args.every_nth != 0:
                    continue
                if global_idx >= len(cache):
                    continue
                m = mask[j, 0] > 0.5
                if not m.any():
                    continue
                u_abs2 = u[j].pow(2).sum(dim=0)         # (H, W)
                v_dis = z_tgt[j] - z_src[j]             # (C, H, W)
                v_abs2 = v_dis.pow(2).sum(dim=0)        # (H, W)
                cu = concentration(u_abs2[m])
                cv = concentration(v_abs2[m])
                if not cu or not cv:
                    continue
                rec = cache[global_idx]
                meta = rec["meta"]
                row = {
                    "idx": global_idx,
                    "id": meta.get("id"),
                    "instruction": meta.get("instruction"),
                    "mask_frac": float(meta.get("mask_frac", -1.0)),
                }
                for k, val in cu.items():
                    row[f"u_{k}"] = val
                for k, val in cv.items():
                    row[f"v_{k}"] = val
                records.append(row)
            i0 += b

    if not records:
        print("no records")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=1))

    print(f"\nSpatial concentration, n = {len(records)}")
    print(f"  {'metric':22s}  {'u (model)':>12}  {'v (target)':>12}")
    for key in ("top5", "top10", "top25", "top50", "eff_support_frac"):
        au = np.array([r[f"u_{key}"] for r in records])
        av = np.array([r[f"v_{key}"] for r in records])
        print(f"  {key:22s}  {np.median(au):>12.3f}  {np.median(av):>12.3f}")

    print("\n  Reference: uniform energy gives top10 = 0.10, "
          "top25 = 0.25, eff_support_frac = 1.00.")

    u_t10 = float(np.median([r["u_top10"] for r in records]))
    v_t10 = float(np.median([r["v_top10"] for r in records]))
    print(f"\n  u_top10 median = {u_t10:.3f}")
    print(f"  v_top10 median = {v_t10:.3f}")
    gap = v_t10 - u_t10
    if v_t10 > 0.40 and gap > 0.10:
        print(f"  -> v IS concentrated, u is NOT (gap {gap:+.3f}).")
        print("     Model is over-smoothing. Rank / edge-loss have motivation.")
    elif v_t10 < 0.30 and abs(gap) < 0.10:
        print(f"  -> v ITSELF is diffuse (v_top10 {v_t10:.3f}).")
        print("     The model is fitting the target. Rank / edge-loss would "
              "be fitting noise -- fix the data/task, not the model.")
    else:
        print(f"  -> mixed. v_top10 - u_top10 = {gap:+.3f}. Inspect "
              "per-sample.")
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()