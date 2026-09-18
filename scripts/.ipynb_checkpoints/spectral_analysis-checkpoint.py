"""Spectral analysis of the model's error, to separate two failure modes:

  (a) low-frequency blur -- the model's output u is a spatially smooth
      version of v (keeps direction, loses detail). Signature: gradient
      energy of u is much smaller than of v, and P_u(k)/P_v(k) falls
      steeply with frequency k.

  (b) high-frequency noise -- u = v + noise. Signature: gradient energy
      of u is comparable to or larger than v, and P_u/P_v is flat or
      rising with k.

These have different fixes, so telling them apart first is worth the
five minutes. No training. Uses cached latents + adapter only.

Usage:
    python scripts/spectral_analysis.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --idx 0 2 42 100 500 \
        --out runs/gate_a_sigma_0.0/spectral.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


def _ensure_batch(t: torch.Tensor, ndim: int) -> torch.Tensor:
    while t.dim() < ndim:
        t = t.unsqueeze(0)
    return t


# ----------------------------------------------------------------------
# gradient energy -- cheapest and most interpretable diagnostic
# ----------------------------------------------------------------------

def grad_energy(x: torch.Tensor, mask: torch.Tensor) -> float:
    """Sum of |grad x|^2 over pixels inside the mask.

    Finite differences on the H, W axes, restricted to pixel pairs where
    BOTH endpoints are inside the mask -- otherwise the boundary
    contributes an artificial edge that dominates the sum.

    x:    (C, H, W)
    mask: (1, H, W)
    """
    C, H, W = x.shape
    m = mask.reshape(H, W) > 0.5

    # horizontal differences: x[:, :, 1:] - x[:, :, :-1]
    mh = m[:, 1:] & m[:, :-1]              # (H, W-1)
    gh = x[:, :, 1:] - x[:, :, :-1]        # (C, H, W-1)
    # vertical
    mv = m[1:, :] & m[:-1, :]              # (H-1, W)
    gv = x[:, 1:, :] - x[:, :-1, :]        # (C, H-1, W)

    eh = (gh.pow(2) * mh).sum().item()
    ev = (gv.pow(2) * mv).sum().item()
    n = mh.sum().item() + mv.sum().item()
    if n == 0:
        return float("nan")
    return (eh + ev) / n


# ----------------------------------------------------------------------
# radial power spectrum
# ----------------------------------------------------------------------

def radial_psd(x: torch.Tensor, mask: torch.Tensor,
               n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density of x * mask, averaged
    over channels.

    Returns (bin_centers_normalized, psd). Bin 0 is DC; last bin is near
    Nyquist. Frequency is normalized so the max spatial frequency is 1.0.

    Note: multiplying by a hard mask before FFT creates spectral leakage
    at the mask boundary. Both u and v are treated identically, so the
    LEAKAGE CANCELS IN THE RATIO P_u/P_v -- that ratio is the quantity
    this function exists to compute. Absolute PSD values are not
    meaningful in isolation.
    """
    C, H, W = x.shape
    x = x * mask.reshape(1, H, W)
    F = torch.fft.fft2(x.double(), dim=(-2, -1))   # fp64 for stability
    P = F.abs().pow(2)                             # (C, H, W)

    ky = torch.fft.fftfreq(H, device=x.device).double()
    kx = torch.fft.fftfreq(W, device=x.device).double()
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K = (KY.pow(2) + KX.pow(2)).sqrt()
    K = K / K.max().clamp(min=1e-12)              # in [0, 1]

    edges = torch.linspace(0, 1.0, n_bins + 1, device=x.device).double()
    centers = ((edges[:-1] + edges[1:]) / 2).cpu().numpy()
    psd = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == n_bins - 1:
            sel = (K >= lo) & (K <= hi)
        else:
            sel = (K >= lo) & (K < hi)
        n = int(sel.sum().item())
        if n == 0:
            continue
        # mean over channels and over frequencies in this radial bin
        psd[b] = P[:, sel].sum().item() / (C * n)
    return centers, psd


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None,
                help="specific item indices; omit if using --every-nth")
    
    ap.add_argument("--out", required=True)
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--every-nth", type=int, default=0,
                    help="if > 0, IGNORE --idx and instead sample every "
                         "Nth cached item up to --n-max")
    ap.add_argument("--n-max", type=int, default=500,
                    help="cap for --every-nth")
    args = ap.parse_args()
    if args.every_nth <= 0 and not args.idx:
        ap.error("must pass --idx (one or more) or --every-nth > 0")

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
    n_total = len(cache)

    if args.every_nth > 0:
        idx_list = list(range(0, n_total, args.every_nth))[:args.n_max]
    else:
        idx_list = args.idx

    # per-sample records
    records = []
    psd_u_all = []
    psd_v_all = []

    for idx in idx_list:
        rec = cache[idx]
        meta = rec["meta"]

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
        u = u[0]        # (C, H, W)
        v = v[0]
        m = mask[0]

        # gradient energies inside the mask
        gu = grad_energy(u, m)
        gv = grad_energy(v, m)

        # per-sample cosine (sanity: should match diagnose)
        num = float((u * v * m).sum())
        den = (float((u.pow(2) * m).sum()) ** 0.5
               * float((v.pow(2) * m).sum()) ** 0.5)
        cos = num / den if den > 1e-8 else float("nan")

        # radial PSD
        centers, psd_u = radial_psd(u, m, args.n_bins)
        _, psd_v = radial_psd(v, m, args.n_bins)
        psd_u_all.append(psd_u)
        psd_v_all.append(psd_v)

        # per-sample key ratios
        # high-freq half = upper half of radial bins (k > 0.5)
        half = args.n_bins // 2
        hf_u = float(psd_u[half:].sum())
        hf_v = float(psd_v[half:].sum())
        lf_u = float(psd_u[:half].sum())
        lf_v = float(psd_v[:half].sum())

        rec_out = {
            "idx": idx,
            "id": meta.get("id"),
            "instruction": meta.get("instruction"),
            "mask_frac": float(meta.get("mask_frac", -1.0)),
            "moved": float(meta.get("moved", -1.0)),
            "cos_in_mask": cos,
            "grad_energy_u": gu,
            "grad_energy_v": gv,
            "grad_ratio_u_over_v": gu / gv if gv > 1e-12 else float("nan"),
            "lf_ratio_u_over_v": lf_u / lf_v if lf_v > 1e-12 else float("nan"),
            "hf_ratio_u_over_v": hf_u / hf_v if hf_v > 1e-12 else float("nan"),
            "psd_u": psd_u.tolist(),
            "psd_v": psd_v.tolist(),
        }
        records.append(rec_out)
        print(f"  idx {idx:5d}  mask {rec_out['mask_frac']:.3f}  "
              f"cos {cos:+.3f}  grad u/v {rec_out['grad_ratio_u_over_v']:.3f}  "
              f"hf u/v {rec_out['hf_ratio_u_over_v']:.3f}  "
              f"{meta.get('instruction', '')[:50]}")

    # aggregate PSD over samples
    psd_u_mean = np.mean(np.stack(psd_u_all), axis=0)
    psd_v_mean = np.mean(np.stack(psd_v_all), axis=0)

    print("\n" + "=" * 74)
    print(f"AGGREGATE over {len(records)} samples")
    print("=" * 74)

    cos_arr = np.array([r["cos_in_mask"] for r in records])
    gr_arr = np.array([r["grad_ratio_u_over_v"] for r in records])
    hf_arr = np.array([r["hf_ratio_u_over_v"] for r in records])
    lf_arr = np.array([r["lf_ratio_u_over_v"] for r in records])

    def _stat(name, arr):
        print(f"  {name:32s}  median {np.median(arr):+.4f}  "
              f"mean {arr.mean():+.4f}  "
              f"p10 {np.percentile(arr, 10):+.4f}  "
              f"p90 {np.percentile(arr, 90):+.4f}")

    _stat("cos(u,v) in mask", cos_arr)
    _stat("grad_energy(u)/grad_energy(v)", gr_arr)
    _stat("LF power ratio (k<0.5)", lf_arr)
    _stat("HF power ratio (k>0.5)", hf_arr)

    print(f"\n  radial PSD (mean over {len(records)} samples):")
    print(f"  {'k (norm)':>10}  {'P_u':>14}  {'P_v':>14}  {'P_u/P_v':>10}")
    centers = np.linspace(0.5 / args.n_bins, 1.0 - 0.5 / args.n_bins, args.n_bins)
    for i, k in enumerate(centers):
        pu = psd_u_mean[i]
        pv = psd_v_mean[i]
        ratio = pu / pv if pv > 1e-30 else float("nan")
        print(f"  {k:>10.4f}  {pu:>14.4e}  {pv:>14.4e}  {ratio:>10.4f}")

    # interpretation
    print("\n  INTERPRETATION:")
    med_grad = float(np.median(gr_arr))
    med_hf = float(np.median(hf_arr))
    if med_grad < 0.5 and med_hf < 0.5:
        print(f"    grad ratio {med_grad:.3f} < 0.5 and HF ratio {med_hf:.3f} "
              "< 0.5")
        print("    -> LOW-FREQUENCY BLUR. u keeps direction, loses detail.")
        print("    -> consistent with conditional-mean averaging. "
              "Direction C is motivated.")
    elif med_grad > 1.5:
        print(f"    grad ratio {med_grad:.3f} > 1.5")
        print("    -> u has MORE high-frequency content than v. "
              "Consistent with additive noise.")
        print("    -> not the conditional-mean signature. Look elsewhere.")
    else:
        print(f"    grad ratio {med_grad:.3f}, HF ratio {med_hf:.3f} -- "
              "in between.")
        print("    -> partial blur, partial noise, or mixed. Inspect the "
              "radial PSD curve:")
        print("       monotone falloff in P_u/P_v -> blur")
        print("       flat or rising P_u/P_v at high k -> noise")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "n_samples": len(records),
        "psd_centers": centers.tolist(),
        "psd_u_mean": psd_u_mean.tolist(),
        "psd_v_mean": psd_v_mean.tolist(),
        "summary": {
            "cos_median": float(np.median(cos_arr)),
            "grad_ratio_median": med_grad,
            "lf_ratio_median": float(np.median(lf_arr)),
            "hf_ratio_median": med_hf,
        },
        "per_sample": records,
    }, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()