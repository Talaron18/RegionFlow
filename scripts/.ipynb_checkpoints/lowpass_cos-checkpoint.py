"""Low-pass cosine check on the model's error.

Hypothesis under test: is the model's output u a spatially MISALIGNED
version of v?

If yes, low-pass filtering both fields -- which destroys fine spatial
alignment but preserves coarse structure -- should RAISE their cosine.

If the misalignment is present at all spatial scales, low-passing won't
help and cos(lp) ~ cos(full).

Cheap: one forward pass per sample, plus one separable Gaussian conv per
sigma. No training, no model weights beyond the adapter.

Usage:
    python scripts/lowpass_cos.py \
        --ckpt runs/gate_a_sigma_0.0/ckpt_latest.pt \
        --cache /data/cache/13061d51aee4 \
        --mask-encoder-kind single \
        --idx 0 2 \
        --out runs/gate_a_sigma_0.0/lowpass_idx02.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter
from transport.data import TripletCache


def _ensure_batch(t: torch.Tensor, ndim: int) -> torch.Tensor:
    while t.dim() < ndim:
        t = t.unsqueeze(0)
    return t


def gaussian_kernel_1d(sigma: float, device, dtype):
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    k = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k.to(dtype)


def gaussian_blur(x: torch.Tensor, sigma: float, k1d) -> torch.Tensor:
    """x: (C, H, W). Separable Gaussian, zero padding."""
    if sigma <= 0:
        return x
    C, H, W = x.shape
    k2d = (k1d[:, None] * k1d[None, :]).unsqueeze(0).unsqueeze(0)
    k2d = k2d.expand(C, 1, -1, -1).contiguous()
    pad = k1d.shape[0] // 2
    x_pad = F.pad(x.unsqueeze(0), (pad, pad, pad, pad),
                  mode="constant", value=0.0)
    return F.conv2d(x_pad, k2d, groups=C)[0]


def erode_mask(mask: torch.Tensor, r: int) -> torch.Tensor:
    """mask: (1, H, W) float, values in {0,1}. r pixels of erosion.

    Uses max-pool on the negation. r=0 returns the mask unchanged.
    """
    if r <= 0:
        return mask
    k = 2 * r + 1
    m = -F.max_pool2d(-mask.unsqueeze(0), k, stride=1, padding=r)[0]
    return m


def masked_cos(a: torch.Tensor, b: torch.Tensor, m_bool: torch.Tensor,
               eps: float = 1e-12) -> float:
    """cosine over pixels where m_bool is True.

    a, b: (C, H, W). m_bool: (H, W) bool.
    """
    if not m_bool.any():
        return float("nan")
    a_sel = a[:, m_bool]
    b_sel = b[:, m_bool]
    num = float((a_sel * b_sel).sum())
    den = float(a_sel.pow(2).sum().sqrt()) * float(b_sel.pow(2).sum().sqrt())
    return num / den if den > eps else float("nan")


def analyze_one(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
                sigmas: list[float]) -> list[dict]:
    """u, v: (C, H, W) -- u is the model's prediction, v is the target
    displacement z_tgt - z_src. mask: (1, H, W) in {0,1}.

    Zero the outside of the mask BEFORE any blur, so (a) mask-exterior
    pixels never contribute to either side and (b) every sigma is
    evaluated on the SAME mask interior -- n does not change with sigma,
    and sigma=0 is directly comparable to the filtered rows.

    Returns one row per sigma.
    """
    mask_hw = mask[0] > 0.5                     # (H, W) bool
    n_in_mask = int(mask_hw.sum().item())

    # mask the fields up front; everything below operates on these.
    u_m = u * mask
    v_m = v * mask

    rows = []
    for sigma in sigmas:
        if sigma <= 0:
            u_lp, v_lp = u_m, v_m
            u_hp = torch.zeros_like(u_m)
            v_hp = torch.zeros_like(v_m)
        else:
            k = gaussian_kernel_1d(sigma, u.device, u.dtype)
            u_lp = gaussian_blur(u_m, sigma, k)
            v_lp = gaussian_blur(v_m, sigma, k)
            u_hp = u_m - u_lp
            v_hp = v_m - v_lp

        # every sigma evaluated on the ORIGINAL mask interior
        cos_full = masked_cos(u_m, v_m, mask_hw)
        cos_lp = masked_cos(u_lp, v_lp, mask_hw)
        cos_hp = (masked_cos(u_hp, v_hp, mask_hw)
                  if sigma > 0 else float("nan"))

        rows.append({
            "sigma": sigma,
            "n_eval_px": n_in_mask,
            "n_mask_px": n_in_mask,
            "eval_note": "original mask, fields pre-masked",
            "cos_full": cos_full,
            "cos_lp": cos_lp,
            "cos_hp": cos_hp,
            "cos_lp_orig_mask": cos_lp,
            "cos_hp_orig_mask": cos_hp,
            "lp_amp_ratio": (float(u_lp.pow(2).sum().sqrt())
                             / float(v_lp.pow(2).sum().sqrt())
                             if float(v_lp.pow(2).sum()) > 1e-12 else float("nan")),
        })
    return rows
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--every-nth", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=500)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask-encoder-kind", default="single",
                    choices=("single", "global_local"))
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
    idx_list = (list(range(0, n_total, args.every_nth))[:args.n_max]
                if args.every_nth > 0 else args.idx)

    # per-sample records
    per_sample = []

    # aggregate: cos values per sigma across samples
    agg = {s: {"full": [], "lp": [], "hp": [], "amp": []}
           for s in args.sigmas}

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
        u = u[0]
        v = v[0]
        m = mask[0]

        rows = analyze_one(u, v, m, args.sigmas)
        for r in rows:
            r["idx"] = idx
            r["id"] = meta.get("id")
            r["instruction"] = meta.get("instruction")
            r["mask_frac"] = float(meta.get("mask_frac", -1.0))
            agg[r["sigma"]]["full"].append(r["cos_full"])
            if not np.isnan(r["cos_lp"]):
                agg[r["sigma"]]["lp"].append(r["cos_lp"])
            if not np.isnan(r["cos_hp"]):
                agg[r["sigma"]]["hp"].append(r["cos_hp"])
            if not np.isnan(r["lp_amp_ratio"]):
                agg[r["sigma"]]["amp"].append(r["lp_amp_ratio"])
        per_sample.extend(rows)

        if idx in (idx_list[:5] if len(idx_list) > 5 else idx_list):
            tag = (f"  idx {idx:5d}  mask {meta.get('mask_frac', -1):.3f}  "
                   f"cos_full {rows[0]['cos_full']:+.3f}  ")
            for r in rows[1:]:
                tag += f"σ{r['sigma']:g}:{r['cos_lp']:+.3f}  "
            print(tag + f"{meta.get('instruction', '')[:40]}")

    # aggregate table
    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(idx_list)} samples")
    print("=" * 78)
    print(f"  {'sigma':>6}  {'n':>4}  {'cos(full)':>10}  {'cos(lp)':>10}  "
          f"{'cos(hp)':>10}  {'lp |u|/|v|':>12}")
    for s in args.sigmas:
        fv = np.array(agg[s]["full"])
        lv = np.array(agg[s]["lp"])
        hv = np.array(agg[s]["hp"])
        av = np.array(agg[s]["amp"])
        f_med = np.median(fv) if fv.size else float("nan")
        l_med = np.median(lv) if lv.size else float("nan")
        h_med = np.median(hv) if hv.size else float("nan")
        a_med = np.median(av) if av.size else float("nan")
        n_used = len(lv) if lv.size else 0
        print(f"  {s:>6g}  {n_used:>4d}  {f_med:>+10.4f}  {l_med:>+10.4f}  "
              f"{h_med:>+10.4f}  {a_med:>12.4f}")

    # interpretation
    print("\n  INTERPRETATION:")
    lps = {s: np.median(agg[s]["lp"]) if agg[s]["lp"] else float("nan")
           for s in args.sigmas}
    full0 = lps.get(0.0, float("nan"))
    valid = [s for s in args.sigmas if s > 0 and not np.isnan(lps[s])]
    if valid and not np.isnan(full0):
        best_s = max(valid, key=lambda s: lps[s])
        best_v = lps[best_s]
        gain = best_v - full0
        if gain > 0.10:
            print(f"    cos(lp) rises from {full0:+.3f} (full) to "
                  f"{best_v:+.3f} at sigma={best_s:g}  "
                  f"(gain {gain:+.3f})")
            print("    -> LOW-FREQUENCY ALIGNED, HIGH-FREQUENCY MISALIGNED.")
            print("    -> spatial misalignment confirmed; Direction C "
                  "(restore spatial concentration) is motivated.")
        elif gain > 0.03:
            print(f"    cos(lp) rises modestly: {full0:+.3f} -> "
                  f"{best_v:+.3f} at sigma={best_s:g}  (gain {gain:+.3f})")
            print("    -> some low-frequency alignment, but weak. "
                  "Direction C may help but is not the whole story.")
        else:
            print(f"    cos(lp) does not improve: {full0:+.3f} -> "
                  f"{best_v:+.3f} (best, at sigma={best_s:g})")
            print("    -> MISALIGNMENT IS AT ALL SCALES. Low-passing does "
                  "not recover cosine.")
            print("    -> problem is not 'fine position noise' but more "
                  "global; look at conditioning and z_src usage.")

    # save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "n_samples": len(idx_list),
        "sigmas": args.sigmas,
        "aggregate": {str(s): {
            "cos_full_median": float(np.median(agg[s]["full"]))
                              if agg[s]["full"] else None,
            "cos_lp_median": float(np.median(agg[s]["lp"]))
                             if agg[s]["lp"] else None,
            "cos_hp_median": float(np.median(agg[s]["hp"]))
                             if agg[s]["hp"] else None,
            "lp_amp_median": float(np.median(agg[s]["amp"]))
                             if agg[s]["amp"] else None,
            "n": len(agg[s]["lp"]),
        } for s in args.sigmas},
        "per_sample": per_sample,
    }, indent=1, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()