"""
End-to-end check on a single image. No benchmark data needed.

Order matters:

  1. VAE round trip      -- the ceiling. Nothing can beat this.
  2. Inversion round trip -- invert, regenerate unchanged, compare. If this is
     poor, "background preservation" downstream is inversion error, not the
     effect of any field.
  3. A real edit          -- with a hand-drawn box mask, at several radii.

Generates its own test image, so it runs before PIE-Bench arrives.

    python scripts/smoke_edit.py
    python scripts/smoke_edit.py --image photo.jpg --box 120 80 300 260 \\
        --source "a red car on a street" --target "a blue car on a street"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
OUT = Path("/data/outputs/smoke")


def synthetic(size=512):
    """Three similar objects; we will edit only the middle one."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size, 3), 0.82, np.float32)
    img[yy > size * 0.72] = (0.55, 0.45, 0.35)          # table
    centres = [size * 0.25, size * 0.5, size * 0.75]
    r = size * 0.11
    masks = []
    for cx in centres:
        m = ((yy - size * 0.5) ** 2 + (xx - cx) ** 2) < r**2
        img[m] = (0.75, 0.15, 0.12)                      # three red spheres
        masks.append(m)
    return (img * 255).astype(np.uint8), masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--box", nargs=4, type=int)
    ap.add_argument("--source", default="three red apples on a wooden table")
    ap.add_argument("--target", default="three [green] apples on a wooden table")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=512)
    # Ablations. The alpha pin holds background tokens to the source, which
    # makes background preservation near-perfect regardless of the lambda
    # field -- so a sweep run with the pin on cannot measure the field's
    # contribution to background preservation at all. Run both.
    ap.add_argument("--no-alpha", action="store_true",
                    help="disable the inversion-depth pin; isolates lambda")
    ap.add_argument("--no-lambda", action="store_true",
                    help="disable the attention bias; isolates alpha")
    ap.add_argument("--sigma-min", type=float, default=0.05,
                    help="skip inversion below this sigma. At sigma~0 the "
                         "model cannot know the noise and collapses the latent "
                         "toward zero, producing 16px patch striping.")
    ap.add_argument("--sigma-max", type=float, default=0.98)
    ap.add_argument("--sigma-sweep", action="store_true",
                    help="round-trip PSNR and noise std across sigma_min")
    ap.add_argument("--step-sweep", action="store_true",
                    help="round-trip PSNR at several step counts. If PSNR does "
                         "NOT improve with steps, inversion has a bug rather "
                         "than discretisation error.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    from bench.data import parse_brackets
    from bench.inversion import SD3EditPipeline
    from region.alpha_field import build_retention_field, paired_configs
    from region.lambda_field import build_region_field, target_token_area

    if args.image:
        img = np.array(Image.open(args.image).convert("RGB").resize((args.size,) * 2))
        x0, y0, x1, y1 = args.box
        mask = np.zeros((args.size, args.size), bool)
        mask[y0:y1, x0:x1] = True
        others = []
    else:
        img, all_masks = synthetic(args.size)
        mask = all_masks[1]                     # the MIDDLE object only
        others = [all_masks[0], all_masks[2]]

    target_prompt, spans = parse_brackets(args.target)
    print(f"source : {args.source}")
    print(f"target : {target_prompt}   (edit spans {spans})")
    print(f"mask   : {target_token_area(mask):.1f} tokens, "
          f"{mask.mean():.1%} of image")

    from diffusers import StableDiffusion3Pipeline

    print(f"\nloading {MODEL} ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    ed = SD3EditPipeline(pipe)

    # --- 1 & 2 -----------------------------------------------------------
    print(f"\n--- round trip ({args.steps} steps) ---")
    if args.sigma_sweep:
        print("  sigma_min sweep. noise std should approach 1.0; a std well")
        print("  below 1 means inversion is collapsing rather than noising.")
        print(f"    {'sigma_min':>10} {'PSNR':>8} {'noise std':>10}")
        for sm in (0.0, 0.02, 0.05, 0.10, 0.20):
            r = ed.roundtrip_error(img, args.source, steps=args.steps,
                                   sigma_min=sm, sigma_max=args.sigma_max)
            print(f"    {sm:>10.2f} {r['roundtrip_psnr']:8.2f} {r['noise_std']:10.3f}")
        return

    if args.step_sweep:
        print("  step sweep -- PSNR should rise with steps if this is merely")
        print("  Euler discretisation error; flat means a bug in invert().")
        for n in (10, 25, 50, 100):
            r = ed.roundtrip_error(img, args.source, steps=n)
            print(f"    {n:>4} steps -> {r['roundtrip_psnr']:6.2f} dB")
        return
    rt = ed.roundtrip_error(img, args.source, steps=args.steps,
                            sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    print(f"  VAE only        {rt['vae_only_psnr']:6.2f} dB   <- ceiling")
    print(f"  full round trip {rt['roundtrip_psnr']:6.2f} dB")
    print(f"  noise std       {rt['noise_std']:6.3f}      <- want ~1.0")
    gap = rt["vae_only_psnr"] - rt["roundtrip_psnr"]
    print(f"  inversion cost  {gap:6.2f} dB")
    if rt["roundtrip_psnr"] < 20:
        print("  !! Reconstruction is poor. Naive Euler inversion is drifting.")
        print("     Any background-preservation number measured now is mostly")
        print("     inversion error. Raise --steps, or accept it as the")
        print("     baseline that BiFM is meant to improve.")
    Image.fromarray(rt["recon"]).save(OUT / "roundtrip.png")
    Image.fromarray(img).save(OUT / "source.png")

    # --- 3 ---------------------------------------------------------------
    mode = ("lambda+alpha" if not (args.no_alpha or args.no_lambda)
            else "lambda only" if args.no_alpha
            else "alpha only" if args.no_lambda else "neither")
    print(f"\n--- edits across band radii [{mode}] ---")
    if not args.no_alpha:
        print("  NOTE: the alpha pin is on, so 'background' is copied from the")
        print("        source and is near-zero by construction. Re-run with")
        print("        --no-alpha to see what the lambda field does alone.")
    lat = ed.encode_image(img)
    inv = ed.invert(lat, args.source, steps=args.steps,
                    sigma_min=args.sigma_min, sigma_max=args.sigma_max)

    for r in (0.0, 1.0, 2.0, 4.0):
        cfg_l, cfg_t = paired_configs(r, lam_in=4.0, lam_out=-2.0)
        lam = build_region_field(mask, cfg_l)
        tau = build_retention_field(mask, cfg_t)
        out = ed.generate(
            inv, target_prompt, source_prompt=args.source,
            lam_field=None if args.no_lambda else lam,
            tau_field=None if args.no_alpha else tau,
            steps=args.steps, guidance=5.0,
        )
        Image.fromarray(out).save(OUT / f"edit_r{r:g}.png")

        d = np.abs(img.astype(np.float32) - out.astype(np.float32)).mean(2) / 255.0
        inside = float(d[mask].mean())
        outside = float(d[~mask].mean())
        line = (f"  r={r:<4} target {inside:.4f}  background {outside:.4f}  "
                f"ratio {inside / max(outside, 1e-6):6.1f}x")
        if others:
            oth = float(np.mean([d[m].mean() for m in others]))
            sel = inside / max(inside + oth, 1e-9)
            line += f"  others {oth:.4f}  selectivity {sel:.2f}"
        print(line)

    print(f"\nwrote images to {OUT}")
    if others:
        print(
            "\nselectivity 1.00 = only the selected object changed; 0.50 = the\n"
            "other two changed just as much (instance leakage)."
        )


if __name__ == "__main__":
    main()
