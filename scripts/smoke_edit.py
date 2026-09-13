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


def rt_ceiling(ed, img):
    """VAE-only PSNR -- the best any round trip can do on this image."""
    a = np.asarray(img, np.float32) / 255.0
    b = np.asarray(ed.decode_latent(ed.encode_image(img)), np.float32) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", metavar="PATH",
                    help="Your own photograph. There is no default image and "
                         "no file shipped with the project -- supply a real "
                         "photo with several similar objects, textured "
                         "background and natural lighting. Omit to use the "
                         "built-in synthetic test image instead.")
    ap.add_argument("--preview-box", action="store_true",
                    help="Draw the box and SAM mask on the resized image and "
                         "exit, so you can check the coordinates before "
                         "spending a full run on a wrong box.")
    ap.add_argument("--box", nargs=4, type=int)
    ap.add_argument("--box-space", choices=("auto", "original", "resized"),
                    default="auto",
                    help="Coordinate space of --box. 'original' = pixels of "
                         "the source file, 'resized' = pixels of the --size "
                         "square actually fed to the model. 'auto' guesses "
                         "from magnitude, which is WRONG for any box that "
                         "fits inside --size but was measured on a larger "
                         "photo. Say which one you mean.")
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
    ap.add_argument("--solver", choices=["euler", "heun"], default="heun",
                    help="ODE integrator. Euler is O(h) and 1 model call per "
                         "step; Heun is O(h^2) and 2. Inversion must recover "
                         "ONE specific trajectory, so truncation error turns "
                         "into a different image rather than a slightly worse "
                         "one -- which is why Euler survived the flat "
                         "synthetic and failed on a real photograph.")
    ap.add_argument("--solver-sweep", action="store_true",
                    help="Compare euler and heun at matched NFE. This is the "
                         "test that decides whether the real-image failure is "
                         "ordinary discretisation error or something "
                         "structural.")
    ap.add_argument("--sam", action="store_true",
                    help="Turn --box into a SAM object mask instead of using "
                         "the rectangle. On real photographs a box mask puts "
                         "the boundary in arbitrary background, which the seam "
                         "metrics will read as a huge artefact regardless of r.")
    ap.add_argument("--sam-checkpoint", default="/data/torch/sam_vit_h_4b8939.pth")
    ap.add_argument("--find-instances", metavar="CATEGORY",
                    help="Auto-find other objects of this category for the "
                         "leakage metric, e.g. 'apple'. Needs --sam.")
    ap.add_argument("--softness", type=float, default=0.08,
                    help="Temporal width of the alpha pin's handover, in units "
                         "of sigma. This is a CONFOUND for the r-sweep: a soft "
                         "handover smooths the pin in time, which smooths it in "
                         "space too, so it may be doing the seam prevention "
                         "that r is credited with. Set 0 for a hard pin -- if "
                         "seams appear at r=0 and vanish as r grows, the band "
                         "is real; if softness alone prevents them, r is "
                         "redundant.")
    ap.add_argument("--dilate-mask", type=int, default=0,
                    help="Grow the mask by N pixels so its boundary sits in "
                         "FLAT background rather than on the object edge. The "
                         "seam metrics read ~0 when the boundary coincides "
                         "with a real edge (the source already has one there), "
                         "so an object-tight mask cannot show whether the band "
                         "does anything. 15-20 px is a realistic imprecise mask.")
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
    from bench.metrics import boundary_metrics
    from bench.inversion import SD3EditPipeline
    from region.alpha_field import build_retention_field, paired_configs
    from region.lambda_field import build_region_field, target_token_area

    if args.image:
        img = np.array(Image.open(args.image).convert("RGB").resize((args.size,) * 2))
        if not args.box:
            ap.error("--image requires --box x0 y0 x1 y1")

        # Boxes are read off the ORIGINAL photo but the image is resized to
        # --size first, so coordinates routinely exceed the frame. Rescale
        # rather than silently clipping to an empty or wrong region.
        orig_w, orig_h = Image.open(args.image).size
        x0, y0, x1, y1 = args.box

        # The old magnitude heuristic (rescale iff max(box) > size) silently
        # does the wrong thing for any box measured on the original photo that
        # happens to fit inside --size: a small object near the top-left of a
        # 4000px image gets used verbatim in 512-space and lands elsewhere,
        # with no warning. Decide explicitly and always say which branch ran.
        space = args.box_space
        if space == "auto":
            space = "original" if max(args.box) > args.size else "resized"
            if (orig_w, orig_h) != (args.size, args.size):
                print(f"WARNING: --box-space auto guessed '{space}' from "
                      f"magnitude. Source is {orig_w}x{orig_h}, model frame is "
                      f"{args.size}px. Pass --box-space explicitly.")
        if space == "original":
            sx, sy = args.size / orig_w, args.size / orig_h
            x0, y0 = int(x0 * sx), int(y0 * sy)
            x1, y1 = int(x1 * sx), int(y1 * sy)
            print(f"box rescaled from original {orig_w}x{orig_h} to {args.size}px: "
                  f"{tuple(args.box)} -> ({x0}, {y0}, {x1}, {y1})")
        else:
            print(f"box used as-is in {args.size}px model space: "
                  f"({x0}, {y0}, {x1}, {y1})")
        x0, x1 = sorted((max(0, x0), min(args.size, x1)))
        y0, y1 = sorted((max(0, y0), min(args.size, y1)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            ap.error(f"box ({x0},{y0},{x1},{y1}) is empty after clipping to "
                     f"{args.size}px -- check the coordinates")
        others = []

        # Written on EVERY run, not just --preview-box. Previously the preview
        # was produced only by preview runs, which exit early -- so a stale
        # box_preview.png from an older invocation sat in the output directory
        # beside fresh edit_r*.png showing a different region entirely.
        prev = img.copy()
        prev[y0:y1, x0:x1] = (prev[y0:y1, x0:x1] * 0.6 + np.array([255, 0, 0]) * 0.4)
        Image.fromarray(prev.astype(np.uint8)).save(OUT / "box_preview.png")
        print(f"box preview -> {OUT/'box_preview.png'}")

        # A stale SAM overlay is the same trap. Remove it unless this run
        # regenerates one.
        stale = OUT / "mask_preview.png"
        if not args.sam and stale.exists():
            stale.unlink()
            print(f"removed stale {stale.name} (this run is not using --sam)")

        if args.preview_box:
            if args.sam:
                from region.segment import SAMSegmenter
                seg = SAMSegmenter(args.sam_checkpoint)
                m = seg.mask_from_box(img, (x0, y0, x1, y1))
                ov = img.copy()
                ov[m] = (ov[m] * 0.5 + np.array([0, 255, 0]) * 0.5)
                Image.fromarray(ov.astype(np.uint8)).save(OUT / "mask_preview.png")
                print(f"SAM mask covers {m.mean():.2%} of image "
                      f"-> {OUT/'mask_preview.png'}")
            return

        if args.sam:
            from region.segment import SAMSegmenter, find_similar_instances
            seg = SAMSegmenter(args.sam_checkpoint)
            mask = seg.mask_from_box(img, (x0, y0, x1, y1))
            print(f"SAM mask: {mask.mean():.2%} of image "
                  f"(box was {((x1-x0)*(y1-y0))/args.size**2:.2%})")
            if args.find_instances:
                from region.segment import make_clip_scorer
                others = find_similar_instances(
                    img, mask, args.find_instances, seg.sam,
                    clip_scorer=make_clip_scorer(),
                )
                print(f"found {len(others)} other '{args.find_instances}' instances")
        else:
            mask = np.zeros((args.size, args.size), bool)
            mask[y0:y1, x0:x1] = True
            print("NOTE: using the raw box as the mask. Its boundary runs "
                  "through background, so seam metrics will be dominated by "
                  "that rather than by r. Use --sam for a real object mask.")
    else:
        img, all_masks = synthetic(args.size)
        mask = all_masks[1]                     # the MIDDLE object only
        others = [all_masks[0], all_masks[2]]

    if args.dilate_mask > 0:
        from scipy.ndimage import distance_transform_edt as _edt
        tight = mask.copy()
        mask = _edt(~tight) <= args.dilate_mask
        print(f"mask dilated by {args.dilate_mask}px: "
              f"{tight.mean():.1%} -> {mask.mean():.1%} of image")
        # Non-target objects must not be swallowed by the dilation, or
        # selectivity becomes meaningless.
        for i, o in enumerate(others):
            if (mask & o).mean() > 0.001:
                print(f"  !! dilated mask overlaps other object {i} -- "
                      f"reduce --dilate-mask")

    target_prompt, spans = parse_brackets(args.target)

    # Inversion is conditioned on the source prompt and generation on the
    # target; the token diff only makes sense if they share structure. Two
    # unrelated sentences mark nearly every token as an edit token, so the
    # lambda field has nothing specific to steer.
    sw, tw = args.source.split(), target_prompt.split()
    shared = len(set(sw) & set(tw)) / max(len(set(sw) | set(tw)), 1)
    if shared < 0.4:
        print(f"  !! source and target prompts share only {shared:.0%} of words.")
        print("     Keep the sentence identical and change ONLY the edited")
        print("     words, e.g. 'many blue cans on a wooden shelf' ->")
        print("     'many [red] cans on a wooden shelf'. Rewriting the whole")
        print("     sentence marks everything as an edit token.")

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
    if args.solver_sweep:
        print("  Matched-NFE comparison. If heun-50 beats euler-100 at equal")
        print("  model calls, the failure was discretisation order and this")
        print("  unblocks real images. If both stay poor, more steps will")
        print("  never fix it and the problem is elsewhere.")
        print(f"    {'solver':>8}{'steps':>7}{'NFE':>6}{'PSNR':>8}{'noise std':>11}")
        for solver, n in [("euler", 50), ("euler", 100), ("euler", 200),
                          ("heun", 25), ("heun", 50), ("heun", 100)]:
            r = ed.roundtrip_error(img, args.source, steps=n,
                                   sigma_min=args.sigma_min,
                                   sigma_max=args.sigma_max, solver=solver)
            print(f"    {solver:>8}{n:>7}{r['nfe']:>6}{r['roundtrip_psnr']:>8.2f}"
                  f"{r['noise_std']:>11.3f}")
        print(f"\n  VAE ceiling for this image: {rt_ceiling(ed, img):.2f} dB")
        return

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
                            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                            solver=args.solver)
    print(f"  VAE only        {rt['vae_only_psnr']:6.2f} dB   <- ceiling")
    print(f"  full round trip {rt['roundtrip_psnr']:6.2f} dB")
    print(f"  noise std       {rt['noise_std']:6.3f}  (solver={args.solver}, "
          f"{rt['nfe']} model calls)")
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
    print(f"\n--- edits across band radii [{mode}, softness={args.softness}] ---")
    if not args.no_alpha:
        print("  NOTE: the alpha pin is on, so 'background' is copied from the")
        print("        source and is near-zero by construction. Re-run with")
        print("        --no-alpha to see what the lambda field does alone.")
    lat = ed.encode_image(img)
    inv = ed.invert(lat, args.source, steps=args.steps,
                    sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                    solver=args.solver)

    for r in (0.0, 1.0, 2.0, 4.0):
        cfg_l, cfg_t = paired_configs(r, lam_in=4.0, lam_out=-2.0)
        lam = build_region_field(mask, cfg_l)
        tau = build_retention_field(mask, cfg_t)
        out = ed.generate(
            inv, target_prompt, source_prompt=args.source,
            lam_field=None if args.no_lambda else lam,
            tau_field=None if args.no_alpha else tau,
            steps=args.steps, guidance=5.0, softness=args.softness,
            solver=args.solver,
        )
        Image.fromarray(out).save(OUT / f"edit_r{r:g}.png")

        d = np.abs(img.astype(np.float32) - out.astype(np.float32)).mean(2) / 255.0
        inside = float(d[mask].mean())
        outside = float(d[~mask].mean())
        bm = boundary_metrics(img, out, mask)
        line = (f"  r={r:<4} target {inside:.4f}  bg {outside:.4f}  "
                f"width {bm['transition_width']:5.2f}  "
                f"seam {bm['seam_excess']:7.2f}")
        if others:
            oth = float(np.mean([d[m].mean() for m in others]))
            sel = inside / max(inside + oth, 1e-9)
            line += f"  others {oth:.4f}  selectivity {sel:.2f}"
        print(line)

    print(f"\nwrote images to {OUT}")
    print(
        "\n  width = pixels the change takes to fall from 90% to 10% at the\n"
        "          boundary. This is what r directly controls, so it is the\n"
        "          one number that must rise with r for the band to be real.\n"
        "  seam  = contour-following edge the edit ADDED, relative to the\n"
        "          source. Reads ~0 when the mask boundary already sits on an\n"
        "          object edge -- use --dilate-mask to move it off one."
    )
    if others:
        print(
            "\nselectivity 1.00 = only the selected object changed; 0.50 = the\n"
            "other two changed just as much (instance leakage)."
        )


if __name__ == "__main__":
    main()
