#!/usr/bin/env python3
"""Sweep tau_target: does lowering regeneration depth fix the phantom edits?

WHY THIS EXISTS

The Stage 0 run produced 4/12 formal passes whose images were unusable. Inside
the mask, saturation collapsed (ball 0.619 -> 0.087) while the image gradient
ROSE (1.80 -> 4.77). Saturation down with gradient up is not a recolour; it is
a region replaced by high-frequency noise. The ball's violet dot vanished, and
on the hair case the subject's face was erased and refilled with texture.

The cause is a default nobody swept. RetentionConfig.tau_target is 1.0, and
its own docstring says:

    tau_target: 1.0 = full regeneration. Lower it for attribute edits where
                the silhouette should be retained.

tau is the time at which a token stops tracking the inverted trajectory and
starts generating freely (see pin_weights). tau_target = 1.0 frees the target
tokens IMMEDIATELY, at t = 1, i.e. from pure noise. So the pipeline was never
recolouring the ball. It was deleting the ball and asking SD3 to draw a new one
from a caption, inside a mask, with no structural conditioning. Phantom objects
are the correct output of that procedure.

A colour edit wants the opposite: keep the silhouette, texture and shading, and
change only chroma. That means releasing the target tokens LATE -- low
tau_target -- so they inherit structure from the source trajectory and only the
last part of the path is free to move colour.

WHAT THIS PREDICTS

If the diagnosis is right, as tau_target falls from 1.0:
    structure    rises   (luminance correlation with source, inside mask)
    saturation   recovers toward the source
    hue          still shifts toward the target colour
and there is a middle band where structure is high AND hue has moved. If no
such band exists -- structure and hue trade off monotonically with no window --
then latent-space depth control cannot express "change colour, keep form", and
that is a real negative result about the method, not a tuning failure.

METRICS

  struct   Pearson correlation of source vs edit luminance inside the mask.
           1.0 = form perfectly retained. This is the number the previous run
           was missing; PSNR and mean-abs-difference cannot see it.
  sat      mean saturation inside the mask, against the source value.
  dhue     mean circular hue rotation inside the mask, in degrees. A colour
           edit should move this substantially.
  bg       mean abs change outside the mask. Must stay low or the pin broke.
  clip     CLIP(target caption) on the edited crop minus on the source crop.

Run the Stage 0 script first; this reuses its captions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench.data import load_magicbrush
from bench.inversion import SD3EditPipeline
from bench.metrics import boundary_metrics
from region.alpha_field import RetentionConfig, build_retention_field
from region.lambda_field import RegionFieldConfig, build_region_field

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"


def parse_brackets_local(target: str):
    import re

    spans = re.findall(r"\[([^\]]+)\]", target)
    return re.sub(r"[\[\]]", "", target), spans


def mask_to_box(m, pad=4):
    ys, xs = np.where(m)
    h, w = m.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def luma(a):
    a = np.asarray(a, np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sat_of(a):
    a = np.asarray(a, np.float32)
    mx, mn = a.max(2), a.min(2)
    return np.where(mx > 1, (mx - mn) / np.maximum(mx, 1), 0.0)


def hue_of(a):
    """Hue in radians. Uses the standard HSV formula, undefined where sat=0."""
    import colorsys  # noqa: F401  (kept for reference; vectorised below)

    a = np.asarray(a, np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    d = np.maximum(mx - mn, 1e-6)
    h = np.zeros_like(mx)
    i = mx == r
    h[i] = ((g - b)[i] / d[i]) % 6
    i = mx == g
    h[i] = (b - r)[i] / d[i] + 2
    i = mx == b
    h[i] = (r - g)[i] / d[i] + 4
    return h * np.pi / 3.0


def circular_hue_shift(src, edit, m):
    """Mean rotation in degrees, weighted by saturation so grey pixels -- whose
    hue is meaningless -- do not dominate the average."""
    hs, he = hue_of(src)[m], hue_of(edit)[m]
    w = np.minimum(sat_of(src)[m], sat_of(edit)[m])
    if w.sum() < 1e-6:
        return float("nan")
    d = np.angle(np.exp(1j * (he - hs)))
    return float(np.degrees(np.average(d, weights=w)))


def to_lab(a):
    """sRGB uint8 -> CIE L*a*b* (D65). Vectorised, no skimage dependency.

    Needed because `dhue` measures a channel some edits do not live in. The
    sheep case is white -> black: a pure lightness move with source saturation
    0.164, already at the floor. Hue rotation there is noise, and the grid
    reported it as if it were edit signal.
    """
    a = np.asarray(a, np.float32) / 255.0
    m = a <= 0.04045
    lin = np.where(m, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    def f(t):
        d = 6.0 / 29.0
        return np.where(t > d ** 3, np.cbrt(np.maximum(t, 1e-12)),
                        t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def lab_deltas(src, edit, m):
    """(dL, dE) inside the mask.

    dL  signed mean lightness change. Negative = darker. This is the axis a
        white->black edit actually moves along.
    dE  mean CIE76 distance. One number covering chroma AND lightness, so a
        grid mixing colour directions can be read on a single column. It does
        NOT distinguish 'moved to the requested colour' from 'moved anywhere' --
        that is what CLIP and the contact sheet are for.
    """
    ls, le = to_lab(src), to_lab(edit)
    d = le - ls
    return float(d[..., 0][m].mean()), float(np.linalg.norm(d, axis=-1)[m].mean())


def struct_corr(src, edit, m):
    """Correlation of local EDGE ENERGY inside the mask.

    The obvious version -- correlate raw luminance -- is wrong here. Luminance
    weights green at 0.587 and blue at 0.114, so an honest green->blue recolour
    inverts the luminance ordering and scores -1.0 despite preserving every
    edge. Validated on synthetic pairs: raw luminance gave -1.000 for a correct
    recolour, gradient magnitude gives ~1.0. Gradient magnitude is what "did
    the form survive" actually means, and it is sign-invariant.

    1.0  edges preserved (recoloured object)
    ~0   edges unrelated to the source (phantom fill)
    0.0  returned when either side is flat, which is itself a failure
    """
    def edges(a):
        L = luma(a)
        gx = np.zeros_like(L)
        gy = np.zeros_like(L)
        gx[:, 1:] = np.abs(np.diff(L, axis=1))
        gy[1:, :] = np.abs(np.diff(L, axis=0))
        return np.hypot(gx, gy)

    a, b = edges(src)[m], edges(edit)[m]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def contact_sheet(panels, labels, path, size=384):
    from PIL import ImageDraw

    ims = [Image.fromarray(p).resize((size, size), Image.LANCZOS) for p in panels]
    sheet = Image.new("RGB", (size * len(ims), size + 22), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        sheet.paste(im, (i * size, 22))
        d.text((i * size + 4, 5), lab[:64], fill=(0, 0, 0))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/datasets/MagicBrush")
    ap.add_argument("--out", default="/data/outputs/tau_sweep")
    ap.add_argument("--sam", default="/data/torch/sam_vit_h_4b8939.pth")
    ap.add_argument("--captions", default="/data/outputs/stage0/captions.json")
    ap.add_argument("--case", default="17320_t1",
                    help="single case. Default is the best-conditioned one: "
                         "31.2 dB reconstruction, SAM IoU 0.822, compact "
                         "object, and a violet dot that survives only if the "
                         "edit is genuinely local.")
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--taus", default="0.2,0.35,0.5,0.7,1.0")
    ap.add_argument("--lam-ins", default="4.0",
                    help="comma list of lam_in values. The attention bias "
                         "added to (image token -> edit token) logits inside "
                         "the target. Never swept before this run; 4.0 was "
                         "the untested default and produced dhue ~0 at every "
                         "tau where structure survived.")
    ap.add_argument("--lam-out", type=float, default=-2.0)
    ap.add_argument("--tau-min", type=float, default=0.1)
    ap.add_argument("--softness", type=float, default=0.08)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--solver", default="heun")
    ap.add_argument("--no-sam", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taus = [float(t) for t in args.taus.split(",")]

    caps = {k: tuple(v) for k, v in
            json.loads(Path(args.captions).read_text()).items()
            if not k.startswith("_")}
    if args.case not in caps:
        print(f"{args.case} has no caption in {args.captions}. "
              f"Available: {sorted(caps)}")
        return

    cases = load_magicbrush(args.data, limit=args.pool)
    by_id = {c.case_id: c for c in cases}
    if args.case not in by_id:
        print(f"{args.case} not in the first {args.pool} cases; raise --pool")
        return
    c = by_id[args.case]

    src_cap, tgt_raw = caps[args.case]
    tgt_cap, spans = parse_brackets_local(tgt_raw)
    if not spans:
        print("target caption has no [brackets]; nothing to diff")
        return
    print(f"{args.case}\n  source: {src_cap}\n  target: {tgt_cap}  "
          f"(edit spans {spans})")

    # ---- phase 1: SAM, then release ---------------------------------------
    if args.no_sam:
        mask = c.target_mask
        print(f"  MagicBrush mask {mask.mean():.2%}")
    else:
        from region.segment import SAMSegmenter

        seg = SAMSegmenter(args.sam)
        mask = seg.mask_from_box(c.image, mask_to_box(c.target_mask))
        inter = (mask & c.target_mask).sum()
        union = (mask | c.target_mask).sum()
        print(f"  SAM mask {mask.mean():.2%}, IoU vs GT {inter/max(union,1):.3f}")
        del seg
        torch.cuda.empty_cache()

    Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"mask_{args.case}.png")

    # ---- phase 2: diffusion ------------------------------------------------
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda")
    ed = SD3EditPipeline(pipe)

    clip = None
    try:
        from region.segment import make_clip_scorer

        clip = make_clip_scorer(device="cuda")
    except Exception as e:  # noqa: BLE001
        print(f"  [clip] unavailable ({type(e).__name__})")

    box = mask_to_box(c.target_mask)
    src = c.image
    src_sat = float(sat_of(src)[mask].mean())
    src_grad = float(np.abs(np.diff(luma(src), axis=1))[mask[:, :-1]].mean())

    rt = ed.roundtrip_error(src, src_cap, steps=args.steps, solver=args.solver)
    print(f"  reconstruction {rt['roundtrip_psnr']:.2f} dB "
          f"(ceiling {rt['vae_only_psnr']:.2f})")
    print(f"  source in-mask: saturation {src_sat:.3f}  gradient {src_grad:.2f}")

    lat = ed.encode_image(src)
    inv = ed.invert(lat, src_cap, steps=args.steps, solver=args.solver)

    lam_ins = [float(v) for v in args.lam_ins.split(",")]

    print(f"\n  softness={args.softness}  guidance={args.guidance}  "
          f"r={args.radius}  solver={args.solver}-{args.steps}")
    print(f"  {'tau':>5} {'lam_in':>7} {'struct':>7} {'sat':>7} {'dhue':>8} "
          f"{'dL*':>7} {'dE':>6} {'grad':>7} {'bg':>7} {'clip':>8}")

    panels, labels, rows = [src], ["source"], []
    for lam_in in lam_ins:
      lam = build_region_field(
          mask, RegionFieldConfig(band_tokens=args.radius,
                                  lam_in=lam_in, lam_out=args.lam_out))
      for tau_t in taus:
        tau = build_retention_field(
            mask, RetentionConfig(band_tokens=args.radius,
                                  tau_min=args.tau_min,
                                  tau_target=tau_t,
                                  softness=args.softness))
        edit = ed.generate(
            inv, tgt_cap, source_prompt=src_cap,
            lam_field=lam, tau_field=tau,
            steps=args.steps, guidance=args.guidance,
            softness=args.softness, solver=args.solver,
        )
        Image.fromarray(edit).save(out / f"edit_tau{tau_t:g}_lam{lam_in:g}.png")

        st = struct_corr(src, edit, mask)
        sa = float(sat_of(edit)[mask].mean())
        dh = circular_hue_shift(src, edit, mask)
        dL, dE = lab_deltas(src, edit, mask)
        gr = float(np.abs(np.diff(luma(edit), axis=1))[mask[:, :-1]].mean())
        d = np.abs(src.astype(np.float32) - edit.astype(np.float32)).mean(2) / 255.0
        bg = float(d[~mask].mean())
        cl = None
        if clip is not None:
            x0, y0, x1, y1 = box
            cl = float(clip(edit[y0:y1, x0:x1], tgt_cap)
                       - clip(src[y0:y1, x0:x1], tgt_cap))
        bm = boundary_metrics(src, edit, mask)

        print(f"  {tau_t:>5.2f} {lam_in:>7.1f} {st:>7.3f} {sa:>7.3f} "
              f"{dh:>7.1f}d {dL:>+7.2f} {dE:>6.2f} {gr:>7.2f} {bg:>7.4f} "
              + (f"{cl:>+8.4f}" if cl is not None else "     n/a"))
        rows.append(dict(tau_target=tau_t, lam_in=lam_in, struct=st, sat=sa,
                         dhue=dh, dL=dL, dE=dE, grad=gr, bg=bg, clip=cl,
                         seam=bm["seam_excess"],
                         src_sat=src_sat, src_grad=src_grad))
        panels.append(edit)
        labels.append(f"t={tau_t:g} L={lam_in:g} st={st:.2f} dE={dE:.0f}")

    contact_sheet(panels, labels, out / f"sheet_tau_{args.case}.png")
    (out / "results.json").write_text(json.dumps(rows, indent=2))

    # ---- read the sweep ----------------------------------------------------
    print("\n  reference: source saturation %.3f, gradient %.2f" % (src_sat, src_grad))
    print("  (struct thresholds below are UNCALIBRATED. On synthetic pairs the "
          "metric\n   scored 0.27 for a correct recolour, 0.02 for a noise "
          "fill, 0.00 for a flat\n   blob -- ordering is right, absolute scale "
          "is not established. Rank by it,\n   do not trust the pass/fail.)")
    # Keyed on dE, not dhue: dE covers chroma and lightness edits alike, so a
    # white->black case is scored on the axis it actually moves along.
    good = [r for r in rows
            if r["struct"] > 0.30 and r["dE"] > 12.0
            and r["sat"] > 0.6 * src_sat]
    if good:
        b = max(good, key=lambda r: r["dE"])
        print(f"  WINDOW EXISTS: tau_target={b['tau_target']:g} "
              f"lam_in={b['lam_in']:g} keeps form (struct {b['struct']:.3f}) "
              f"while moving colour dE={b['dE']:.1f} (dL* {b['dL']:+.1f}, "
              f"dhue {b['dhue']:.0f} deg) and holding saturation at "
              f"{b['sat']:.3f}. This is the operating point.")
    else:
        hi = [r for r in rows if r["struct"] > 0.30]
        mv = [r for r in rows if r["dE"] > 12.0]
        if hi and not mv:
            print("  NO WINDOW: structure is retained but hue never moves. "
                  "alpha is holding the region verbatim and lambda is not "
                  "strong enough to move colour against it. Raise --lam-ins.")
        elif mv and not hi:
            print("  NO WINDOW: hue moves but structure is always destroyed, "
                  "at every depth. Depth control cannot express 'recolour, "
                  "keep form' on this image. That is a finding about the "
                  "method, not a tuning failure -- report it.")
        else:
            print("  NO WINDOW: structure and hue never co-occur. Look at the "
                  "sheet; the trade-off curve is the result.")
    lam_vals = sorted({r["lam_in"] for r in rows})
    if len(lam_vals) > 1:
        print("\n  does lambda move colour at fixed tau?  (dE = CIE76, covers chroma and lightness)")
        for t in sorted({r["tau_target"] for r in rows}):
            line = f"    tau={t:<5g}"
            for L in lam_vals:
                m = [r for r in rows
                     if r["tau_target"] == t and r["lam_in"] == L]
                if m:
                    line += (f"  L={L:g}: dE {m[0]['dE']:5.1f}"
                             f" dL* {m[0]['dL']:+6.1f}")
            print(line)
        print("    Flat rows mean the attention bias cannot move colour "
              "against the pin\n    at any strength -- which would make "
              "lambda a localisation mechanism only,\n    not an edit "
              "mechanism, and the method needs a third component.")

    print("\n  LOOK AT THE SHEET. struct is edge-energy correlation and will "
          "happily reward a ball that kept its edges but never changed hue.")
    print(f"  images -> {out}")


if __name__ == "__main__":
    main()
