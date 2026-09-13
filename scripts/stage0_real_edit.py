#!/usr/bin/env python3
"""Stage 0 deliverable: one successful edit on a REAL photograph with a SAM mask.

Everything measured so far -- softness, r, the seam mechanism, the temporal
handover story -- was measured on synthetic images or on hand-drawn boxes that
straddled objects. Not one edit has succeeded on a real photograph with a real
segmentation mask. Until that exists there is no Stage 0 result, only
infrastructure.

WHAT THIS RUNS

  cases      3 MagicBrush rows, chosen for reconstruction quality, each with a
             hand-written source/target caption pair (see CAPTIONS below --
             they are the input you must supply, and the reason this is not
             fully automatic)
  mask       SAM, seeded by the bounding box of the MagicBrush mask. The
             MagicBrush mask is NOT used as the edit mask; it is held back as
             ground truth to score SAM against.
  solver     heun, 50 steps, fixed. Settled at n=12: +3.08 dB over euler-100
             at matched NFE, 9/12 wins.
  grid       softness {0.0, 0.08} x guidance {2.5, 5.0}, r=0
             12 runs total. r stays 0 because r has never improved
             selectivity in any measurement and increases spill.

WHY THESE AXES

  softness   fills the empty row of the experiment matrix. The temporal
             handover result -- softness, not r, suppresses seams -- was
             measured on synthetic colour edits with exact masks. This asks
             whether it survives a real photo and an imprecise mask. It is
             the one axis with a prior prediction, so it is the one that can
             actually be wrong.
  guidance   never swept. Every edit so far ran at the default 5.0. The
             reconstruction error on real photographs is low-frequency
             structural drift, and CFG amplifies whatever the trajectory is
             doing, so 5.0 may be actively hurting.

WHY COLOUR AND NOT SHAPE

No shape edit has succeeded on any image at any setting, synthetic included.
Attempting one here would confound "region control does not work on real
photos" with "shape edits do not work at all". Land colour first.

SUCCESS IS DECLARED BEFORE THE RUN, NOT AFTER

The failure mode of this whole session has been reading a verdict off numbers
that did not support it. So the criteria are fixed here, in code, and the
script prints PASS/FAIL against them rather than leaving it to the eye:

  1. target changed      mean |delta| inside mask   > 0.06
  2. background held     mean |delta| outside mask  < 0.02
  3. contrast            inside / outside           > 4.0
  4. semantics           the edit is the RIGHT change, not just any change.
                         Scored by CLIP against the target caption if
                         available, otherwise flagged for human review.

Criteria 1-3 can all pass while the edit is wrong -- a grey smear inside the
mask satisfies them. Criterion 4 is the one that matters and it is the one a
metric cannot fully settle. Contact sheets are written for exactly that reason.
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
from region.alpha_field import build_retention_field, paired_configs
from region.lambda_field import build_region_field

MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"

# ---------------------------------------------------------------------------
# THE CAPTIONS ARE THE INPUT. EDIT THEM.
#
# MagicBrush ships an instruction ("make the sky purple"), not a caption pair,
# and the lambda field needs a pair to diff for edit tokens. Prompt mismatch
# was measured at 3.65 dB on can.jpg -- larger than the entire euler->heun
# gap -- so a wrong caption here will dominate everything else in the run.
#
# Write these by LOOKING at the image. Run with --list to dump the candidate
# images and their instructions, write captions, then run for real.
#
# Bracket the changed span in the target, as PIE-Bench does: "[red]".
# ---------------------------------------------------------------------------
CAPTIONS = {
    # "140513_t1": (
    #     "a green mountain bicycle leaning against a brick wall",
    #     "a [red] mountain bicycle leaning against a brick wall",
    # ),
}


def parse_brackets_local(target: str):
    """'a [red] car' -> ('a red car', ['red'])."""
    import re

    spans = re.findall(r"\[([^\]]+)\]", target)
    return re.sub(r"[\[\]]", "", target), spans


def mask_to_box(m: np.ndarray, pad: int = 4):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    h, w = m.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def masked_psnr(a, b, m=None):
    """PSNR restricted to a region.

    Global roundtrip PSNR overstates the damage the EDIT pays. The alpha pin
    copies source latents into the preserved region, so the background never
    travels through the drifting trajectory -- which is why edits localise
    correctly even at 15 dB global reconstruction. What the edit actually pays
    is the reconstruction error INSIDE the mask. Report both.
    """
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    e = ((a - b) ** 2).mean(2)
    e = e.mean() if m is None else e[m].mean()
    return float("inf") if e < 1e-9 else float(10 * np.log10(255.0 ** 2 / e))


def load_clip(device="cuda"):
    """Optional. Returns score(crop_rgb_uint8, text) -> float, or None."""
    try:
        from region.segment import make_clip_scorer

        return make_clip_scorer(device=device)
    except Exception as e:  # noqa: BLE001
        print(f"  [clip] unavailable ({type(e).__name__}); criterion 4 will "
              f"need human review of the contact sheets")
        return None


def contact_sheet(panels, labels, path, size=384):
    """Horizontal strip with captions burned in, for eyeballing criterion 4."""
    from PIL import ImageDraw

    ims = [Image.fromarray(p).resize((size, size), Image.LANCZOS) for p in panels]
    sheet = Image.new("RGB", (size * len(ims), size + 22), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        sheet.paste(im, (i * size, 22))
        d.text((i * size + 4, 5), lab[:60], fill=(0, 0, 0))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/datasets/MagicBrush")
    ap.add_argument("--out", default="/data/outputs/stage0")
    ap.add_argument("--sam", default="/data/torch/sam_vit_h_4b8939.pth")
    ap.add_argument("--pool", type=int, default=12,
                    help="how many MagicBrush cases to consider")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--solver", default="heun")
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--list", action="store_true",
                    help="dump candidate images + instructions and exit, so "
                         "captions can be written")
    ap.add_argument("--captions", default=None,
                    help="JSON file {case_id: [source, target]}. Preferred "
                         "over editing CAPTIONS in this file.")
    ap.add_argument("--colour-only", action="store_true",
                    help="keep only colour/material/attribute instructions. "
                         "Add/remove/replace edits have never succeeded at "
                         "any setting, so including them confounds 'region "
                         "control fails on real photos' with 'shape edits "
                         "fail everywhere'.")
    ap.add_argument("--no-sam", action="store_true",
                    help="fall back to the MagicBrush mask. This does NOT "
                         "satisfy the deliverable; it isolates SAM as a "
                         "variable if the SAM run fails.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = load_magicbrush(args.data, limit=args.pool)

    if args.colour_only:
        kw = ("color", "colour", "red", "orange", "yellow", "green", "blue",
              "purple", "pink", "brown", "black", "white", "grey", "gray",
              "golden", "silver", "paint", "wooden", "metal", "rusty",
              "brighter", "darker")
        keep = [c for c in cases
                if any(k in c.target_prompt.lower() for k in kw)]
        print(f"[colour filter] {len(keep)}/{len(cases)} instructions look "
              f"like colour/material edits")
        if not keep:
            print("  none in this pool. Raise --pool (MagicBrush dev has "
                  "~500 turn-1 rows) and re-run --list.")
            return
        cases = keep

    by_id = {c.case_id: c for c in cases}

    captions = dict(CAPTIONS)
    if args.captions:
        loaded = json.loads(Path(args.captions).read_text())
        captions.update({k: tuple(v) for k, v in loaded.items()})

    # ---- --list: write the images out so captions can be written ----------
    if args.list:
        (out / "candidates").mkdir(exist_ok=True)
        print("\ncandidates -- open these, write a caption pair for each, "
              "paste into CAPTIONS at the top of this file:\n")
        for c in cases:
            Image.fromarray(c.image).save(out / "candidates" / f"{c.case_id}.png")
            print(f'    "{c.case_id}": (')
            print(f'        "",   # source caption: describe the photo as it IS')
            print(f'        "",   # target: same caption, changed span in [brackets]')
            print(f'    ),        # instruction was: {c.target_prompt!r}')
        skel = {c.case_id: ["", ""] for c in cases}
        (out / "captions.json").write_text(json.dumps(skel, indent=2))
        print(f"\nimages   -> {out/'candidates'}")
        print(f"skeleton -> {out/'captions.json'}   "
              f"(fill it in, then pass --captions)")
        print("\nWrite the SOURCE caption by looking at the photo and "
              "describing what is there.\nDo not paraphrase the instruction: "
              "'A wide landscape view' is not a caption,\nand a wrong caption "
              "was measured at 3.65 dB -- larger than the solver effect.")
        return

    have = [cid for cid in captions if cid in by_id and all(captions[cid])]

    # Validate BEFORE loading 15 GB of weights. A target with no [brackets]
    # yields no edit spans, so the lambda field is empty and the run silently
    # measures the alpha pin alone -- which looks like a result and is not one.
    bad = []
    for cid in list(have):
        src, tgt = captions[cid]
        _, spans = parse_brackets_local(tgt)
        if not spans:
            bad.append((cid, "target has no [brackets] marking the changed span"))
        elif src.strip() == tgt.replace("[", "").replace("]", "").strip():
            bad.append((cid, "source and target are identical once brackets "
                             "are stripped -- nothing to diff"))
    if bad:
        print("\nunusable caption pairs:")
        for cid, why in bad:
            print(f"  {cid}: {why}")
            print(f"    source: {captions[cid][0]!r}")
            print(f"    target: {captions[cid][1]!r}")
        print("\nThe target must repeat the source with the changed span in "
              "brackets:\n"
              '    "green buds with yellow flowers"\n'
              '    "green buds with [red] flowers"\n'
              "Fix these and re-run; nothing was loaded.")
        have = [c for c in have if c not in {b[0] for b in bad}]
        if not have:
            return

    if not have:
        print("No usable captions. Run with --list first, write them, then "
              "pass --captions caps.json (or edit CAPTIONS in this file).\n"
              "This is deliberate: a caption paraphrased from the instruction "
              "rather than written from the image adds a confound already "
              "measured at 3.65 dB.")
        return
    have = have[: args.n_cases]
    print(f"running {len(have)} case(s): {have}")

    # ---- PHASE 1: all SAM masks, then release SAM -------------------------
    # SD3-Medium plus T5 plus SAM ViT-H on one 24 GB card is tight, and
    # holding both open also doubles the file handles -- the previous run died
    # with 'CUDA driver error: unknown error' alongside Errno 24 (too many open
    # files), which is fd exhaustion, not OOM. Segment everything first, free
    # SAM, then load the diffusion model.
    masks = {}
    if not args.no_sam:
        from region.segment import SAMSegmenter

        seg = SAMSegmenter(args.sam)
        for cid in have:
            c = by_id[cid]
            masks[cid] = seg.mask_from_box(c.image, mask_to_box(c.target_mask))
            print(f"  SAM {cid}: {masks[cid].mean():.3%} of image, "
                  f"IoU vs MagicBrush GT {iou(masks[cid], c.target_mask):.3f}")
        del seg
        torch.cuda.empty_cache()
        print("  SAM released")
    else:
        for cid in have:
            masks[cid] = by_id[cid].target_mask
            print(f"  {cid}: MagicBrush mask {masks[cid].mean():.3%} "
                  f"(NOT the deliverable)")

    # ---- PHASE 2: diffusion -----------------------------------------------
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda")
    ed = SD3EditPipeline(pipe)
    clip = load_clip()

    rows = []
    for cid in have:
        c = by_id[cid]
        src_cap, tgt_cap_raw = captions[cid]
        tgt_cap, spans = parse_brackets_local(tgt_cap_raw)
        gt = c.target_mask
        box = mask_to_box(gt)

        print(f"\n{'='*70}\n{cid}\n  source: {src_cap}\n  target: {tgt_cap}"
              f"   (edit spans {spans})")

        # ---- mask (segmented in phase 1) ----------------------------------
        mask = masks[cid]
        if iou(mask, gt) < 0.3 and not args.no_sam:
            print("  !! SAM disagrees with the annotation. The edit may be "
                  "targeting the wrong object; check mask_*.png before "
                  "believing any number below.")

        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"mask_{cid}.png")

        # ---- reconstruction, global and masked ----------------------------
        rt = ed.roundtrip_error(c.image, src_cap, steps=args.steps,
                                solver=args.solver)
        m_in = masked_psnr(c.image, rt["recon"], mask)
        m_out = masked_psnr(c.image, rt["recon"], ~mask)
        print(f"  reconstruction: global {rt['roundtrip_psnr']:.2f} dB   "
              f"in-mask {m_in:.2f}   out-mask {m_out:.2f}   "
              f"ceiling {rt['vae_only_psnr']:.2f}")
        Image.fromarray(rt["recon"]).save(out / f"recon_{cid}.png")

        # ---- one inversion, reused across the grid ------------------------
        lat = ed.encode_image(c.image)
        inv = ed.invert(lat, src_cap, steps=args.steps, solver=args.solver)

        cfg_l, cfg_t = paired_configs(args.radius, lam_in=4.0, lam_out=-2.0)
        lam = build_region_field(mask, cfg_l)
        tau = build_retention_field(mask, cfg_t)

        panels, labels = [c.image], ["source"]
        for softness in (0.0, 0.08):
            for guidance in (2.5, 5.0):
                edit = ed.generate(
                    inv, tgt_cap, source_prompt=src_cap,
                    lam_field=lam, tau_field=tau,
                    steps=args.steps, guidance=guidance,
                    softness=softness, solver=args.solver,
                )
                tag = f"{cid}_s{softness:g}_g{guidance:g}"
                Image.fromarray(edit).save(out / f"edit_{tag}.png")

                d = (np.abs(c.image.astype(np.float32) - edit.astype(np.float32))
                     .mean(2) / 255.0)
                inside, outside = float(d[mask].mean()), float(d[~mask].mean())
                contrast = inside / max(outside, 1e-9)
                bm = boundary_metrics(c.image, edit, mask)

                sem = None
                if clip is not None:
                    x0, y0, x1, y1 = box
                    sem = float(clip(edit[y0:y1, x0:x1], tgt_cap)
                                - clip(c.image[y0:y1, x0:x1], tgt_cap))

                ok = [inside > 0.06, outside < 0.02, contrast > 4.0]
                if sem is not None:
                    ok.append(sem > 0.0)
                verdict = "PASS" if all(ok) else "fail"

                print(f"  s={softness:<5g} g={guidance:<4g} "
                      f"in {inside:.4f} out {outside:.4f} "
                      f"contrast {contrast:5.1f} seam {bm['seam_excess']:6.2f}"
                      + (f" clip {sem:+.4f}" if sem is not None else "")
                      + f"   {verdict} {''.join('.x'[not o] for o in ok)}")

                rows.append(dict(case=cid, softness=softness, guidance=guidance,
                                 inside=inside, outside=outside,
                                 contrast=contrast, seam=bm["seam_excess"],
                                 transition=bm["transition_width"],
                                 clip_delta=sem, verdict=verdict,
                                 recon_global=rt["roundtrip_psnr"],
                                 recon_in_mask=m_in))
                panels.append(edit)
                labels.append(f"s={softness:g} g={guidance:g} {verdict}")

        contact_sheet(panels, labels, out / f"sheet_{cid}.png")

    (out / "results.json").write_text(json.dumps(rows, indent=2))

    # ---- summary ----------------------------------------------------------
    passes = [r for r in rows if r["verdict"] == "PASS"]
    print(f"\n{'='*70}")
    print(f"{len(passes)}/{len(rows)} runs pass all criteria")
    if passes:
        best = max(passes, key=lambda r: r["contrast"])
        print(f"  best: {best['case']} softness={best['softness']:g} "
              f"guidance={best['guidance']:g} contrast={best['contrast']:.1f}")
        print("  LOOK AT THE CONTACT SHEET before recording this. The criteria "
              "cannot tell a correct edit from a plausible-looking wrong one.")
    else:
        print("  No run passed. That is a result, not a failure of the script.")
        print("  Read the columns: 'in' too low means the edit never took "
              "(try higher guidance); 'out' too high means the pin is leaking "
              "(check the mask); contrast fine but CLIP negative means the "
              "region changed to the wrong thing (caption or edit-span "
              "problem, not a region-control problem).")

    if rows:
        s0 = np.mean([r["seam"] for r in rows if r["softness"] == 0.0])
        s8 = np.mean([r["seam"] for r in rows if r["softness"] == 0.08])
        print(f"\n  softness 0.00 mean seam {s0:6.2f}")
        print(f"  softness 0.08 mean seam {s8:6.2f}")
        print("  On synthetic colour edits softness cut the seam by ~10x. "
              "If these two are equal, the temporal-handover result does not "
              "transfer to real photographs with imprecise masks -- which is "
              "worth more than another confirmation would be.")
    print(f"\n  images -> {out}")


if __name__ == "__main__":
    main()
