"""
Gate C: run an editing model over the benchmark and score instance selectivity.

THE GATE. For each case: feed the source image and the referring-expression
instruction to the model, then measure how much the output changed the TARGET
instance versus its same-category SIBLINGS.

Read against the calibrated reference points (runs/gate_c/calibration.json):

    leakage_ratio  0.011   ground truth, human-approved single-instance edit
                   1.001   category-level -- the model edited every instance

    A model score is a position between those. Both endpoints are MEASURED.

NEVER QUOTE leakage_ratio ALONE. A pure global tone shift and a full
category-level edit both read exactly 1.000; only `sibling_excess_over_bg`
separates them (0.000 vs +0.157).

CHECK THE VALIDITY RATE FIRST. `instance_selectivity` compares the output to the
SOURCE, so a model that changes little everywhere produces a low ratio for the
wrong reason. `valid = target_change > 10 * background` is what catches that.
A low ratio with a low validity rate means the edits did not take -- it is not
evidence of selectivity. Read n_valid before reading any ratio.

THE ABLATION (--ablation). Every case carries the same edit twice:

    editing_instruction       "Remove the giraffe right next to the lady..."
    edit_instruction_single   "Remove the giraffe."

Same forward pass, different text column, SAME SEED. The difference separates
two things the headline number conflates:

    leakage under the full expression   -> the editing mechanism leaks
    leakage only without it             -> the model failed to RESOLVE the
                                           reference, and would have been
                                           selective if it had

That distinction is closer to RegionFlow's actual claim than the headline is.

FROZEN. Thresholds, filters and the audit are fixed (see gate_c_note.md §9).
Revising any of them after seeing a score makes the result unfalsifiable.

USAGE
    python bench/run_gate_c.py --model none --limit 8      # plumbing, no GPU
    python bench/run_gate_c.py --model refedit-sd3 --ablation \\
        --out runs/gate_c/refedit_sd3.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from frame_fit import apply_rect
from instance_bench import aggregate, instance_selectivity
from off_instance import off_instance_change
from off_instance import self_test as off_self_test
from refedit_siblings import bench_image, load_bench, load_coco, raster

PROMPT_COLS = {"full": "editing_instruction", "single": "edit_instruction_single"}

# Checkpoints that load through the SAME UltraEdit pipeline class. RefEdit-SD3
# is finetuned on UltraEdit-FREEFORM (the *_wo_mask script), so handing it a
# mask is out of distribution and a null result from it would be
# uninterpretable. `sd3-ultraedit-mask` is the same architecture actually
# TRAINED with masks, and is the arm that answers "is localisation the
# problem?".
MODEL_REPOS = {
    "refedit-sd3": "bpathir1/RefEdit-SD3",
    "sd3-ultraedit-mask": "BleachNick/SD3_UltraEdit_w_mask",
}

# Mask arms. `white` is not decoration: it is the control that separates "the
# pipeline behaves differently when mask_img is not None" from "the pipeline
# uses the CONTENT of the mask". Without it, none -> oracle confounds those two
# changes and neither branch of the reading key is safe.
MASK_MODES = ("none", "white", "oracle", "sibling")


def outcome_class(r: dict) -> str:
    """Describe every case, including ones `valid` discards.

    Uses only COMPARISONS between quantities already computed -- no new
    threshold, so nothing here is tunable and nothing is being relaxed.

      no_change     nothing on either instance exceeds background: the model
                    returned ~the source. Genuinely unmeasurable.
      mis_targeted  the SIBLING changed more than the target: the model
                    resolved the referring expression to the wrong instance.
                    The requested edit did not happen. `valid` currently reads
                    this as "the edit did not take" and DROPS it, which biases
                    every reported leakage number in the model's favour.
      measurable    the target moved more than any sibling; the ratio means
                    what it is supposed to mean.
    """
    tgt, sib, bg = r["target_change"], r["sibling_change"], r["background_change"]
    if max(tgt, sib) <= bg:
        return "no_change"
    if sib > tgt:
        return "mis_targeted"
    return "measurable"


def case_seed(case_id: str, base: int) -> int:
    """Deterministic per case, so a rerun reproduces and the ablation pair
    shares noise. Comparing full vs stripped under different noise would
    measure the sampler, not the instruction."""
    h = hashlib.sha256(f"{case_id}:{base}".encode()).hexdigest()
    return int(h[:8], 16)


def build_masks(case, coco_index):
    rec = coco_index.get(case["coco_image_id"])
    if rec is None:
        return None, None
    H, W = case["coco_hw"]
    anns = {a["id"]: a for a in rec["anns"]}

    def m(i):
        a = anns.get(int(i))
        if a is None:
            return None
        return apply_rect(raster(a["segmentation"], W, H), case["rect"], (512, 512))

    tmask = m(case["target_id"])
    if tmask is None or not tmask.any():
        return None, None
    smasks = [x for x in (m(i["id"]) for i in case["instances"]
                          if i["category"] == case["category"]
                          and i["id"] != case["target_id"])
              if x is not None and x.any()]
    return (tmask, smasks) if smasks else (None, None)


class IdentityModel:
    """Returns the source unchanged. Exercises the whole harness with no GPU.

    Its expected scoring is known: every case invalid, because nothing changed.
    If a --model none run reports valid cases, the harness is wrong before any
    real model is loaded.
    """

    name = "none"

    def __call__(self, img, prompt, seed, mask=None):
        return img


class SD3InstructPix2Pix:
    """Any checkpoint loading through StableDiffusion3InstructPix2PixPipeline.

    The conditioning mask is NOT built here. It is built in `run()` next to the
    target and sibling masks it is derived from, and passed in. One place, one
    definition -- the alternative is a second mask convention living inside the
    model, which is how polarity bugs survive.
    """

    def __init__(self, repo, name, steps=50, image_guidance=1.5, guidance=7.5,
                 device="cuda", dtype=None):
        import torch  # noqa: PLC0415
        try:
            from diffusers import StableDiffusion3InstructPix2PixPipeline  # noqa: PLC0415
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "StableDiffusion3InstructPix2PixPipeline not found. It takes a "
                "`mask_img` argument and comes from UltraEdit's diffusers fork, "
                "not stock diffusers. Install that fork "
                "(training/RefEdit-SD3/UltraEdit in the refedit repo)."
            ) from exc
        self.torch = torch
        self.name, self.repo = name, repo
        self.steps, self.ig, self.g = steps, image_guidance, guidance
        self.device = device
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            repo, torch_dtype=dtype or torch.float16).to(device)

    def __call__(self, img, prompt, seed, mask=None):
        gen = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt, image=img, mask_img=mask,
                         num_inference_steps=self.steps,
                         image_guidance_scale=self.ig,
                         guidance_scale=self.g,
                         generator=gen).images[0]


def load_model(args):
    if args.model == "none":
        return IdentityModel()
    if args.model in MODEL_REPOS:
        if args.model == "refedit-sd3" and args.mask_mode != "none":
            print("  NOTE: RefEdit-SD3 is finetuned on UltraEdit-FREEFORM, so a "
                  "mask is\n  out of distribution for it. Keep this arm as a "
                  "secondary probe; the\n  primary mask arm is "
                  "--model sd3-ultraedit-mask.")
        return SD3InstructPix2Pix(
            MODEL_REPOS[args.model], args.model, steps=args.steps,
            image_guidance=args.image_guidance, guidance=args.guidance,
            device=args.device)
    raise SystemExit(f"unknown model {args.model!r}")


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation, matching off_instance._dilate. A tight COCO polygon may
    not give a regenerating model room to make a coherent edit at the boundary;
    the amount of room is a FROZEN choice, set once by looking at smoke-test
    output and never touched after a score is seen."""
    if r <= 0:
        return mask
    m = mask
    for axis in (0, 1):
        out = m.copy()
        for s_ in range(1, r + 1):
            out |= np.roll(m, s_, axis=axis)
            out |= np.roll(m, -s_, axis=axis)
        m = out
    return m


def cond_mask(mode: str, tmask, smasks, size, dilate_px: int):
    """The conditioning image handed to `mask_img`.

    POLARITY. UltraEdit's README seeds a BLANK WHITE mask for free-form
    editing, i.e. white = editable. So the oracle mask is white INSIDE the
    target and black outside. `self_test` asserts this, because an inverted
    mask produces plausible-looking output and would be read as a finding.
    """
    if mode == "none":
        return None
    if mode == "white":
        return Image.new("RGB", size, (255, 255, 255))
    if mode == "oracle":
        m = tmask
    elif mode == "sibling":
        # POSITIVE CONTROL. Point the mask at a sibling and mis-targeting
        # should go to ~100%. If it does not, the model is ignoring the mask
        # and BOTH branches of the oracle reading key are void. Largest sibling
        # by area, deterministic.
        if not smasks:
            return None
        m = max(smasks, key=lambda x: np.asarray(x).sum())
    else:
        raise SystemExit(f"unknown mask mode {mode!r}")
    m = dilate(np.asarray(m) > 0, dilate_px)
    return Image.fromarray((m * 255).astype(np.uint8)).convert("RGB")


def run(args):
    blob = json.loads(Path(args.cases).read_text())
    cases = blob["cases"]
    if args.limit:
        cases = cases[:args.limit]
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists() and not args.overwrite:
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                # Keyed on the ARM too. Without mask_mode in the key, writing a
                # new arm into an existing results file silently resumes over
                # the completed one and reports it as finished.
                done.add((r["id"], r["prompt_kind"],
                          r.get("mask_mode", "none")))
        print(f"resuming: {len(done)} case/prompt results already in {out_path}")

    kinds = ["full"] + (["single"] if args.ablation else [])
    model = load_model(args)
    print(f"model={model.name}  cases={len(cases)}  prompts={kinds}  "
          f"mask={args.mask_mode}\n")

    img_dir = Path(args.save_images) if args.save_images else None
    if img_dir:
        img_dir.mkdir(parents=True, exist_ok=True)

    t0, n = time.time(), 0
    with out_path.open("a") as fh:
        for case in cases:
            idx = int(case["id"].rsplit("-", 1)[1])
            row = rows.get(idx)
            if row is None:
                continue
            tmask, smasks = build_masks(case, coco_index)
            if tmask is None:
                print(f"  {case['id']}: masks unavailable, skipped")
                continue
            src_img = bench_image(row["input_image"])
            src = np.asarray(src_img)

            cmask = cond_mask(args.mask_mode, tmask, smasks, src_img.size,
                              args.mask_dilate)
            if args.mask_mode == "sibling" and cmask is None:
                print(f"  {case['id']}: no sibling to point at, skipped")
                continue

            for kind in kinds:
                if (case["id"], kind, args.mask_mode) in done:
                    continue
                prompt = row.get(PROMPT_COLS[kind]) or row["editing_instruction"]
                # SAME seed as the mask=none run, so the arms pair case by case
                # and the only thing that differs is the spatial condition.
                seed = case_seed(case["id"], args.seed)
                edited = model(src_img, prompt, seed, cmask)
                if edited.size != (512, 512):
                    edited = edited.resize((512, 512), Image.BILINEAR)
                out = np.asarray(edited.convert("RGB"))
                res = instance_selectivity(src, out, tmask, smasks)
                # ADDITIVE. Separate namespace, separate function, nothing in
                # the frozen metric reads it. See off_instance.__doc__.
                off = off_instance_change(src, out, tmask, smasks)
                eps = 1e-9
                rec = {"id": case["id"], "prompt_kind": kind, "prompt": prompt,
                       "outcome": outcome_class(res),
                       "target_over_bg": round(
                           res["target_change"] / (res["background_change"] + eps), 3),
                       "signal_over_bg": round(
                           max(res["target_change"], res["sibling_change"])
                           / (res["background_change"] + eps), 3),
                       "difficulty": case["difficulty"],
                       "category": case["category"],
                       "n_siblings": case["n_siblings"], "seed": seed,
                       "model": model.name, "mask_mode": args.mask_mode,
                       **off,
                       **{k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in res.items() if k != "invalid_reason"}}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if img_dir:
                    edited.save(img_dir / f"{case['id']}_{kind}.png")
                n += 1
                if n % 10 == 0:
                    el = time.time() - t0
                    total = len(cases) * len(kinds) - len(done)
                    print(f"  {n}/{total}  {el / n:.1f}s/edit  "
                          f"eta {(total - n) * el / n / 60:.1f}m")

    report(out_path, args)


def report(out_path: Path, args):
    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    if not recs:
        print("no results")
        return
    print("\n" + "=" * 74)
    print(f"GATE C -- {recs[0]['model']}, {len(recs)} scored edits")
    print("=" * 74)

    by_kind = collections.defaultdict(list)
    for r in recs:
        by_kind[r["prompt_kind"]].append(r)

    summary = {}
    for kind, rs in sorted(by_kind.items()):
        agg = aggregate(rs)
        summary[kind] = agg
        # aggregate() returns n=0 when nothing is valid, which would print
        # "0/0" and hide how many edits were actually attempted.
        # Outcome breakdown over ALL cases, not just the ones `valid` keeps.
        oc = collections.Counter(r.get("outcome", outcome_class(r)) for r in rs)
        mis = [r for r in rs if r.get("outcome", outcome_class(r)) == "mis_targeted"]
        nv, n = agg.get("n_valid", 0), len(rs)
        print(f"\n  prompt = {kind}   ({PROMPT_COLS[kind]})")
        print(f"    outcomes: " + "  ".join(
            f"{k} {v} ({v / n:.0%})" for k, v in sorted(oc.items())))
        if mis:
            print(f"    MIS-TARGETED {len(mis)}/{n} ({len(mis) / n:.0%}) -- the "
                  "model edited a SIBLING instead of the")
            print("      target. These are the model's worst failures and "
                  "`valid` drops most of")
            print("      them, so any ratio below is computed on a subset that "
                  "excludes them.")
            ex = sorted(mis, key=lambda r: -r["leakage_ratio"])[:3]
            for r in ex:
                print(f"        {r['id']:<13} tgt {r['target_change']:.4f} "
                      f"sib {r['sibling_change']:.4f} ratio {r['leakage_ratio']:.1f}"
                      f"  valid={r['valid']}")
        print(f"    valid {nv}/{n} ({nv / max(n, 1):.1%})"
              + ("   <- READ THIS FIRST; a low ratio with few valid cases means"
                 "\n        the edits did not take, not that the model is selective"
                 if nv / max(n, 1) < 0.5 else ""))
        if nv == 0:
            print("    no case had a measurable edit; nothing to report")
            continue
        ex = np.array([r["sibling_excess_over_bg"] for r in rs if r["valid"]])
        print(f"    leakage_ratio   mean {agg['leakage_ratio_mean']:.3f}  "
              f"median {agg['leakage_ratio_median']:.3f}  "
              f"p90 {agg['leakage_ratio_p90']:.3f}")
        print(f"    sibling_excess  mean {agg['sibling_excess_mean']:+.4f}  "
              f"median {np.median(ex):+.4f}")
        print(f"    cases with ratio > 0.5: {agg['frac_cases_ratio_above_0p5']:.1%}")
        print(f"    reference: ground truth 0.011 / excess -0.005   |   "
              f"category-level 1.001 / excess +0.157")

        # OFF-INSTANCE, over ALL cases -- this class is invisible to `valid`
        # by construction, so gating it on validity would hide exactly what it
        # was added to find.
        oi = [r for r in rs if r.get("off_instance_bbox") is not None]
        if oi:
            ratio = np.array([r["off_instance_ratio"] for r in oi])
            over = np.array([r["off_instance_over_target"] for r in oi])
            print(f"    off-instance    ratio median {np.median(ratio):.2f}  "
                  f"p90 {np.percentile(ratio, 90):.2f}   "
                  f"over-target median {np.median(over):.2f}")
            print(f"      cases with a hot spot louder than the target "
                  f"(over-target > 1): {(over > 1).sum()}/{len(oi)}"
                  f"  <- candidate third-mode failures; LOOK at the bbox crop")

        for key, label in ((lambda r: r["difficulty"], "difficulty"),
                           (lambda r: "person" if r["category"] == "person"
                            else "other", "category")):
            d = collections.defaultdict(list)
            for r in rs:
                if r["valid"]:
                    d[key(r)].append(r["leakage_ratio"])
            print(f"    by {label}: " + "  ".join(
                f"{k} {np.median(v):.3f} (n={len(v)})" for k, v in sorted(d.items())))

    if len(by_kind) > 1 and all(summary[k].get("n_valid") for k in by_kind):
        # PAIRED. The two prompt kinds have different valid subsets, so
        # differencing their medians compares different cases and means
        # nothing. Pair on case id, keeping only cases valid under BOTH.
        full = {r["id"]: r for r in by_kind["full"] if r["valid"]}
        sing = {r["id"]: r for r in by_kind["single"] if r["valid"]}
        both = sorted(set(full) & set(sing))
        print(f"\n  ABLATION (paired on {len(both)} cases valid under both; "
              f"full has {len(full)}, stripped {len(sing)})")
        if len(both) < 10:
            print("    too few paired cases to read a difference")
        else:
            df = np.array([sing[i]["leakage_ratio"] - full[i]["leakage_ratio"]
                           for i in both])
            de = np.array([sing[i]["sibling_excess_over_bg"]
                           - full[i]["sibling_excess_over_bg"] for i in both])
            # sign test: no distributional assumption, tiny n, paired
            pos = int((df > 0).sum())
            from math import comb  # noqa: PLC0415
            n = int((df != 0).sum())
            pv = sum(comb(n, k) for k in range(min(pos, n - pos) + 1)) / 2 ** n * 2
            print(f"    ratio  median {np.median(df):+.3f}  mean {df.mean():+.3f}"
                  f"   worse-without-expression in {pos}/{n} cases "
                  f"(sign test p={min(pv, 1.0):.3f})")
            print(f"    excess median {np.median(de):+.4f}")
        d = (summary["single"]["leakage_ratio_median"]
             - summary["full"]["leakage_ratio_median"])
        print(f"    unpaired medians differ by {d:+.3f} -- NOT the ablation, "
              "different subsets")

        # The ratio ablation is conditional on hitting the right instance.
        # Whether the expression helps the model FIND the instance is a
        # separate, paired, binary question -- McNemar, on all cases.
        fo = {r["id"]: r.get("outcome", outcome_class(r)) for r in by_kind["full"]}
        so = {r["id"]: r.get("outcome", outcome_class(r)) for r in by_kind["single"]}
        ids = sorted(set(fo) & set(so))
        b = sum(1 for i in ids if fo[i] != "mis_targeted" and so[i] == "mis_targeted")
        c = sum(1 for i in ids if fo[i] == "mis_targeted" and so[i] != "mis_targeted")
        from math import comb  # noqa: PLC0415
        n = b + c
        pv = (sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n * 2) if n else 1.0
        print(f"\n  MIS-TARGETING, paired over {len(ids)} cases (McNemar)")
        print(f"    full {sum(1 for i in ids if fo[i] == 'mis_targeted')}/{len(ids)}"
              f"   stripped {sum(1 for i in ids if so[i] == 'mis_targeted')}/{len(ids)}")
        print(f"    discordant: {b} became mis-targeted without the expression, "
              f"{c} the reverse  (p={min(pv, 1.0):.3f})")
        print("    -> this is whether the referring expression helps the model "
              "FIND the\n       instance. The ratio ablation above is separate: "
              "whether it reduces\n       leakage GIVEN the instance was found.")
        

    # ---- sensitivity to the validity threshold --------------------------
    # The frozen gate is `target_change > 10 * background`. That 10x was
    # calibrated on MagicBrush ground truth, where DALL-E 2 composites the
    # edited region back and background change is 0.0093. RefEdit-SD3
    # regenerates the frame and its ambient change is ~0.06 -- so the gate
    # encodes a background-preservation assumption this model class does not
    # meet, and it discards 75% of cases.
    #
    # Reported as a GRID, not replaced by a hand-picked value. If the
    # conclusions hold across K the result is robust to the choice; if they
    # flip, that is itself the finding. K must exceed 1.0: a pure global tone
    # shift reads exactly tgt/bg = 1.000 (measured in calibration), so any
    # K <= 1 would admit a model that changed nothing but the overall tone.
    print("\n" + "=" * 74)
    print("  SENSITIVITY to the validity threshold K  (valid: tgt > K x bg)")
    print("=" * 74)
    print(f"  {'K':>4}  {'kind':<7} {'n_valid':>8} {'ratio med':>10} "
          f"{'>0.5':>7} {'excess med':>11} {'pairs':>6} {'ablation p':>11}")
    for K in (10.0, 5.0, 3.0, 2.0):
        pairs = {}
        for kind in sorted(by_kind):
            rs = by_kind[kind]
            vs = [r for r in rs
                  if r["target_change"] > K * (r["background_change"] + 1e-9)]
            pairs[kind] = {r["id"]: r for r in vs}
            if not vs:
                print(f"  {K:>4.0f}  {kind:<7} {0:>8}")
                continue
            lr = np.array([r["leakage_ratio"] for r in vs])
            ex = np.array([r["sibling_excess_over_bg"] for r in vs])
            cell = (f"  {K:>4.0f}  {kind:<7} {len(vs):>8} "
                    f"{np.median(lr):>10.3f} {(lr > 0.5).mean():>7.1%} "
                    f"{np.median(ex):>+11.4f}")
            if kind == "single" and "full" in pairs:
                both = sorted(set(pairs["full"]) & set(pairs["single"]))
                if len(both) >= 6:
                    df = np.array([pairs["single"][i]["leakage_ratio"]
                                   - pairs["full"][i]["leakage_ratio"] for i in both])
                    from math import comb  # noqa: PLC0415
                    pos, n = int((df > 0).sum()), int((df != 0).sum())
                    pv = (sum(comb(n, k) for k in range(min(pos, n - pos) + 1))
                          / 2 ** n * 2) if n else 1.0
                    cell += f" {len(both):>6} {min(pv, 1.0):>11.3f}"
                else:
                    cell += f" {len(both):>6} {'--':>11}"
            print(cell)
    print("\n  Mis-targeting does NOT depend on K -- it is a comparison of "
          "sibling_change\n  to target_change, so the 28% / 35% rates stand "
          "whatever K is chosen.")

    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.summary_out}")


def self_test():
    rng = np.random.default_rng(0)
    src = rng.integers(40, 200, (512, 512, 3), dtype=np.uint8)
    t = np.zeros((512, 512), bool); t[100:200, 100:200] = True
    s = np.zeros((512, 512), bool); s[100:200, 300:400] = True

    r = instance_selectivity(src, src, t, [s])
    assert not r["valid"], "identity must not score as a usable case"

    assert case_seed("refedit-1", 0) == case_seed("refedit-1", 0)
    assert case_seed("refedit-1", 0) != case_seed("refedit-2", 0)
    assert case_seed("refedit-1", 0) != case_seed("refedit-1", 1)

    assert outcome_class({"target_change": 0.0, "sibling_change": 0.0,
                          "background_change": 0.0}) == "no_change"
    assert outcome_class({"target_change": 0.03, "sibling_change": 0.38,
                          "background_change": 0.056}) == "mis_targeted"
    assert outcome_class({"target_change": 0.21, "sibling_change": 0.02,
                          "background_change": 0.011}) == "measurable"

    m = IdentityModel()
    img = Image.fromarray(src)
    assert np.array_equal(np.asarray(m(img, "anything", 0, None)), src)

    # POLARITY. White = editable, per UltraEdit's README. An inverted mask
    # yields plausible images and would be read as a model finding, so assert
    # it here rather than discovering it three arms later.
    cm = np.asarray(cond_mask("oracle", t, [s], (512, 512), 0))
    assert cm[150, 150, 0] == 255, "oracle mask must be WHITE inside the target"
    assert cm[10, 10, 0] == 0, "oracle mask must be BLACK outside the target"

    # The sibling control must point somewhere ELSE than the target.
    cm = np.asarray(cond_mask("sibling", t, [s], (512, 512), 0))
    assert cm[150, 350, 0] == 255 and cm[150, 150, 0] == 0

    # Dilation grows the mask and nothing else.
    a = (np.asarray(cond_mask("oracle", t, [s], (512, 512), 0))[..., 0] > 0)
    b = (np.asarray(cond_mask("oracle", t, [s], (512, 512), 8))[..., 0] > 0)
    assert b.sum() > a.sum() and (a & ~b).sum() == 0

    assert cond_mask("none", t, [s], (512, 512), 8) is None
    assert cond_mask("sibling", t, [], (512, 512), 8) is None

    off_self_test()
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--model", default="none",
                   choices=("none",) + tuple(MODEL_REPOS))
    p.add_argument("--ablation", action="store_true",
                   help="also run edit_instruction_single with the same seed")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--image-guidance", type=float, default=1.5)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-mode", default="none", choices=MASK_MODES,
                   help="none: free-form (the completed Gate C run). white: "
                        "blank mask, the mask-PRESENT-but-uninformative "
                        "control. oracle: the target instance. sibling: "
                        "positive control, should drive mis-targeting to ~100%%")
    p.add_argument("--mask-dilate", type=int, default=8,
                   help="FROZEN before scoring. Set it by looking at smoke-test "
                        "output, record the value, never touch it again")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="runs/gate_c/results.jsonl")
    p.add_argument("--summary-out", default="runs/gate_c/summary.json")
    p.add_argument("--save-images", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
    elif args.report_only:
        report(Path(args.out), args)
    else:
        run(args)


if __name__ == "__main__":
    main()
