"""
Calibrate `instance_selectivity` -- and now `off_instance_change` -- before
trusting either.

THE RULE THIS IMPLEMENTS. Score the ground truth on your metric before you
score anything else. Stage 0's `struct` metric looked meaningful until the
ground truth was pushed through it; the oracle transport result went the same
way. A leakage number from an uncalibrated metric is not evidence.

OFF_INSTANCE_EDIT_RATIO (10.0, in off_instance.py) was frozen against a
self-test bound and 12 real disagreement cases, explicitly NOT against a
ground-truth floor (gate_c_report.md SS6 "Off-instance measure has no null
floor yet"). Every control below now also reports `off_instance_ratio` /
`off_instance_over_target`, giving that floor: oracle_local and
category_level -- which only ever touch the target and/or siblings, both
part of the TRACKED instance set -- must read near-zero off-instance
regardless of how they read on leakage_ratio (siblings are not "off";
leakage onto them is leakage_ratio's job, not this one's). global_shift is
the discriminator that matters most: every pixel changes, including
off-instance, but uniformly, so peak == ambient and the ratio stays near 1
rather than tripping OFF_INSTANCE_EDIT_RATIO. ground_truth gives the real
floor -- what a human-approved edit's off-instance ratio actually looks
like on real MagicBrush regeneration noise, which is the number that was
missing.

That floor (ratio median 83.27, ~8x OFF_INSTANCE_EDIT_RATIO, vs. over_target
median 0.619) is why `run_gate_c.outcome_class` no longer uses the ratio at
all -- see OFF_INSTANCE_EDIT_OVER_TARGET in off_instance.py and
GATE_C_REPORT.md SS7. `off_instance_ratio` stays reported here as a
diagnostic, not as the classifier's own input.

MagicBrush is the only source in play with ground-truth EDITED images --
RefEdit-Bench and MIRA-Bench are both VLM-scored precisely because they have
none -- so it is the only place this can be done at all. Referential ambiguity
in MagicBrush instructions does not matter here: calibration asks what the
metric READS on a known-good edit, not whether a model picked the right
instance.

FOUR CONTROLS WITH KNOWN ANSWERS, then the real edits.

  identity        edited == source. Nothing changed anywhere.
                  EXPECT valid=False. If this scores as a usable case, the
                  degenerate-null guard is not working.

  global_shift    every pixel brightened slightly, nothing edited.
                  EXPECT leakage_ratio ~ 1.0 and sibling_excess_over_bg ~ 0.
                  This is the metric's blind spot, stated numerically: a
                  ratio near 1 does NOT by itself mean category-level editing.
                  It is why `sibling_excess_over_bg` exists.

  oracle_local    only the target region perturbed, siblings untouched.
                  EXPECT leakage_ratio ~ 0, excess ~ 0. The achievable floor.

  category_level  target AND every sibling perturbed equally.
                  EXPECT leakage_ratio ~ 1, excess strongly positive. This is
                  the failure mode the whole benchmark exists to detect; if
                  the metric cannot separate this from oracle_local, it cannot
                  measure anything.

  ground_truth    real MagicBrush source -> target pairs.
                  A human-approved single-instance edit SHOULD look closer to
                  oracle_local than to category_level. If it does not, the
                  metric is reading something other than leakage -- DALL-E 2
                  regenerates the whole frame, so recompression and global
                  tone drift are live confounds.

THE THING MOST LIKELY TO BREAK. `valid = target_change > 10 * (bg + eps)` is a
strict gate, and MagicBrush targets are full-frame regenerations where
background change is NOT small. If `valid` almost never fires on ground truth,
that threshold is mis-set for this data and must be reported and revised
BEFORE any model is scored -- not quietly relaxed afterwards.

USAGE
    python bench/calibrate_metric.py --self-test
    python bench/calibrate_metric.py --frames bench/frames_dev.json --split dev
"""

from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from instance_bench import instance_selectivity
from magicbrush_coco import (_img_bytes, load_coco, load_frames,
                             read_magicbrush, resolve_case)
from off_instance import off_instance_change


def as_frame(cell, hw) -> np.ndarray:
    img = Image.open(io.BytesIO(_img_bytes(cell))).convert("RGB")
    h, w = hw
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img)


def perturb(src: np.ndarray, mask: np.ndarray, shift=(60, -40, 30)) -> np.ndarray:
    out = src.astype(np.int16).copy()
    for c in range(3):
        out[..., c][mask] = np.clip(out[..., c][mask] + shift[c], 0, 255)
    return out.astype(np.uint8)


def controls(src: np.ndarray, tmask: np.ndarray, smasks: list) -> dict:
    """Synthetic edits whose correct reading is known in advance."""
    out = {}
    out["identity"] = src.copy()

    shifted = np.clip(src.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    out["global_shift"] = shifted

    out["oracle_local"] = perturb(src, tmask)

    cat = perturb(src, tmask)
    for sm in smasks:
        cat = perturb(cat, sm)
    out["category_level"] = cat

    # THE CONTROL THAT WAS MISSING. The model edits a SIBLING and leaves the
    # target alone -- it resolved the referring expression to the wrong
    # instance. This is the worst outcome for instance selectivity, worse than
    # leaking onto siblings, because the requested edit did not happen at all.
    #
    # It was absent from the first calibration battery, and its absence hid a
    # real defect: `valid = target_change > 10*bg` reads a mis-targeted edit as
    # "the edit did not take" and DISCARDS it. The gate's bias therefore runs
    # in the model's favour -- leakage gets computed only on the subset where
    # the model at least hit the right object.
    mis = src.copy()
    for sm in smasks:
        mis = perturb(mis, sm)
    out["mis_targeted"] = mis
    return out


def summarise(name, results, show_invalid_reason=False):
    n = len(results)
    valid = [r for r in results if r["valid"]]
    if not n:
        print(f"  {name:<15} (no cases)")
        return {}
    def col(key, rs):
        v = np.array([r[key] for r in rs])
        return v
    line = (f"  {name:<15} valid {len(valid):>3}/{n:<3} "
            f"({len(valid) / n:>5.1%})")
    if valid:
        lr = col("leakage_ratio", valid)
        ex = col("sibling_excess_over_bg", valid)
        tc = col("target_change", valid)
        bg = col("background_change", valid)
        line += (f"  ratio med {np.median(lr):>6.3f}  "
                 f"excess med {np.median(ex):>+7.4f}  "
                 f"tgt {np.median(tc):.4f}  bg {np.median(bg):.4f}")
    print(line)
    if not valid and show_invalid_reason and results:
        lr = col("leakage_ratio", results)
        tc = col("target_change", results)
        bg = col("background_change", results)
        print(f"  {'':<15} (all invalid) ratio med {np.median(lr):.3f}  "
              f"tgt {np.median(tc):.4f}  bg {np.median(bg):.4f}  "
              f"tgt/bg {np.median(tc / (bg + 1e-9)):.2f}")
    out = {"n": n, "n_valid": len(valid),
          "ratio_median": float(np.median(col("leakage_ratio", valid))) if valid else None,
          "excess_median": float(np.median(col("sibling_excess_over_bg", valid))) if valid else None}
    # off_instance is not gated by `valid` -- it is a separate measure of
    # change OUTSIDE the tracked instance set, unconditional on whether the
    # target itself was edited enough to pass the leakage gate. Reported
    # over all n results, matching how run_gate_c.py reports it.
    if results and "off_instance_ratio" in results[0]:
        oratio = col("off_instance_ratio", results)
        oover = col("off_instance_over_target", results)
        print(f"  {'':<15} off-instance: ratio med {np.median(oratio):>6.2f}  "
              f"over_target med {np.median(oover):>6.3f}")
        out["off_instance_ratio_median"] = float(np.median(oratio))
        out["off_instance_over_target_median"] = float(np.median(oover))
    return out


def run_controls_on_cases(args):
    """Controls only, on the RefEdit benchmark cases themselves.

    The ground-truth floor is measured on MagicBrush -- 1024px frames, loose
    hand-drawn target masks -- while the benchmark is RefEdit at 512px with
    tighter polygons and a different sibling-size distribution. Nothing
    guarantees the metric behaves the same across that gap, and the synthetic
    controls need no ground-truth image, so they can be checked directly on the
    cases that will actually be scored.

    If separation collapses here, the MagicBrush calibration does not transfer
    and the floor has to be re-derived on this geometry.
    """
    from refedit_siblings import load_bench, bench_image, raster, load_coco as _lc  # noqa: PLC0415
    from frame_fit import apply_rect  # noqa: PLC0415

    blob = json.loads(Path(args.refedit_cases).read_text())
    cases = blob["cases"]
    coco_index, _c, _u = _lc(Path(args.coco_annotations),
                             tuple(args.coco_splits.split(",")),
                             cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    buckets = collections.defaultdict(list)
    used = 0
    for case in cases:
        idx = int(case["id"].rsplit("-", 1)[1])
        row = rows.get(idx)
        rec = coco_index.get(case["coco_image_id"])
        if row is None or rec is None:
            continue
        H, W = case["coco_hw"]
        anns = {a["id"]: a for a in rec["anns"]}
        def m(i):
            a = anns.get(int(i))
            return None if a is None else apply_rect(
                raster(a["segmentation"], W, H), case["rect"], (512, 512))
        tmask = m(case["target_id"])
        if tmask is None or not tmask.any():
            continue
        smasks = [x for x in (m(i["id"]) for i in case["instances"]
                              if i["category"] == case["category"]
                              and i["id"] != case["target_id"])
                  if x is not None and x.any()]
        if not smasks:
            continue
        src = np.asarray(bench_image(row["input_image"]))
        for name, edited in controls(src, tmask, smasks).items():
            r = instance_selectivity(src, edited, tmask, smasks)
            r.update(off_instance_change(src, edited, tmask, smasks))
            buckets[name].append(r)
        used += 1

    print(f"\ncontrols on {used} RefEdit benchmark cases "
          f"(512px frames, COCO polygon masks)\n")
    print("=" * 96)
    out = {}
    for name in ("identity", "global_shift", "oracle_local", "mis_targeted",
                 "category_level"):
        out[name] = summarise(name, buckets[name], show_invalid_reason=True)
    mt_all = buckets["mis_targeted"]
    mt_valid = sum(1 for r in mt_all if r["valid"])
    mt_ratio = float(np.median([r["leakage_ratio"] for r in mt_all]))
    print(f"  {'ok  ' if mt_valid else 'FAIL'}  mis-targeted (sibling edited, "
          f"target untouched): {mt_valid}/{len(mt_all)} valid, "
          f"ratio med {mt_ratio:.1f}")
    if not mt_valid:
        print("        -> the validity gate DISCARDS mis-targeted edits, reading")
        print("           them as 'the edit did not take'. That is the model's")
        print("           worst failure mode and the exclusion favours the model.")
        print("           Record the outcome class for every case, valid or not.")

    ol, cl = out["oracle_local"], out["category_level"]
    if ol["ratio_median"] is None or cl["ratio_median"] is None:
        print("\n  FAIL  a control produced no valid case on this geometry")
        return
    sep = cl["ratio_median"] - ol["ratio_median"]
    print(f"\n  separation on RefEdit geometry: {sep:.3f}")
    print("  -> compare with the MagicBrush separation; a large drop means the\n"
          "     ground-truth floor does not transfer and must be re-derived here.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"source": "refedit-controls", "n": used, "controls": out}, indent=1))
        print(f"\nwrote {args.out}")


def run(args):
    frames = load_frames(args.frames)
    coco, cats = load_coco(args.coco_annotations)
    rows = read_magicbrush(args.magicbrush, args.split)
    stats = collections.Counter()

    cases = []
    for row in rows:
        rec, masks = resolve_case(row, coco, cats, stats, frames=frames)
        if rec is None:
            continue
        if row.get("target_img") is None:
            stats["no ground-truth target image"] += 1
            continue
        cases.append((row, rec, masks))
    print(f"\n{len(cases)} calibration cases "
          f"(of {len(rows)} turn-1 rows in {args.split})\n")
    if not cases:
        print("nothing to calibrate on")
        return

    buckets = collections.defaultdict(list)
    per_case = []
    for row, rec, masks in cases:
        hw = tuple(rec["edit_hw"])
        src = as_frame(row["source_img"], hw)
        gt = as_frame(row["target_img"], hw)
        tmask = masks[rec["target_id"]]
        smasks = [masks[i["id"]] for i in rec["instances"]
                  if i["category"] == rec["category"]
                  and i["id"] != rec["target_id"] and i["id"] in masks]
        if not smasks:
            continue

        for name, edited in controls(src, tmask, smasks).items():
            cr = instance_selectivity(src, edited, tmask, smasks)
            cr.update(off_instance_change(src, edited, tmask, smasks))
            buckets[name].append(cr)
        r = instance_selectivity(src, gt, tmask, smasks)
        r.update(off_instance_change(src, gt, tmask, smasks))
        buckets["ground_truth"].append(r)
        per_case.append({"id": rec["id"], "instruction": rec["instruction"],
                         "category": rec["category"],
                         "n_siblings": len(smasks), **{
                             k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in r.items() if k != "invalid_reason"}})

    print("=" * 96)
    print("  control        validity            leakage_ratio      "
          "sibling_excess    target/background")
    print("=" * 96)
    out = {}
    for name in ("identity", "global_shift", "oracle_local",
                 "mis_targeted", "category_level", "ground_truth"):
        out[name] = summarise(name, buckets[name],
                              show_invalid_reason=name in ("identity", "ground_truth"))

    print("\nreading it:")
    ok = True
    if out["identity"]["n_valid"] > 0:
        print("  FAIL  identity scored as valid -- the degenerate-null guard "
              "is not working")
        ok = False
    else:
        print("  ok    identity correctly rejected as a non-edit")

    mt_all = buckets["mis_targeted"]
    mt_valid = sum(1 for r in mt_all if r["valid"])
    mt_ratio = float(np.median([r["leakage_ratio"] for r in mt_all]))
    print(f"  {'ok  ' if mt_valid else 'FAIL'}  mis-targeted (sibling edited, "
          f"target untouched): {mt_valid}/{len(mt_all)} valid, "
          f"ratio med {mt_ratio:.1f}")
    if not mt_valid:
        print("        -> the validity gate DISCARDS mis-targeted edits, reading")
        print("           them as 'the edit did not take'. That is the model's")
        print("           worst failure mode and the exclusion favours the model.")
        print("           Record the outcome class for every case, valid or not.")

    ol, cl = out["oracle_local"], out["category_level"]
    cl_exc = float(np.median([r["sibling_excess_over_bg"]
                              for r in buckets["category_level"]]))
    if ol["ratio_median"] is not None and cl["ratio_median"] is not None:
        sep = cl["ratio_median"] - ol["ratio_median"]
        print(f"  {'ok  ' if sep > 0.5 else 'FAIL'}  oracle {ol['ratio_median']:.3f} "
              f"vs category-level {cl['ratio_median']:.3f} (separation {sep:.3f})")
        if sep <= 0.5:
            print("        -> the metric cannot separate the two failure modes "
                  "it exists to distinguish")
            ok = False

    gs_all = buckets["global_shift"]
    gs_ratio = float(np.median([r["leakage_ratio"] for r in gs_all]))
    gs_exc = float(np.median([r["sibling_excess_over_bg"] for r in gs_all]))
    if out["global_shift"]["n_valid"] == 0:
        print("  ok    a pure global shift is rejected by the validity gate")
    else:
        print(f"  WARN  {out['global_shift']['n_valid']} global-shift cases passed "
              "`valid`; the gate is too loose")
        ok = False
    print(f"  note  global shift reads ratio {gs_ratio:.3f} -- the SAME as a "
          f"category-level edit --\n        but excess {gs_exc:+.5f} vs "
          f"category-level's {cl_exc:+.5f}. Quote the excess:\n"
          "        leakage_ratio alone cannot tell a leak from a global tone change.")

    gt = out["ground_truth"]
    if gt["n_valid"] == 0:
        print("  FAIL  NO ground-truth edit passed `valid`. "
              "target_change > 10*background")
        print("        is mis-set for full-frame regenerations. Fix the "
              "threshold before\n        scoring any model -- not after seeing "
              "the model's number.")
        ok = False
    elif gt["ratio_median"] is not None and ol["ratio_median"] is not None:
        d_ol = abs(gt["ratio_median"] - ol["ratio_median"])
        d_cl = abs(gt["ratio_median"] - cl["ratio_median"])
        verdict = "oracle_local" if d_ol < d_cl else "category_level"
        print(f"  {'ok  ' if verdict == 'oracle_local' else 'FAIL'}  ground "
              f"truth reads {gt['ratio_median']:.3f}, closer to {verdict}")
        if verdict != "oracle_local":
            print("        -> a human-approved single-instance edit looks "
                  "category-level to\n           the metric. The metric is "
                  "reading something else.")
            ok = False

    print(f"\n  CALIBRATION {'PASSED' if ok else 'FAILED'} -- "
          f"{'safe to score models' if ok else 'do NOT score models yet'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"split": args.split, "n_cases": len(per_case),
             "controls": out, "ground_truth_per_case": per_case}, indent=1))
        print(f"\nwrote {args.out}")


def self_test():
    rng = np.random.default_rng(0)
    src = rng.integers(40, 200, (128, 128, 3), dtype=np.uint8)
    t = np.zeros((128, 128), bool); t[20:50, 20:50] = True
    s1 = np.zeros((128, 128), bool); s1[20:50, 78:108] = True
    s2 = np.zeros((128, 128), bool); s2[78:108, 20:50] = True
    sm = [s1, s2]

    c = controls(src, t, sm)
    r_id = instance_selectivity(src, c["identity"], t, sm)
    assert not r_id["valid"], r_id
    assert r_id["target_change"] == 0.0

    r_ol = instance_selectivity(src, c["oracle_local"], t, sm)
    assert r_ol["valid"], r_ol
    assert r_ol["leakage_ratio"] < 0.01, r_ol
    assert abs(r_ol["sibling_excess_over_bg"]) < 1e-6, r_ol

    r_cl = instance_selectivity(src, c["category_level"], t, sm)
    assert r_cl["valid"], r_cl
    assert r_cl["leakage_ratio"] > 0.8, r_cl
    assert r_cl["sibling_excess_over_bg"] > 0.05, r_cl
    assert r_cl["leakage_ratio"] - r_ol["leakage_ratio"] > 0.5

    # the blind spot, demonstrated rather than asserted from the docstring
    r_gs = instance_selectivity(src, c["global_shift"], t, sm)
    assert abs(r_gs["leakage_ratio"] - 1.0) < 0.05, r_gs
    assert abs(r_gs["sibling_excess_over_bg"]) < 1e-3, r_gs

    # the missing control: siblings edited, target untouched
    r_mt = instance_selectivity(src, c["mis_targeted"], t, sm)
    assert r_mt["sibling_change"] > 10 * r_mt["target_change"], r_mt
    assert not r_mt["valid"], (
        "if this ever passes, the gate has changed and the note below is stale")
    print(f"  mis-targeted: ratio {r_mt['leakage_ratio']:.1f}, "
          f"valid={r_mt['valid']} <- discarded by the gate")

    # off_instance floor. Siblings are TRACKED (part of `inst`, not "off"),
    # so oracle_local/category_level/mis_targeted -- which only ever touch
    # target and/or siblings -- must all read near-zero off-instance
    # regardless of how they read on leakage_ratio. off_instance and
    # leakage_ratio are answering different questions and these controls
    # prove it: a control that saturates leakage_ratio (category_level)
    # should NOT also saturate off_instance_ratio.
    o_ol = off_instance_change(src, c["oracle_local"], t, sm)
    assert o_ol["off_instance_ratio"] < 0.1, o_ol
    assert o_ol["off_instance_over_target"] < 0.1, o_ol

    o_cl = off_instance_change(src, c["category_level"], t, sm)
    assert o_cl["off_instance_ratio"] < 0.1, o_cl

    o_mt = off_instance_change(src, c["mis_targeted"], t, sm)
    assert o_mt["off_instance_ratio"] < 0.1, o_mt

    # THE discriminator this measure exists for: a uniform shift (every
    # pixel, including off-instance, moves by the same amount) must NOT
    # read as a hot spot. Peak == ambient under uniform change by
    # construction, so ratio ~= 1 -- nowhere near OFF_INSTANCE_EDIT_RATIO
    # (10.0) despite literally every pixel having changed.
    o_gs = off_instance_change(src, c["global_shift"], t, sm)
    assert 0.5 < o_gs["off_instance_ratio"] < 2.0, o_gs

    print(f"self-test OK  (oracle {r_ol['leakage_ratio']:.3f} | "
          f"category {r_cl['leakage_ratio']:.3f} | "
          f"global shift {r_gs['leakage_ratio']:.3f}, "
          f"excess {r_gs['sibling_excess_over_bg']:+.5f} | "
          f"off-instance floor: oracle {o_ol['off_instance_ratio']:.2f}, "
          f"category {o_cl['off_instance_ratio']:.2f}, "
          f"global shift {o_gs['off_instance_ratio']:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magicbrush", default="/data/datasets/MagicBrush")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--frames", default="bench/frames_dev.json")
    p.add_argument("--split", default="dev")
    p.add_argument("--out", default="runs/gate_c/calibration.json")
    p.add_argument("--refedit-cases", default=None,
                   help="run the synthetic controls on the RefEdit benchmark "
                        "cases instead, to check the metric transfers")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.refedit_cases:
        run_controls_on_cases(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
