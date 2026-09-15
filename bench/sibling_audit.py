"""
Are the siblings usable as leakage probes?

WHY. `MIN_AREA_FRAC = 0.005` is applied identically to the target and to the
siblings, but the two roles have different requirements:

  TARGET   only has to confirm the edit landed. 0.5% of frame is plenty.
  SIBLING  is the measurement surface. A tiny sibling gives a mean-change
           estimate dominated by noise, so a real leak reads as nothing.

And truncation is a separate axis that area does not capture. A car cut by the
frame to a fifth of itself is a bad probe even at a respectable pixel count:
the visible fraction is arbitrary, and a model regenerating the scene has no
strong object prior on a fragment that is not recognisably a car -- so a change
there is not evidence of CATEGORY-level leakage, which is the thing being
measured.

A third axis has never been checked at all: `instance_bench.qualifies` rejects
on MAX_IOU_WITH_SIBLING = 0.10, and that filter has not been run on these
cases. Where target and sibling overlap heavily, a change cannot be attributed
to either.

WHAT THIS DOES. Rebuilds every instance mask from the stored `rect` and the
COCO annotations -- no downloads, no registration, nothing refitted -- and
reports the three distributions plus a retention table. Thresholds then get set
the way the coverage and margin thresholds were: read off a distribution, from
a plateau, not picked to hit a target count.

WHAT TO WATCH FOR. If edge-truncated siblings are a handful, they are
`unusable` verdicts in the ledger and need no rule. If they are systematic,
they are a filter -- and the case count has to be restated, because a benchmark
whose leakage surface is partly frame fragments measures something other than
instance selectivity.

USAGE
    python bench/sibling_audit.py --cases bench/cases_refedit_full.json
    python bench/sibling_audit.py --cases bench/cases_refedit_full.json \\
        --filtered-out bench/cases_refedit_filtered.json \\
        --min-sib-area 0.01 --max-target-iou 0.10 --drop-truncated
    python bench/sibling_audit.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from refedit_siblings import (load_coco, raster, write_overlay,
                             stratified_sample, write_contact_sheet,
                             load_bench, bench_image)
from frame_fit import apply_rect

BORDER_TOL = 2          # px from the COCO frame edge that counts as touching


def instance_stats(ann, rect, coco_wh, out_hw=(512, 512), target=None):
    """Geometry of one instance, in both the COCO frame and the bench frame."""
    W, H = coco_wh
    m_coco = raster(ann["segmentation"], W, H)
    if not m_coco.any():
        return None
    ys, xs = np.where(m_coco)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    sides = {
        "left": x0 <= BORDER_TOL,
        "right": x1 >= W - 1 - BORDER_TOL,
        "top": y0 <= BORDER_TOL,
        "bottom": y1 >= H - 1 - BORDER_TOL,
    }
    n_sides = sum(sides.values())

    # How much of the object's own boundary lies ALONG a frame edge. A fragment
    # cut by the frame has a long straight run against it; an object merely
    # reaching the edge touches it briefly.
    edge_px = 0
    if sides["left"]:
        edge_px += int(m_coco[:, :BORDER_TOL + 1].any(axis=1).sum())
    if sides["right"]:
        edge_px += int(m_coco[:, -(BORDER_TOL + 1):].any(axis=1).sum())
    if sides["top"]:
        edge_px += int(m_coco[:BORDER_TOL + 1, :].any(axis=0).sum())
    if sides["bottom"]:
        edge_px += int(m_coco[-(BORDER_TOL + 1):, :].any(axis=0).sum())
    perim = 2 * ((x1 - x0 + 1) + (y1 - y0 + 1))
    cut_frac = edge_px / perim if perim else 0.0

    m_bench = apply_rect(m_coco, rect, out_hw)
    area_frac = float(m_bench.mean())
    out = {"area_frac": area_frac, "n_border_sides": n_sides,
           "cut_frac": float(cut_frac), "px": int(m_bench.sum())}
    if target is not None:
        u = int((m_bench | target).sum())
        out["iou_target"] = float((m_bench & target).sum() / u) if u else 0.0
    return out, m_bench


def analyse(cases, coco_index):
    rows = []
    for case in cases:
        rec = coco_index.get(case["coco_image_id"])
        if rec is None:
            continue
        H, W = case["coco_hw"]
        anns = {a["id"]: a for a in rec["anns"]}
        tgt_ann = anns.get(int(case["target_id"]))
        if tgt_ann is None:
            continue
        t = instance_stats(tgt_ann, case["rect"], (W, H))
        if t is None:
            continue
        t_stats, t_mask = t

        sibs = []
        for inst in case["instances"]:
            if inst["category"] != case["category"] or inst["id"] == case["target_id"]:
                continue
            a = anns.get(int(inst["id"]))
            if a is None:
                continue
            r = instance_stats(a, case["rect"], (W, H), target=t_mask)
            if r is None:
                continue
            s, _ = r
            s["id"] = inst["id"]
            sibs.append(s)
        rows.append({"case": case, "target": t_stats, "siblings": sibs})
    return rows


def hist(name, v, lo=0.0, hi=1.0, bins=10):
    v = np.asarray(v)
    print(f"\n  {name}  (n={len(v)})")
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    top = max(counts.max(), 1)
    for c, e in zip(counts, edges[:-1]):
        print(f"    {e:.2f}-{e + (hi - lo) / bins:.2f} "
              f"{'#' * int(round(38 * c / top)):<38} {c}")


def report(rows, args):
    all_sib = [s for r in rows for s in r["siblings"]]
    if not all_sib:
        print("no siblings found")
        return

    area = np.array([s["area_frac"] for s in all_sib])
    cut = np.array([s["cut_frac"] for s in all_sib])
    iou = np.array([s.get("iou_target", 0.0) for s in all_sib])
    nb = collections.Counter(s["n_border_sides"] for s in all_sib)

    print("=" * 70)
    print(f"{len(rows)} cases, {len(all_sib)} siblings")
    print("=" * 70)
    print(f"  sibling area frac   median {np.median(area):.4f}  "
          f"p10 {np.percentile(area, 10):.4f}  p90 {np.percentile(area, 90):.4f}")
    print(f"  target  area frac   median "
          f"{np.median([r['target']['area_frac'] for r in rows]):.4f}")
    print(f"  siblings below 1% of frame: {(area < 0.01).sum()} "
          f"({(area < 0.01).mean():.1%})")
    print(f"  siblings below 2% of frame: {(area < 0.02).sum()} "
          f"({(area < 0.02).mean():.1%})")

    print(f"\n  siblings touching the COCO frame edge, by number of sides: "
          f"{sorted(nb.items())}")
    print(f"  cut_frac (share of bbox perimeter lying along a frame edge)")
    print(f"    median {np.median(cut):.3f}  p75 {np.percentile(cut, 75):.3f}  "
          f"p90 {np.percentile(cut, 90):.3f}  >0.25: {(cut > 0.25).sum()}")

    print(f"\n  sibling-target IoU  median {np.median(iou):.3f}  "
          f"max {iou.max():.3f}  over instance_bench's 0.10: {(iou > 0.10).sum()}")

    hist("sibling area fraction", area, 0, 0.20, 10)
    hist("cut_frac", cut, 0, 1.0, 10)

    print("\n  retention table -- cases keeping >=1 usable sibling\n")
    print(f"    {'min area':>9}  {'all':>6} {'+iou<=0.10':>12} "
          f"{'+cut<=0.25':>12} {'+both':>8}")
    for thr in (0.000, 0.005, 0.010, 0.020, 0.030):
        cols = []
        for use_iou, use_cut in ((0, 0), (1, 0), (0, 1), (1, 1)):
            n = 0
            for r in rows:
                ok = [s for s in r["siblings"]
                      if s["area_frac"] >= thr
                      and (not use_iou or s.get("iou_target", 0) <= args.max_target_iou)
                      and (not use_cut or s["cut_frac"] <= args.max_cut_frac)]
                n += bool(ok)
            cols.append(n)
        print(f"    {thr:>9.3f}  {cols[0]:>6} {cols[1]:>12} {cols[2]:>12} {cols[3]:>8}")
    print(f"\n    (of {len(rows)} cases; thresholds pick themselves where the "
          "column stops moving)")

    worst = sorted(all_sib, key=lambda s: -s["cut_frac"])[:8]
    print("\n  most frame-truncated siblings:")
    for s in worst:
        owner = next(r for r in rows if any(x is s for x in r["siblings"]))
        print(f"    cut {s['cut_frac']:.2f} area {s['area_frac']:.4f} "
              f"sides {s['n_border_sides']}  {owner['case']['id']:<13} "
              f"{owner['case']['category']}")

    small = sorted(all_sib, key=lambda s: s["area_frac"])[:8]
    print("\n  smallest siblings:")
    for s in small:
        owner = next(r for r in rows if any(x is s for x in r["siblings"]))
        print(f"    area {s['area_frac']:.4f} ({s['px']:>5} px) cut "
              f"{s['cut_frac']:.2f}  {owner['case']['id']:<13} "
              f"{owner['case']['category']}")


def write_filtered(rows, args, path: Path):
    out = []
    for r in rows:
        keep = [s for s in r["siblings"]
                if s["area_frac"] >= args.min_sib_area
                and s.get("iou_target", 0) <= args.max_target_iou
                and (not args.drop_truncated or s["cut_frac"] <= args.max_cut_frac)]
        if not keep:
            continue
        case = dict(r["case"])
        keep_ids = {s["id"] for s in keep}
        case["instances"] = [
            i for i in case["instances"]
            if i["id"] == case["target_id"] or i["category"] != case["category"]
            or i["id"] in keep_ids]
        case["n_siblings"] = len(keep)
        case["sibling_stats"] = {s["id"]: {k: round(v, 4) for k, v in s.items()
                                           if k != "id"} for s in keep}
        out.append(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"source": "RefEdit-Bench x COCO, sibling-filtered", "n": len(out),
         "sibling_filter": {"min_sib_area": args.min_sib_area,
                            "max_target_iou": args.max_target_iou,
                            "max_cut_frac": args.max_cut_frac if args.drop_truncated else None},
         "cases": out}, indent=1))
    print(f"\nwrote {len(out)} cases -> {path}")
    print(f"  difficulty: {collections.Counter(c['difficulty'] for c in out).most_common()}")
    print(f"  categories: {collections.Counter(c['category'] for c in out).most_common(8)}")
    return out


def render_audit(kept, coco_index, args):
    """Overlays + contact sheet + ledger for the FILTERED case set.

    Regenerated here rather than in refedit_siblings because the audit has to
    show the benchmark as it will actually be used: a sibling the filter
    removed must not appear in red, or the auditor is judging masks that will
    not exist. Target-recovery verdicts already recorded stay valid -- sibling
    filtering does not change which instance is the target -- so carry them
    over by id rather than re-judging.
    """
    rows = {r["index"]: r for r in load_bench(args.bench)}
    out_dir = Path(args.overlay_dir)
    sample = stratified_sample(kept, args.overlays, args.overlay_seed)

    prior = {}
    if args.carry_verdicts and Path(args.carry_verdicts).exists():
        for e in json.loads(Path(args.carry_verdicts).read_text()):
            if e.get("verdict"):
                prior[e["id"]] = (e["verdict"], e.get("notes", ""))
        print(f"\ncarrying {len(prior)} verdicts from {args.carry_verdicts}")

    entries, ledger = [], []
    for case in sample:
        idx = int(case["id"].rsplit("-", 1)[1])
        row = rows.get(idx)
        if row is None:
            continue
        rec = coco_index.get(case["coco_image_id"])
        H, W = case["coco_hw"]
        anns = {a["id"]: a for a in rec["anns"]}
        masks = {}
        for inst in case["instances"]:
            a = anns.get(int(inst["id"]))
            if a is None:
                continue
            masks[inst["id"]] = apply_rect(raster(a["segmentation"], W, H),
                                           case["rect"], (512, 512))
        if case["target_id"] not in masks:
            continue
        write_overlay(row, case, masks, out_dir, outline_px=args.outline_px,
                      crop=args.crop)
        name = (f"{case['id']}_{case['difficulty']}_"
                f"{case['category'].replace(' ', '-')}_"
                f"agree{case['mask_agreement']:.2f}.png")
        v, n = prior.get(case["id"], (None, ""))
        entries.append((out_dir / name, [
            f"{case['id']} {case['difficulty']} {case['category']} "
            f"agr{case['mask_agreement']:.2f} sib{case['n_siblings']}"
            + ("  [carried]" if v else ""),
            row["editing_instruction"]]))
        ledger.append({"id": case["id"], "difficulty": case["difficulty"],
                       "category": case["category"],
                       "n_siblings": case["n_siblings"],
                       "mask_agreement": case["mask_agreement"],
                       "noun_match": case.get("noun_match"),
                       "instruction": row["editing_instruction"],
                       "verdict": v, "notes": n})

    sheet = write_contact_sheet(entries, out_dir / "contact_sheet.png")
    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_out).write_text(json.dumps(ledger, indent=1))

    st = collections.Counter(
        (c["difficulty"], "person" if c["category"] == "person" else "other")
        for c in sample)
    todo = sum(1 for e in ledger if not e["verdict"])
    print(f"\n  {len(entries)} overlays -> {out_dir}/")
    print(f"  contact sheet -> {sheet}")
    print(f"  strata: {sorted(st.items())}")
    print(f"  ledger -> {args.audit_out}  ({todo} still to judge)")


def self_test():
    W, H = 640, 480
    full = [[200, 150, 200, 330, 380, 330, 380, 150]]          # interior
    cutL = [[0, 150, 0, 330, 90, 330, 90, 150]]                # against x=0
    ann_full = {"id": 1, "segmentation": full}
    ann_cut = {"id": 2, "segmentation": cutL}

    s_full, m_full = instance_stats(ann_full, [0, 0, W, H], (W, H))
    s_cut, _ = instance_stats(ann_cut, [0, 0, W, H], (W, H))
    assert s_full["n_border_sides"] == 0, s_full
    assert s_cut["n_border_sides"] == 1, s_cut
    assert s_cut["cut_frac"] > 0.20, s_cut
    assert s_full["cut_frac"] == 0.0, s_full
    assert s_full["area_frac"] > s_cut["area_frac"]

    # overlap with the target is detected
    ovl = [[210, 160, 210, 320, 370, 320, 370, 160]]
    s_ovl, _ = instance_stats({"id": 3, "segmentation": ovl}, [0, 0, W, H],
                              (W, H), target=m_full)
    assert s_ovl["iou_target"] > 0.7, s_ovl
    s_far, _ = instance_stats({"id": 4, "segmentation":
                               [[500, 50, 500, 120, 600, 120, 600, 50]]},
                              [0, 0, W, H], (W, H), target=m_full)
    assert s_far["iou_target"] == 0.0, s_far
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="bench/cases_refedit_full.json")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--cache", default="/data/datasets/coco/desc")
    p.add_argument("--min-sib-area", type=float, default=0.01)
    p.add_argument("--max-target-iou", type=float, default=0.10)
    p.add_argument("--max-cut-frac", type=float, default=0.25)
    p.add_argument("--drop-truncated", action="store_true")
    p.add_argument("--filtered-out", default=None)
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--overlays", type=int, default=0,
                   help="regenerate N stratified overlays from the FILTERED set")
    p.add_argument("--overlay-dir", default="runs/gate_c/refedit_filtered")
    p.add_argument("--overlay-seed", type=int, default=0)
    p.add_argument("--outline-px", type=int, default=4)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--audit-out", default="runs/gate_c/audit_ledger_filtered.json")
    p.add_argument("--carry-verdicts", default=None,
                   help="an earlier ledger; verdicts for surviving cases are "
                        "carried over instead of being re-judged")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    blob = json.loads(Path(args.cases).read_text())
    cases = blob["cases"]
    coco_index, _cats, _urls = load_coco(
        Path(args.coco_annotations), tuple(args.coco_splits.split(",")),
        cache=Path(args.cache))
    rows = analyse(cases, coco_index)
    report(rows, args)
    if args.filtered_out:
        kept = write_filtered(rows, args, Path(args.filtered_out))
        if args.overlays:
            render_audit(kept, coco_index, args)


if __name__ == "__main__":
    main()
