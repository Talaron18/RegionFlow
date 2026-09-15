"""
Blind multi-arm quality audit.

WHY THIS AND NOT `quality_audit.py --sample`. That tool scores ONE arm, draws
its own stratified sample from that arm's own outcome distribution, and names
every panel `<id>_<outcome>.png`. Three separate runs of it would compare three
different case sets, judged in three sittings, with the arm's identity and the
metric's verdict printed on each image. After a week of expecting mask
conditioning to help, that is not a measurement.

This is the one number the Gate C arms now rest on. Every automated figure has
either saturated (mis-targeting 0/151 on oracle -- a mask-conditioned model
cannot fail a where-did-it-change metric), lost its baseline (the checkpoint
takes in_channels=48, so there is no no-mask arm), or has no null (off-instance
ratio). Edit quality is what separates reading A from reading C, and it is
entirely hand-scored.

SO: ONE id set across all arms, panels pooled and shuffled, arm identity and
outcome class hidden until the ledger is scored.

    PAIRED. The same ids in every arm, so a verdict difference is the arm, not
    the draw. Per-arm stratification would defeat this: oracle has zero
    `mis_targeted` cases, so stratifying within it would silently pick
    different cases than the `none` arm.

    CARRY-OVER. The original 40 come first, by id, so the 18% figure stays
    comparable. Extra ids are drawn from the REFERENCE arm's strata only.

    BLIND. Panels are `panel_0000.png` in shuffled order. The key lives in a
    separate file you do not open. Nothing in the filename, the sheet caption,
    or the ledger says which arm or what the metric thought.

RUBRIC, unchanged from quality_audit.VERDICTS and imported from it. One call
the smoke run forced: the woman-in-black case, where "erase the woman" produced
a man in black, is `wrong_edit` -- a DIFFERENT edit than requested inside the
target region -- not `degraded`. Degenerated faces are `destroyed`. Those two
land on opposite sides of the clean rate.

    python bench/blind_audit.py --build \\
        --arm none:runs/gate_c/refedit_sd3.jsonl:runs/gate_c/edits \\
        --arm white:runs/gate_c/ue_white.jsonl:runs/gate_c/img_white \\
        --arm oracle:runs/gate_c/ue_oracle.jsonl:runs/gate_c/img_oracle \\
        --reference none \\
        --carry-over runs/gate_c/quality_ledger.json --extra 20

    # ... fill "verdict" in runs/gate_c/blind_ledger.json, from the panels ...

    python bench/blind_audit.py --score runs/gate_c/blind_ledger.json \\
        --key runs/gate_c/blind_key.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from quality_audit import (METRIC_BLIND, VERDICTS, load_bench, load_coco,
                           masks_for_case, sheet, side_by_side, stratified)
from refedit_siblings import bench_image
from run_gate_c import outcome_class


def read_arm(spec: str):
    name, results, edits = spec.split(":")
    recs = [json.loads(l) for l in Path(results).read_text().splitlines()
            if l.strip()]
    for r in recs:
        r.setdefault("outcome", outcome_class(r))
    return name, {r["id"]: r for r in recs}, Path(edits)


def build(args) -> None:
    arms = [read_arm(s) for s in args.arm]
    names = [a[0] for a in arms]
    if args.reference not in names:
        raise SystemExit(f"--reference {args.reference!r} not among {names}")
    ref = dict(zip(names, arms))[args.reference][1]

    cases = {c["id"]: c for c in
             json.loads(Path(args.cases).read_text())["cases"]}
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    # ---- the id set, fixed once, before a single panel is rendered --------
    carried = []
    if args.carry_over:
        carried = [e["id"] for e in
                   json.loads(Path(args.carry_over).read_text())]
        print(f"carrying over {len(carried)} ids from {args.carry_over}")
    pool = [r for i, r in ref.items()
            if i in cases and i not in set(carried)
            and r["prompt_kind"] == args.prompt_kind]
    extra = [r["id"] for r in stratified(pool, args.extra, args.seed)]
    ids = carried + extra
    missing = [i for i in ids if i not in cases]
    if missing:
        raise SystemExit(f"{len(missing)} carried ids are not in --cases: "
                         f"{missing[:5]} -- wrong cases file?")
    print(f"{len(carried)} carried + {len(extra)} new = {len(ids)} ids "
          f"x {len(arms)} arms = {len(ids) * len(arms)} panels")

    # ---- render every (id, arm) ------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels, skipped = [], collections.Counter()

    for cid in ids:
        case = cases[cid]
        idx = int(cid.rsplit("-", 1)[1])
        row, rec = rows.get(idx), coco_index.get(case["coco_image_id"])
        if row is None or rec is None:
            skipped["no bench row or coco record"] += 1
            continue
        tmask, smasks = masks_for_case(case, rec)
        if tmask is None:
            skipped["target mask empty"] += 1
            continue
        src = bench_image(row["input_image"])

        for name, recs, edits in arms:
            r = recs.get(cid)
            path = edits / f"{cid}_{args.prompt_kind}.png"
            if r is None or not path.exists():
                skipped[f"{name}: no edit image"] += 1
                continue
            panels.append({
                "id": cid, "arm": name,
                "img": side_by_side(src, Image.open(path), tmask, smasks),
                "instruction": row["editing_instruction"],
                "outcome": r["outcome"], "valid": r["valid"],
                "difficulty": case["difficulty"], "category": case["category"],
                "target_change": r["target_change"],
                "leakage_ratio": r["leakage_ratio"],
            })

    if skipped:
        print("skipped: " + ", ".join(f"{k} x{v}" for k, v in skipped.items()))

    # ---- shuffle, then write panels under opaque names --------------------
    # A DIFFERENT seed from the sampling seed. Sharing one would make panel
    # order a deterministic function of the draw, and anyone who reran the
    # build could recover the key without opening it.
    random.Random(args.shuffle_seed).shuffle(panels)

    ledger, key, entries = [], [], []
    for n, p in enumerate(panels):
        pid = f"panel_{n:04d}"
        path = out_dir / f"{pid}.png"
        p["img"].save(path)
        # CAPTION CARRIES THE INSTRUCTION AND NOTHING ELSE. The verdict needs
        # to know what was asked; it must not know what the metric concluded
        # or which arm produced it.
        entries.append((path, [pid, p["instruction"][:110]]))
        ledger.append({"panel": pid, "verdict": None, "notes": "",
                       "instruction": p["instruction"]})
        key.append({k: p[k] for k in
                    ("id", "arm", "outcome", "valid", "difficulty",
                     "category", "target_change", "leakage_ratio")}
                   | {"panel": pid})

    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).write_text(json.dumps(ledger, indent=1))
    Path(args.key).write_text(json.dumps(key, indent=1))
    s = sheet(entries, out_dir / "blind_sheet.png")

    print(f"\n{len(ledger)} panels -> {out_dir}/")
    print(f"contact sheet -> {s}   left = source, right = edited")
    print(f"ledger  -> {args.ledger}   <- fill 'verdict' here")
    print(f"KEY     -> {args.key}      <- DO NOT OPEN until scoring")
    print("\nverdicts: " + " | ".join(VERDICTS))
    print("  wrong_edit = a DIFFERENT edit than asked, inside the target "
          "(erase the woman -> a man)")
    print("  destroyed  = target structurally lost (face gone, anatomy broken)")
    print("  both are scored as SUCCESSES by instance_selectivity")


def score(args) -> None:
    led = {e["panel"]: e for e in json.loads(Path(args.score).read_text())}
    key = json.loads(Path(args.key).read_text())
    done = [k | led[k["panel"]] for k in key
            if led.get(k["panel"], {}).get("verdict")]
    print(f"{len(done)}/{len(key)} judged")
    if not done:
        return
    bad = {e["verdict"] for e in done} - set(VERDICTS)
    if bad:
        raise SystemExit(f"unknown verdicts: {sorted(bad)}")

    arms = sorted({e["arm"] for e in done})
    print(f"\n  VERDICTS BY ARM")
    print(f"    {'arm':<10} {'n':>4}  " + "".join(f"{v:>11}" for v in VERDICTS)
          + f"{'clean %':>10}")
    for a in arms:
        rs = [e for e in done if e["arm"] == a]
        c = collections.Counter(e["verdict"] for e in rs)
        print(f"    {a:<10} {len(rs):>4}  "
              + "".join(f"{c[v]:>11}" for v in VERDICTS)
              + f"{c['clean'] / len(rs):>9.0%}")

    # PAIRED. Only ids judged in EVERY arm, so a clean-rate difference is the
    # arm and not which cases happened to get scored first.
    per = collections.defaultdict(dict)
    for e in done:
        per[e["id"]][e["arm"]] = e["verdict"]
    full = [i for i, d in per.items() if len(d) == len(arms)]
    print(f"\n  PAIRED on {len(full)} ids judged in all {len(arms)} arms")
    if full:
        for a in arms:
            n = sum(1 for i in full if per[i][a] == "clean")
            print(f"    {a:<10} clean {n}/{len(full)} = {n / len(full):.0%}")
        if args.reference in arms:
            print(f"\n    transitions vs '{args.reference}':")
            for a in arms:
                if a == args.reference:
                    continue
                t = collections.Counter(
                    (per[i][args.reference], per[i][a]) for i in full)
                worse = sum(v for (x, y), v in t.items()
                            if x == "clean" and y != "clean")
                better = sum(v for (x, y), v in t.items()
                             if x != "clean" and y == "clean")
                print(f"      {a:<10} clean->not {worse}   not->clean {better}")

    # The conditional every leakage figure inherits.
    print(f"\n  AMONG CASES THAT PRODUCE THE LEAKAGE NUMBERS (valid+measurable)")
    for a in arms:
        rs = [e for e in done if e["arm"] == a and e["valid"]
              and e["outcome"] == "measurable"]
        if not rs:
            print(f"    {a:<10} none judged")
            continue
        clean = sum(1 for e in rs if e["verdict"] == "clean")
        blind = sum(1 for e in rs if e["verdict"] in METRIC_BLIND)
        print(f"    {a:<10} n={len(rs):<3} clean {clean / len(rs):>4.0%}   "
              f"destroyed/wrong_edit {blind / len(rs):>4.0%}  <- metric calls "
              f"these successes")

    print("""
  READING. RefEdit-SD3's rate was 18% (7/40).
    oracle at or below ~18%  -> branch A: conditioning solved localisation and
                                did not buy edit quality. Direction is the
                                residual, and it is RegionFlow's target.
    oracle well above ~18%   -> branch C: mask conditioning is most of the win.
                                Pre-committed. Report it.""")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=[],
                   metavar="NAME:RESULTS:EDITDIR")
    p.add_argument("--reference", default="none")
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--prompt-kind", default="full", choices=("full", "single"))
    p.add_argument("--carry-over", default="runs/gate_c/quality_ledger.json",
                   help="ledger whose ids are reused so the 18% stays comparable")
    p.add_argument("--extra", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="sampling")
    p.add_argument("--shuffle-seed", type=int, default=1729,
                   help="panel order; deliberately NOT the sampling seed")
    p.add_argument("--out-dir", default="runs/gate_c/blind")
    p.add_argument("--ledger", default="runs/gate_c/blind_ledger.json")
    p.add_argument("--key", default="runs/gate_c/blind_key.json")
    p.add_argument("--build", action="store_true")
    p.add_argument("--score", default=None)
    args = p.parse_args()

    if args.score:
        score(args)
    elif args.build:
        if not args.arm:
            raise SystemExit("--build needs at least one --arm")
        build(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
