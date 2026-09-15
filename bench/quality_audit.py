"""
Audit target-region EDIT QUALITY, which instance_selectivity cannot see.

WHY. `instance_selectivity` measures WHERE change happened, never whether the
change is any good. Damage inside the target mask counts as a successful edit,
so the more thoroughly a model destroys the target, the more confidently the
case passes `valid`. A model could post a perfect leakage ratio by obliterating
every target and leaving siblings alone.

The hand review of RefEdit-SD3 outputs found this is not hypothetical:

    plane's shape completely lost            -> metric: large target_change
    head twisted 180 degrees                 -> metric: large target_change
    limb twisted, racket floating            -> metric: large target_change
    eyes removed instead of the glasses      -> metric: large target_change
    face lost                                -> metric: large target_change
    whole saddle removed, not just the word  -> metric: large target_change

Every one reads as a clean, confident edit. This module turns those notes into
a rate, with the same discipline as the benchmark audit: stratified sample,
written verdicts, reported denominator.

THE CROSS-TAB IS THE POINT. Scoring the filled ledger reports quality BY the
metric's own outcome class -- in particular, among the cases that actually
contribute to the reported leakage ratio (valid + measurable), what fraction
are clean edits? That number scopes every leakage figure in the paper.

VERDICTS

    clean       the requested edit happened, target looks plausible
    degraded    requested edit happened but with artifacts -- phantom
                structures, implausible lighting, mild distortion
    destroyed   target structurally lost: anatomy broken, shape gone, face
                or limbs missing
    wrong_edit  a DIFFERENT edit than requested inside the target region
                (saddle removed instead of the word; glasses restyled rather
                than removed)
    no_edit     target essentially unchanged

`destroyed` and `wrong_edit` are the ones the metric scores as successes.

USAGE
    python bench/quality_audit.py --self-test
    python bench/quality_audit.py --sample 40 --edits runs/gate_c/edits
    python bench/quality_audit.py --score runs/gate_c/quality_ledger.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from frame_fit import apply_rect
from refedit_siblings import bench_image, load_bench, load_coco, raster

VERDICTS = ("clean", "degraded", "destroyed", "wrong_edit", "no_edit")
#: Verdicts the metric cannot distinguish from a good edit.
METRIC_BLIND = ("destroyed", "wrong_edit")


def outline(mask: np.ndarray, px: int = 3) -> np.ndarray:
    inner = mask.copy()
    for _ in range(max(1, px)):
        inner = (inner & np.pad(inner, 1)[:-2, 1:-1] & np.pad(inner, 1)[2:, 1:-1]
                 & np.pad(inner, 1)[1:-1, :-2] & np.pad(inner, 1)[1:-1, 2:])
    return mask & ~inner


def side_by_side(src: Image.Image, edited: Image.Image, tmask, smasks) -> Image.Image:
    """Source | edited, same outlines on both so the eye compares like with like.

    Outline only -- no fill. A filled overlay hides exactly what this audit is
    looking for: what happened to the pixels inside the target.
    """
    t_edge = outline(tmask)
    s_edge = np.zeros_like(t_edge)
    for m in smasks:
        s_edge |= outline(m)

    panels = []
    for im in (src, edited):
        a = np.asarray(im.convert("RGB")).astype(np.float32).copy()
        if a.shape[:2] != t_edge.shape:
            a = np.asarray(im.convert("RGB").resize(
                (t_edge.shape[1], t_edge.shape[0]), Image.BILINEAR)).astype(np.float32)
        a[s_edge] = np.array([235, 60, 60], np.float32)
        a[t_edge] = np.array([255, 235, 40], np.float32)
        panels.append(Image.fromarray(a.astype(np.uint8)))

    h = panels[0].height
    out = Image.new("RGB", (panels[0].width + panels[1].width + 4, h), (20, 20, 24))
    out.paste(panels[0], (0, 0))
    out.paste(panels[1], (panels[0].width + 4, 0))
    return out


def sheet(entries, out_path: Path, cols=2, thumb_w=520, cap=30):
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    th = thumb_w // 2
    rows = (len(entries) + cols - 1) // cols
    W = cols * (thumb_w + 6) + 6
    H = rows * (th + cap + 6) + 6
    canvas = Image.new("RGB", (W, H), (22, 22, 26))
    dr = ImageDraw.Draw(canvas)
    for i, (path, lines) in enumerate(entries):
        r, c = divmod(i, cols)
        x, y = 6 + c * (thumb_w + 6), 6 + r * (th + cap + 6)
        try:
            im = Image.open(path).convert("RGB").resize((thumb_w, th), Image.BILINEAR)
        except Exception:  # noqa: BLE001
            continue
        canvas.paste(im, (x, y))
        for k, line in enumerate(lines[:2]):
            dr.text((x + 2, y + th + 2 + k * 13), line[:110],
                    fill=(230, 230, 235) if k == 0 else (165, 165, 175), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def stratified(recs, n, seed=0):
    """Spread across outcome class x person/other.

    Outcome class is the primary stratum: the question is what quality looks
    like WITHIN each class the metric assigns, especially `measurable`, whose
    cases are the ones that produce the reported leakage numbers.
    """
    buckets = collections.defaultdict(list)
    for r in recs:
        buckets[(r["outcome"], r["category"] == "person")].append(r)
    rng = np.random.default_rng(seed)
    for v in buckets.values():
        rng.shuffle(v)
    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    out = []
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def build(args):
    recs = [json.loads(l) for l in Path(args.results).read_text().splitlines()
            if l.strip()]
    from run_gate_c import outcome_class  # noqa: PLC0415
    for r in recs:
        r.setdefault("outcome", outcome_class(r))
    recs = [r for r in recs if r["prompt_kind"] == args.prompt_kind]

    cases = {c["id"]: c for c in
             json.loads(Path(args.cases).read_text())["cases"]}
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}

    sample = stratified([r for r in recs if r["id"] in cases], args.sample, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, ledger = [], []
    for r in sample:
        case = cases[r["id"]]
        idx = int(r["id"].rsplit("-", 1)[1])
        row, rec = rows.get(idx), coco_index.get(case["coco_image_id"])
        edit_path = Path(args.edits) / f"{r['id']}_{args.prompt_kind}.png"
        if row is None or rec is None or not edit_path.exists():
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
                              and i["id"] != case["target_id"]) if x is not None]

        img = side_by_side(bench_image(row["input_image"]),
                           Image.open(edit_path), tmask, smasks)
        name = f"{r['id']}_{r['outcome']}.png"
        img.save(out_dir / name)
        entries.append((out_dir / name, [
            f"{r['id']}  {r['outcome']}  {case['difficulty']}  {case['category']}"
            f"  tgt {r['target_change']:.3f}  ratio {r['leakage_ratio']:.2f}"
            f"  valid={r['valid']}",
            row["editing_instruction"]]))
        ledger.append({"id": r["id"], "outcome": r["outcome"],
                       "difficulty": case["difficulty"],
                       "category": case["category"],
                       "target_change": r["target_change"],
                       "leakage_ratio": r["leakage_ratio"],
                       "valid": r["valid"],
                       "instruction": row["editing_instruction"],
                       "verdict": None, "notes": ""})

    s = sheet(entries, out_dir / "quality_sheet.png")
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).write_text(json.dumps(ledger, indent=1))
    print(f"{len(ledger)} pairs -> {out_dir}/")
    print(f"contact sheet -> {s}   left = source, right = edited")
    print("yellow = target outline, red = siblings (OUTLINE only, so the "
          "target pixels stay visible)")
    print(f"ledger -> {args.ledger}")
    print(f"strata: {sorted(collections.Counter((l['outcome'], l['category'] == 'person') for l in ledger).items())}")
    print("\nverdicts: " + " | ".join(VERDICTS))
    print("  destroyed and wrong_edit are what the metric scores as SUCCESSES")


def score(path: Path):
    led = json.loads(path.read_text())
    done = [e for e in led if e.get("verdict")]
    print(f"{len(done)}/{len(led)} judged")
    if not done:
        return
    v = collections.Counter(e["verdict"] for e in done)
    print("\n  verdicts: " + "  ".join(f"{k} {v[k]} ({v[k] / len(done):.0%})"
                                       for k in VERDICTS if v[k]))

    print("\n  quality BY the metric's outcome class")
    print(f"    {'outcome':<13} {'n':>3}  " + "".join(f"{k:>11}" for k in VERDICTS))
    for oc in ("measurable", "mis_targeted", "no_change"):
        rs = [e for e in done if e["outcome"] == oc]
        if not rs:
            continue
        c = collections.Counter(e["verdict"] for e in rs)
        print(f"    {oc:<13} {len(rs):>3}  "
              + "".join(f"{c[k]:>11}" for k in VERDICTS))

    contrib = [e for e in done if e["valid"] and e["outcome"] == "measurable"]
    if contrib:
        blind = sum(1 for e in contrib if e["verdict"] in METRIC_BLIND)
        clean = sum(1 for e in contrib if e["verdict"] == "clean")
        print(f"\n  CASES THAT PRODUCE THE REPORTED LEAKAGE NUMBERS "
              f"(valid + measurable): {len(contrib)}")
        print(f"    clean edits:                   {clean}/{len(contrib)} "
              f"= {clean / len(contrib):.0%}")
        print(f"    destroyed or wrong_edit:       {blind}/{len(contrib)} "
              f"= {blind / len(contrib):.0%}  <- scored as successes")
        print("    -> every leakage figure is conditional on this. Quote it "
              "with them.")


def self_test():
    m = np.zeros((64, 64), bool); m[20:40, 20:40] = True
    e = outline(m, 3)
    assert e.sum() > 0 and e.sum() < m.sum()
    assert not e[30, 30] and e[20, 30]

    a = Image.fromarray(np.full((512, 512, 3), 90, np.uint8))
    b = Image.fromarray(np.full((512, 512, 3), 160, np.uint8))
    t = np.zeros((512, 512), bool); t[100:300, 100:300] = True
    s = np.zeros((512, 512), bool); s[350:450, 350:450] = True
    img = side_by_side(a, b, t, [s])
    assert img.size == (1028, 512), img.size
    arr = np.asarray(img)
    # the interior of the target must be UNMODIFIED on both panels
    assert arr[200, 200, 0] == 90 and arr[200, 200 + 516, 0] == 160

    recs = [{"id": f"r-{i}", "outcome": ["measurable", "mis_targeted", "no_change"][i % 3],
             "category": "person" if i % 2 else "car"} for i in range(30)]
    sel = stratified(recs, 12, seed=0)
    assert len(sel) == 12 and len({r["id"] for r in sel}) == 12
    oc = collections.Counter(r["outcome"] for r in sel)
    assert min(oc.values()) >= 3, oc
    print("self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="runs/gate_c/refedit_sd3.jsonl")
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--edits", default="runs/gate_c/edits")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--prompt-kind", default="full", choices=("full", "single"))
    p.add_argument("--sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/gate_c/quality")
    p.add_argument("--ledger", default="runs/gate_c/quality_ledger.json")
    p.add_argument("--score", default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif args.score:
        score(Path(args.score))
    else:
        build(args)


if __name__ == "__main__":
    main()
