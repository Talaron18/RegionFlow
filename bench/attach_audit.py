"""Attachment-edit count: when an edit must attach an object to the referred person's
body (hold / wear / sit on), how often does the pipeline fail with aiming correct, and
how?  (MODERN_BASE_EVAL.md, "Attachment-edit count -- scope"; rubric and bands FROZEN
there.)

Two sets, judged in ONE shuffled blind sitting (set and arm hidden):
  fresh   arm-T cases from bench/fresh_cases.py: attach edits + in-body controls on the
          same person; arms plain / paste / cropbox from `fresh_cases.py --edit`.
  refedit the ~10 RefEdit-Bench attachment cases (bench/attach_ids_refedit.json), from
          existing run_gate_c image dirs. Supporting evidence only; needs /data.

    python bench/attach_audit.py --build \
        --fresh-cases runs/fresh/attach.json --fresh-edits runs/fresh/attach_edits \
        --refedit-arm plain:runs/gate_c/img_klein4b_none \
        --refedit-arm paste:runs/gate_c/img_klein4b_comp_predoracle \
        --refedit-arm cropbox:runs/gate_c/img_klein4b_cropbox2_oracle \
        --out-dir runs/fresh/attach_blind
    # fill "verdict" in runs/fresh/attach_blind/ledger.json, then
    python bench/attach_audit.py --score runs/fresh/attach_blind \
        --aiming runs/fresh/attach_sa2va.jsonl

RUBRIC: pick the FIRST verdict that applies, in this order.
  no_object    the requested object/garment is absent, or cut off past recognition
  wrong_owner  present, but held/worn by someone else, or by no one (floating, lying)
  anatomy      on the referent, but a body part is added, missing or malformed (anyone's)
  identity     the referent's face/body/identity changed beyond the edit, or merged
  off_target   the edit is right, but another person or object also changed
  degraded     other visible damage (seams, smears, wrong scale)
  clean        none of the above
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ("no_object", "wrong_owner", "anatomy", "identity", "off_target",
            "degraded", "clean")
ATTACH_FAIL = ("wrong_owner", "anatomy")
FRESH_ARMS = ("plain", "paste", "cropbox")
PRIMARY_ARM = "cropbox"        # fixed before judging: the arm that does not cut objects


def reading(a: float, f: float, c: float) -> str:
    """a: primary-arm clean rate on aimed attach cases; f: share of its failures that
    are wrong_owner/anatomy; c: primary-arm clean rate on aimed in-body controls."""
    if a >= 0.70:
        return "attachment is NOT a major residual for this pipeline: no method target here"
    if a <= 0.50 and f >= 0.50 and c - a >= 0.25:
        return ("attachment is a SPECIFIC residual (worse than in-body edits on the same "
                "people, failing by owner/anatomy): candidate method target")
    if a <= 0.50 and c - a < 0.25:
        return "general quality problem in these scenes, not specific to attachment"
    return "mixed: report the cases"


# ----------------------------------------------------------------- panels ---

def panel(src, edit, box, instruction):
    """source (target outlined green) | edit, instruction as caption."""
    from PIL import Image, ImageDraw  # noqa: PLC0415
    src, edit = src.convert("RGB"), edit.convert("RGB")
    if edit.size != src.size:
        edit = edit.resize(src.size, Image.BICUBIC)
    s = src.copy()
    if box is not None:
        ImageDraw.Draw(s).rectangle(box, outline=(0, 255, 0),
                                    width=max(2, src.size[0] // 200))
    w, h = src.size
    cap = 28
    out = Image.new("RGB", (2 * w + 8, h + cap), "white")
    out.paste(s, (0, cap))
    out.paste(edit, (w + 8, cap))
    ImageDraw.Draw(out).text((6, 6), instruction[:160], fill=(0, 0, 0))
    return out


def scaled_box(box, orig_wh, new_wh):
    sx, sy = new_wh[0] / orig_wh[0], new_wh[1] / orig_wh[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def fresh_items(cases_path: Path, edit_dir: Path) -> list[dict]:
    from PIL import Image  # noqa: PLC0415
    items = []
    for c in json.loads(cases_path.read_text()):
        if c.get("arm") != "T":
            continue
        src_p = edit_dir / f"{c['id']}_source.png"
        if not src_p.exists():
            print(f"  {c['id']}: no edits in {edit_dir}; run fresh_cases.py --edit")
            continue
        src = Image.open(src_p)
        box = scaled_box(c["target"], Image.open(c["image"]).size, src.size)
        for arm in FRESH_ARMS:
            p = edit_dir / f"{c['id']}_{arm}.png"
            if p.exists():
                items.append({"set": "fresh", "id": c["id"], "arm": arm,
                              "edit_kind": c["edit_kind"],
                              "attach_kind": c.get("attach_kind", ""),
                              "domain": c["domain"], "instruction": c["instruction"],
                              "src": src, "edit_path": p, "box": box})
    return items


def refedit_items(ids_path: Path, arms: list[str], args) -> list[dict]:
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from refedit_siblings import bench_image, load_bench, load_coco  # noqa: PLC0415
    from run_gate_c import build_masks  # noqa: PLC0415

    ids = json.loads(ids_path.read_text())["ids"]
    cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text())["cases"]}
    coco_index, _c, _u = load_coco(Path(args.coco_annotations), ("train2014",),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}
    items = []
    for cid in ids:
        row = rows[int(cid.rsplit("-", 1)[1])]
        src = bench_image(row["input_image"]).convert("RGB").resize((512, 512))
        tmask, _s = build_masks(cases[cid], coco_index)
        t = np.asarray(tmask) > 0
        ys, xs = np.nonzero(t)
        box = [xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else None
        for spec in arms:
            name, d = spec.split(":", 1)
            p = Path(d) / f"{cid}_full.png"
            if not p.exists():
                print(f"  {cid}: no {name} image at {p}")
                continue
            items.append({"set": "refedit", "id": cid, "arm": name, "edit_kind": "attach",
                          "attach_kind": "", "domain": "photo",
                          "instruction": row["editing_instruction"],
                          "src": src, "edit_path": p,
                          "box": [int(v) for v in box] if box else None})
    _ = Image
    return items


def build(args) -> None:
    from PIL import Image  # noqa: PLC0415
    items = []
    if args.fresh_cases:
        items += fresh_items(Path(args.fresh_cases), Path(args.fresh_edits))
    if args.refedit_arm:
        items += refedit_items(Path(args.refedit_ids), args.refedit_arm, args)
    if not items:
        raise SystemExit("nothing to build")
    random.Random(args.shuffle_seed).shuffle(items)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger, key = [], []
    for n, it in enumerate(items):
        pid = f"panel_{n:04d}"
        panel(it["src"], Image.open(it["edit_path"]), it["box"],
              it["instruction"]).save(out / f"{pid}.png")
        ledger.append({"panel": pid, "verdict": None, "notes": "",
                       "instruction": it["instruction"]})
        key.append({"panel": pid, **{k: it[k] for k in
                    ("set", "id", "arm", "edit_kind", "attach_kind", "domain")}})
    (out / "ledger.json").write_text(json.dumps(ledger, indent=1))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(items)} panels -> {out}  (fill 'verdict' in ledger.json; do not open key.json)")
    print("verdicts, first that applies: " + " > ".join(VERDICTS))


# ------------------------------------------------------------------ score ---

def score(out_dir: Path, aiming_path: str | None) -> dict:
    led = {e["panel"]: e for e in json.loads((out_dir / "ledger.json").read_text())}
    key = json.loads((out_dir / "key.json").read_text())
    bad = {e["verdict"] for e in led.values() if e["verdict"]} - set(VERDICTS)
    if bad:
        raise SystemExit(f"unknown verdicts {sorted(bad)}; use {VERDICTS}")
    todo = [k["panel"] for k in key if not led[k["panel"]]["verdict"]]
    if todo:
        print(f"NOT COMPLETE: {len(todo)} panels unjudged")
    aimed = None
    if aiming_path:
        rows = [json.loads(l) for l in Path(aiming_path).read_text().splitlines() if l.strip()]
        aimed = {r["id"] for r in rows if r["cls"] == "target"}
    groups = defaultdict(Counter)
    for k in key:
        v = led[k["panel"]]["verdict"]
        if not v:
            continue
        ok = k["set"] == "refedit" or aimed is None or k["id"] in aimed
        tag = "aimed" if ok else "mis-aimed"
        groups[(k["set"], k["edit_kind"], k["arm"], tag)][v] += 1
    print("\n  set      kind    arm      aiming     n  " + " ".join(f"{v[:9]:>9}" for v in VERDICTS))
    for g in sorted(groups):
        c = groups[g]
        print(f"  {g[0]:<8} {g[1]:<7} {g[2]:<8} {g[3]:<9} {sum(c.values()):>3}  "
              + " ".join(f"{c[v]:>9}" for v in VERDICTS))

    def rate(kind):
        c = groups.get(("fresh", kind, PRIMARY_ARM, "aimed"), Counter())
        n = sum(c.values())
        return c, n
    ca, na = rate("attach")
    cc, nc = rate("inbody")
    res = {}
    if na and nc:
        a, c = ca["clean"] / na, cc["clean"] / nc
        fails = na - ca["clean"]
        f = sum(ca[v] for v in ATTACH_FAIL) / fails if fails else 0.0
        res = {"A": a, "F": f, "C": c, "n_attach": na, "n_inbody": nc,
               "reading": reading(a, f, c)}
        print(f"\n  primary arm {PRIMARY_ARM}, aimed fresh cases: attach clean {ca['clean']}/{na} "
              f"(A = {a:.0%}); owner+anatomy share of failures F = {f:.0%}; "
              f"in-body clean {cc['clean']}/{nc} (C = {c:.0%})")
        print(f"  reading: {res['reading']}")
        if aimed is None:
            print("  WARNING: no --aiming file; mis-aimed cases are counted as aimed")
    return res


# -------------------------------------------------------------- self-test ---

def self_test() -> None:
    import tempfile  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415
    assert reading(0.8, 0.9, 0.9).startswith("attachment is NOT")
    assert reading(0.4, 0.6, 0.8).startswith("attachment is a SPECIFIC")
    assert reading(0.4, 0.6, 0.5).startswith("general")
    assert reading(0.6, 0.6, 0.9).startswith("mixed")
    assert reading(0.4, 0.3, 0.9).startswith("mixed")
    assert scaled_box([10, 20, 30, 40], (100, 200), (50, 100)) == [5, 10, 15, 20]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cases = []
        for k in range(4):
            Image.new("RGB", (200, 100), "gray").save(td / f"o{k}.png")
            cases.append({"id": f"t{k}", "arm": "T", "image": str(td / f"o{k}.png"),
                          "instruction": "make the left girl hold a cup",
                          "target": [0, 0, 100, 100], "siblings": [[100, 0, 200, 100]],
                          "domain": "photo", "edit_kind": "attach" if k < 2 else "inbody",
                          "attach_kind": "hold"})
            ed = td / "ed"
            ed.mkdir(exist_ok=True)
            for a in ("source",) + FRESH_ARMS:
                Image.new("RGB", (400, 200), "red").save(ed / f"t{k}_{a}.png")
        (td / "c.json").write_text(json.dumps(cases))
        ns = argparse.Namespace(fresh_cases=str(td / "c.json"), fresh_edits=str(td / "ed"),
                                refedit_arm=[], out_dir=str(td / "blind"), shuffle_seed=1)
        build(ns)
        key = json.loads((td / "blind" / "key.json").read_text())
        assert len(key) == 12 and Image.open(td / "blind" / "panel_0000.png").size == (808, 228)
        led = json.loads((td / "blind" / "ledger.json").read_text())
        kk = {k["panel"]: k for k in key}
        for e in led:
            k = kk[e["panel"]]
            e["verdict"] = ("clean" if k["edit_kind"] == "inbody" or k["id"] == "t0"
                            else "anatomy")
        (td / "blind" / "ledger.json").write_text(json.dumps(led))
        (td / "aim.jsonl").write_text("\n".join(json.dumps({"id": f"t{k}", "cls": "target"})
                                                for k in range(4)))
        r = score(td / "blind", str(td / "aim.jsonl"))
        assert (r["A"], r["F"], r["C"]) == (0.5, 1.0, 1.0), r
        assert r["reading"].startswith("attachment is a SPECIFIC")
        # a mis-aimed case drops out of the primary rate
        (td / "aim.jsonl").write_text("\n".join(json.dumps({"id": f"t{k}", "cls":
                                      "sibling" if k == 1 else "target"}) for k in range(4)))
        r = score(td / "blind", str(td / "aim.jsonl"))
        assert (r["A"], r["n_attach"]) == (1.0, 1), r
    print("SELF-TEST OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--fresh-cases")
    p.add_argument("--fresh-edits", default="runs/fresh/attach_edits")
    p.add_argument("--refedit-arm", action="append", default=[], metavar="NAME:IMGDIR")
    p.add_argument("--refedit-ids", default="bench/attach_ids_refedit.json")
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--out-dir", default="runs/fresh/attach_blind")
    p.add_argument("--shuffle-seed", type=int, default=4242)
    p.add_argument("--score", metavar="OUT_DIR")
    p.add_argument("--aiming", metavar="JSONL", help="fresh_cases.py --run output")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    elif a.build:
        build(a)
    elif a.score:
        score(Path(a.score), a.aiming)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
