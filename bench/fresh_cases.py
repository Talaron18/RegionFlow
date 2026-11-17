"""Fresh-case aiming test: does Sa2VA still find the referred instance on images and
referring expressions it cannot have seen in RefCOCO?  (MODERN_BASE_EVAL.md, section
"Fresh-case aiming test -- scope"; bands and case rules FROZEN there.)

    # 1. write cases (GUI: drag the TARGET box first, then each same-category sibling;
    #    keys: u = undo last box, n = done with this image).  Any tool that gives
    #    pixel boxes [x0, y0, x1, y1] in the original image works instead.
    python bench/fresh_cases.py --annotate IMAGE_DIR --cases runs/fresh/cases.json
    python bench/fresh_cases.py --check --cases runs/fresh/cases.json   # rules, no GPU
    # 2. segment (Sa2VA venv), same frozen prompt and scorer as predict_masks.py
    python bench/fresh_cases.py --run --cases runs/fresh/cases.json \
        --out runs/fresh/sa2va.jsonl --out-dir runs/fresh/masks \
        --save-overlays runs/fresh/overlays
    # smoke: arm "S" cases (any number), then edit with klein (klein container):
    python bench/fresh_cases.py --run --cases runs/fresh/smoke.json --out runs/fresh/smoke.jsonl \
        --out-dir runs/fresh/smoke_masks --save-overlays runs/fresh/smoke_overlays
    python bench/fresh_cases.py --edit --cases runs/fresh/smoke.json \
        --out-dir runs/fresh/smoke_masks --edit-dir runs/fresh/smoke_edits
    # 3. after viewing every overlay
    python bench/fresh_cases.py --report runs/fresh/sa2va.jsonl --cases runs/fresh/cases.json \
        [--overrides runs/fresh/overrides.json]

GROUND TRUTH IS BOXES, not masks: target box T, sibling boxes S_k (S_k minus T, as in
predict_masks.score_masks, so overlap pixels count for the target only). Boxes are
cruder than COCO masks, so every overlay is viewed, and any class the viewer disagrees
with goes in overrides.json as {"id": {"cls": "...", "why": "..."}}. The report prints
the band from the scorer AND with overrides, and says if they differ.

Images are resized to 512x512 exactly as the RefEdit-Bench run did.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from predict_masks import (PROMPT, classify, load_done, make_backend, process_case,
                           read_jsonl, save_overlay)

SIZE = 512
EXPR_TYPES = ("attribute", "spatial", "relation")
ARMS = ("A", "B", "S", "T")  # A: non-COCO images (the band); B: COCO/RefEdit images, fresh text;
                           # S: smoke cases -- seen before freezing, never in any count;
                           # T: attachment-edit count (bench/attach_audit.py)
DOMAINS = ("photo", "illustration", "painting")
A_MIN = {"photo": 25, "illustration": 15, "painting": 15}     # arm A: 60 cases at least
EDIT_KINDS = ("attach", "inbody")
ATTACH_KINDS = ("hold", "wear", "onbody")
T_MIN = {"attach": 30, "inbody": 15, "hold": 10, "wear": 8, "onbody": 5, "domain": 8}
# Band on arm A target rate (fixed 2026-11-18, before any case was written).
BANDS = ((0.70, "<= 70%: aiming is OPEN off RefCOCO; the learned-grounding line returns"),
         (0.85, "71-84%: mixed; report per type, no line reopened or closed"),
         (1.01, ">= 85%: aiming is solved in general for this pipeline"))


def band(rate: float) -> str:
    if rate <= 0.70:
        return BANDS[0][1]
    if rate < 0.85:
        return BANDS[1][1]
    return BANDS[2][1]


def box_mask(box, orig_wh, size=SIZE) -> np.ndarray:
    """[x0, y0, x1, y1] in original pixels -> bool (size, size) after the same
    non-aspect-preserving resize the images get."""
    w, h = orig_wh
    x0, y0, x1, y1 = box
    sx, sy = size / w, size / h
    c0, c1 = sorted((int(round(x0 * sx)), int(round(x1 * sx))))
    r0, r1 = sorted((int(round(y0 * sy)), int(round(y1 * sy))))
    m = np.zeros((size, size), bool)
    m[max(r0, 0):min(r1, size), max(c0, 0):min(c1, size)] = True
    return m


def check_size(case: dict, wh) -> None:
    """Boxes are in the pixels of the image they were drawn on. A resized copy would
    silently move every box (the smoke classroom photo was 2364x1330, boxed at
    2000x1125), so a case that records `image_wh` must match the file."""
    want = case.get("image_wh")
    if want is not None and tuple(want) != tuple(wh):
        raise SystemExit(f"{case['id']}: {case['image']} is {wh[0]}x{wh[1]} but its boxes "
                         f"were drawn at {want[0]}x{want[1]}; redraw or rescale them")


def check_cases(cases: list[dict]) -> list[str]:
    """Rule violations (empty list = OK). Rules are the scope's, not style."""
    errs, ids, per_img = [], set(), Counter()
    for c in cases:
        i = c.get("id", "?")
        for k in ("id", "image", "instruction", "target", "siblings", "expr_type", "arm"):
            if k not in c:
                errs.append(f"{i}: missing {k}")
        if i in ids:
            errs.append(f"{i}: duplicate id")
        ids.add(i)
        if c.get("expr_type") not in EXPR_TYPES:
            errs.append(f"{i}: expr_type must be one of {EXPR_TYPES}")
        if c.get("arm") not in ARMS:
            errs.append(f"{i}: arm must be one of {ARMS}")
        if c.get("arm") in ("A", "B", "T") and c.get("domain") not in DOMAINS:
            errs.append(f"{i}: domain must be one of {DOMAINS}")
        if c.get("arm") == "T":
            if c.get("edit_kind") not in EDIT_KINDS:
                errs.append(f"{i}: edit_kind must be one of {EDIT_KINDS}")
            if c.get("edit_kind") == "attach" and c.get("attach_kind") not in ATTACH_KINDS:
                errs.append(f"{i}: attach_kind must be one of {ATTACH_KINDS}")
        if not c.get("siblings"):
            errs.append(f"{i}: needs at least one same-category sibling box")
        for b in [c.get("target")] + list(c.get("siblings") or []):
            if not (isinstance(b, (list, tuple)) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]):
                errs.append(f"{i}: bad box {b} (want [x0, y0, x1, y1], x1>x0, y1>y0)")
        if c.get("arm") in ("A", "B", "T"):   # smoke cases may reuse an image freely
            per_img[c.get("image")] += 1
    for img, n in per_img.items():
        if n > 2:
            errs.append(f"{img}: {n} counted cases (max 2 per image)")
    a = [c for c in cases if c.get("arm") == "A"]
    if a and any(sum(c.get("expr_type") == t for c in a) < 10 for t in EXPR_TYPES):
        errs.append("arm A: needs >= 10 cases of each expr_type "
                    + str(Counter(c.get("expr_type") for c in a)))
    if a and any(sum(c.get("domain") == d for c in a) < n for d, n in A_MIN.items()):
        errs.append(f"arm A: needs per-domain minimums {A_MIN}, has "
                    + str(Counter(c.get("domain") for c in a)))
    # arm A and arm T must not share images: T's images are segmented and edited
    # in the open, A's must be unseen until A is frozen.
    shared = {c["image"] for c in a} & {c["image"] for c in cases if c.get("arm") == "T"}
    if shared:
        errs.append(f"arms A and T share {len(shared)} image(s), e.g. {sorted(shared)[:2]}")
    errs += check_t([c for c in cases if c.get("arm") == "T"])
    return errs


def check_t(t: list[dict]) -> list[str]:
    """Arm T: attachment edits plus in-body controls on the same person."""
    if not t:
        return []
    errs = []
    att = [c for c in t if c.get("edit_kind") == "attach"]
    inb = [c for c in t if c.get("edit_kind") == "inbody"]
    have = Counter(c.get("attach_kind") for c in att)
    if len(att) < T_MIN["attach"] or len(inb) < T_MIN["inbody"]:
        errs.append(f"arm T: needs >= {T_MIN['attach']} attach and >= {T_MIN['inbody']} "
                    f"inbody cases, has {len(att)} / {len(inb)}")
    for k in ATTACH_KINDS:
        if have[k] < T_MIN[k]:
            errs.append(f"arm T: needs >= {T_MIN[k]} attach_kind={k}, has {have[k]}")
    dom = Counter(c.get("domain") for c in att)
    if any(dom[d] < T_MIN["domain"] for d in DOMAINS):
        errs.append(f"arm T: needs >= {T_MIN['domain']} attach cases per domain, has {dict(dom)}")
    pairs = {(c["image"], tuple(c["target"])) for c in att if "target" in c}
    for c in inb:
        if (c.get("image"), tuple(c.get("target") or ())) not in pairs:
            errs.append(f"{c.get('id')}: inbody control must share image and target box "
                        "with an attach case")
    return errs


# ------------------------------------------------------------------ run ---

def run(args):
    from PIL import Image  # noqa: PLC0415

    cases = json.loads(Path(args.cases).read_text())
    errs = check_cases(cases)
    if errs:
        raise SystemExit("cases fail the scope's rules:\n  " + "\n  ".join(errs))
    out_path, out_dir = Path(args.out), Path(args.out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov = Path(args.save_overlays) if args.save_overlays else None
    if ov:
        ov.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path, args.backend, args.model_path if args.backend == "sa2va" else "",
                     args.overwrite)
    backend = make_backend(args.backend, args.model_path, args.device)
    token = getattr(backend, "image_token", "")
    with out_path.open("a") as fh:
        for n, c in enumerate(cases):
            if (c["id"], "fresh") in done:
                continue
            img = Image.open(c["image"]).convert("RGB")
            wh = img.size
            check_size(c, wh)
            src = img.resize((SIZE, SIZE), Image.BILINEAR)
            T = box_mask(c["target"], wh)
            S = [box_mask(b, wh) for b in c["siblings"]]
            rec, P = process_case(c["id"], "fresh", args.backend,
                                  args.model_path if args.backend == "sa2va" else "",
                                  backend, src, token + PROMPT.format(instruction=c["instruction"]),
                                  T, S, out_dir)
            rec.update(arm=c["arm"], expr_type=c["expr_type"], gt="boxes",
                       domain=c.get("domain", ""), edit_kind=c.get("edit_kind", ""),
                       attach_kind=c.get("attach_kind", ""))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if ov:
                save_overlay(src, P, T, S, ov / f"{c['id']}.png")
            print(f"  {n + 1}/{len(cases)} {c['id']} [{c['arm']}/{c['expr_type']}] {rec['cls']}  "
                  f"in_T {rec['in_T']:.2f} in_S {rec['in_S']:.2f}")
    report(out_path, args.cases, None)


# ----------------------------------------------------------------- edit ---

def edit(args):
    """klein 4B on each case whose Sa2VA mask exists: plain edit, paste-back
    through the mask (dilated 8 px at 512, scaled), and crop-and-paste-the-box.
    Same seed for all three. Runs at ~1 MP with the image's own aspect, not the
    512x512 squash the segmenter sees; the 512 mask is resized NEAREST."""
    from PIL import Image  # noqa: PLC0415

    from flux2_klein_model import Flux2KleinModel, composite, model_size  # noqa: PLC0415
    from run_gate_c import case_seed, dilate  # noqa: PLC0415
    from predict_masks import load_mask  # noqa: PLC0415

    cases = json.loads(Path(args.cases).read_text())
    out = Path(args.edit_dir)
    out.mkdir(parents=True, exist_ok=True)
    plain = Flux2KleinModel(device=args.device, offload=args.offload)
    crop = Flux2KleinModel(pipe=plain.pipe, device=args.device, offload=args.offload,
                           crop=True, crop_paste="box")
    for c in cases:
        mpath = Path(args.out_dir) / f"{c['id']}.png"
        if not mpath.exists():
            print(f"  {c['id']}: no mask at {mpath}; run --run first")
            continue
        img = Image.open(c["image"]).convert("RGB")
        check_size(c, img.size)
        w, h = model_size(*img.size, area=1024 * 1024)
        img = img.resize((w, h), Image.BICUBIC)
        r = max(1, round(8 * max(w, h) / SIZE))
        m512 = load_mask(mpath)
        m = np.asarray(Image.fromarray((m512 * 255).astype(np.uint8)).resize(
            (w, h), Image.NEAREST)) >= 128
        mimg = Image.fromarray((dilate(m, r) * 255).astype(np.uint8))
        seed = case_seed(c["id"], 0)
        e = plain(img, c["instruction"], seed)
        img.save(out / f"{c['id']}_source.png")
        e.save(out / f"{c['id']}_plain.png")
        composite(img, e, mimg, plain.feather).save(out / f"{c['id']}_paste.png")
        if m.any():
            crop(img, c["instruction"], seed, mask=mimg).save(out / f"{c['id']}_cropbox.png")
        else:
            print(f"  {c['id']}: empty mask, no crop arm")
        print(f"  {c['id']}: {w}x{h}, dilate {r}px, seed {seed} -> {out}/{c['id']}_*.png")


# --------------------------------------------------------------- report ---

def summarise(recs: list[dict], overrides: dict | None = None) -> dict:
    out = {}
    for arm in ARMS:
        rs = [r for r in recs if r.get("arm") == arm]
        if not rs:
            continue
        cls = [(overrides or {}).get(r["id"], {}).get("cls", r["cls"]) for r in rs]
        by_type = {}
        for t in EXPR_TYPES:
            ct = [k for r, k in zip(rs, cls) if r["expr_type"] == t]
            by_type[t] = (sum(k == "target" for k in ct), len(ct))
        by_domain = {}
        for d in DOMAINS:
            cd = [k for r, k in zip(rs, cls) if r.get("domain") == d]
            if cd:
                by_domain[d] = (sum(k == "target" for k in cd), len(cd))
        hit = sum(k == "target" for k in cls)
        out[arm] = {"n": len(rs), "target": hit, "rate": hit / len(rs),
                    "classes": dict(Counter(cls)), "by_type": by_type,
                    "by_domain": by_domain}
    return out


def report(path, cases_path, overrides_path) -> None:
    recs = {r["id"]: r for r in read_jsonl(path)}          # last row per id wins
    recs = list(recs.values())
    if cases_path:
        want = {c["id"] for c in json.loads(Path(cases_path).read_text())}
        missing = want - {r["id"] for r in recs}
        if missing:
            print(f"NOT COMPLETE: {len(missing)} cases have no row, e.g. {sorted(missing)[:5]}")
    ovr = json.loads(Path(overrides_path).read_text()) if overrides_path else {}
    for label, o in (("scorer (boxes)", None), ("with overrides", ovr)):
        if label == "with overrides" and not ovr:
            continue
        print(f"== {label}")
        for arm, s in summarise(recs, o).items():
            print(f"  arm {arm}: target {s['target']}/{s['n']} ({100 * s['rate']:.1f}%)  "
                  f"classes {s['classes']}")
            print("    by type: " + "  ".join(f"{t} {a}/{b}" for t, (a, b) in s["by_type"].items()))
            if s["by_domain"]:
                print("    by domain: " + "  ".join(f"{d} {a}/{b}"
                                                  for d, (a, b) in s["by_domain"].items()))
            if arm == "A":
                print(f"    band (all domains): {band(s['rate'])}")
                for d, (h, n) in s["by_domain"].items():
                    if h / n <= 0.70:
                        print(f"    domain reading: aiming is OPEN for {d} ({h}/{n} <= 70%)")
    if ovr:
        a0 = summarise(recs).get("A")
        a1 = summarise(recs, ovr).get("A")
        if a0 and a1 and band(a0["rate"]) != band(a1["rate"]):
            print("  NOTE: overrides move arm A across a band edge; report both.")
        print(f"  overrides: {len(ovr)} " + "; ".join(f"{k}: {v.get('why', '')}"
                                                   for k, v in ovr.items()))
    print("  RefEdit-Bench reference (COCO masks, same prompt): 137/151 (90.7%)")


# ------------------------------------------------------------- annotate ---

def annotate(image_dir: Path, cases_path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.widgets import RectangleSelector  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    cases = json.loads(cases_path.read_text()) if cases_path.exists() else []
    imgs = sorted(p for p in image_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    print("Per image: decide the target and write the instruction FIRST (terminal), then "
          "drag the target box, then every same-category sibling. u = undo, n = save, "
          "s = skip image, q = quit.")
    for p in imgs:
        if sum(c["image"] == str(p) for c in cases) >= 2:
            continue
        print(f"\n{p}  ({sum(c['image'] == str(p) for c in cases)} case(s) so far)")
        instr = input("instruction (blank = skip image): ").strip()
        if not instr:
            continue
        et = input(f"expr_type {EXPR_TYPES}: ").strip()
        arm = input("arm (A = non-COCO image, B = COCO/RefEdit image) [A]: ").strip() or "A"
        boxes, state = [], {"key": None}
        fig, ax = plt.subplots()
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title("target box first, then siblings; u undo, n save, s skip, q quit")

        def on_box(e0, e1):
            b = [min(e0.xdata, e1.xdata), min(e0.ydata, e1.ydata),
                 max(e0.xdata, e1.xdata), max(e0.ydata, e1.ydata)]
            boxes.append([round(v, 1) for v in b])
            ax.add_patch(plt.Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False,
                                       ec="lime" if len(boxes) == 1 else "red", lw=2))
            fig.canvas.draw_idle()

        def on_key(ev):
            if ev.key == "u" and boxes:
                boxes.pop()
                ax.patches[-1].remove()
                fig.canvas.draw_idle()
            elif ev.key in ("n", "s", "q"):
                state["key"] = ev.key
                plt.close(fig)

        sel = RectangleSelector(ax, on_box, useblit=True, interactive=False)  # noqa: F841
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()
        if state["key"] == "q":
            break
        if state["key"] != "n" or len(boxes) < 2:
            print("  not saved (needs n, a target box and >= 1 sibling box)")
            continue
        cases.append({"id": f"fresh-{len(cases) + 1:03d}", "image": str(p),
                      "image_wh": list(Image.open(p).size),
                      "instruction": instr, "expr_type": et, "arm": arm,
                      "target": boxes[0], "siblings": boxes[1:]})
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.write_text(json.dumps(cases, indent=1))
        print(f"  saved {cases[-1]['id']} ({len(cases)} total)")
    errs = check_cases(cases)
    print("rules: OK" if not errs else "rules not yet met:\n  " + "\n  ".join(errs))


# ------------------------------------------------------------ self-test ---

def self_test() -> None:
    m = box_mask([0, 0, 50, 100], (100, 200))          # left half, top half
    assert m.shape == (SIZE, SIZE) and m[:256, :256].all() and not m[256:].any() \
        and not m[:, 256:].any()
    assert box_mask([50, 100, 0, 0], (100, 200)).sum() == m.sum()   # reversed corners
    # scorer path: prediction inside target box -> target; inside sibling -> sibling
    from predict_masks import score_masks  # noqa: PLC0415
    T, S = box_mask([0, 0, 40, 40], (100, 100)), box_mask([60, 60, 100, 100], (100, 100))
    assert score_masks(T, T, [S])["cls"] == "target"
    assert score_masks(S, T, [S])["cls"] == "sibling"
    # overlapping boxes: overlap counted for the target only
    S2 = box_mask([20, 0, 60, 40], (100, 100))
    assert score_masks(T, T, [S2])["cls"] == "target"
    base = {"image": "x.jpg", "instruction": "i", "target": [0, 0, 1, 1],
            "siblings": [[2, 2, 3, 3]], "arm": "A"}
    doms = ["photo"] * 30 + ["illustration"] * 15 + ["painting"] * 15
    good = [dict(base, id=f"c{k}", image=f"im{k // 2}.jpg", expr_type=EXPR_TYPES[k % 3],
                 domain=doms[k]) for k in range(60)]
    assert check_cases(good) == [], check_cases(good)
    assert any("per-domain" in e for e in check_cases(
        [dict(g, domain="photo") for g in good]))
    # arm T: 30 attach (10 hold / 12 wear / 8 onbody, domains 10/10/10) + 15 controls
    tt = []
    for k in range(30):
        tt.append(dict(base, id=f"t{k}", arm="T", image=f"t{k}.jpg", expr_type="spatial",
                       domain=DOMAINS[k % 3], edit_kind="attach",
                       attach_kind="hold" if k < 10 else "wear" if k < 22 else "onbody",
                       target=[0, 0, 5, 5]))
    for k in range(15):
        tt.append(dict(tt[k], id=f"u{k}", edit_kind="inbody"))
    assert check_cases(good + tt) == [], check_cases(good + tt)
    assert any("share" in e for e in check_cases(good + [dict(tt[0], image="im0.jpg")]))
    bad_ctrl = dict(tt[30], target=[1, 1, 9, 9])
    assert any("inbody control" in e for e in check_cases(tt[:30] + [bad_ctrl] + tt[31:]))
    bad = [dict(good[0], siblings=[]), dict(good[1], id="c0")]
    errs = check_cases(bad)
    assert any("sibling" in e for e in errs) and any("duplicate" in e for e in errs)
    assert any("max 2 per image" in e for e in check_cases(
        [dict(g, image="same.jpg") for g in good]))
    assert any(">= 10" in e for e in check_cases(good[:12]))
    recs = [{"id": f"c{k}", "arm": "A", "expr_type": EXPR_TYPES[k % 3],
             "cls": "target" if k < 25 else "sibling"} for k in range(30)]
    s = summarise(recs)["A"]
    assert (s["target"], s["n"]) == (25, 30) and band(s["rate"]) == BANDS[1][1]
    s2 = summarise(recs, {"c25": {"cls": "target"}, "c26": {"cls": "target"}})["A"]
    assert s2["target"] == 27 and band(s2["rate"]) == BANDS[2][1]
    assert band(0.70) == BANDS[0][1] and band(0.85) == BANDS[2][1]
    check_size({"id": "x", "image": "i", "image_wh": [10, 20]}, (10, 20))
    check_size({"id": "x", "image": "i"}, (99, 99))
    try:
        check_size({"id": "x", "image": "i", "image_wh": [2000, 1125]}, (2364, 1330))
        raise AssertionError("size mismatch not caught")
    except SystemExit:
        pass
    print("SELF-TEST OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="runs/fresh/cases.json")
    p.add_argument("--annotate", metavar="IMAGE_DIR")
    p.add_argument("--check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--backend", choices=("sa2va",), default="sa2va")
    p.add_argument("--model-path", default="ByteDance/Sa2VA-Qwen2_5-VL-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/fresh/sa2va.jsonl")
    p.add_argument("--out-dir", default="runs/fresh/masks")
    p.add_argument("--save-overlays", default="runs/fresh/overlays")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--edit", action="store_true", help="klein edits through saved masks")
    p.add_argument("--edit-dir", default="runs/fresh/edits")
    p.add_argument("--offload", action="store_true")
    p.add_argument("--report", metavar="JSONL")
    p.add_argument("--overrides", metavar="JSON")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    elif a.annotate:
        annotate(Path(a.annotate), Path(a.cases))
    elif a.check:
        errs = check_cases(json.loads(Path(a.cases).read_text()))
        print("OK" if not errs else "\n".join(errs))
    elif a.run:
        run(a)
    elif a.edit:
        edit(a)
    elif a.report:
        report(a.report, a.cases, a.overrides)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
