"""Occlusion / contrast test on arm T (MODERN_BASE_EVAL.md, "Hand-visibility test";
reading rule FROZEN there before any tag was written).

Question: do crop-box attach failures concentrate where the hand that must hold the
object is hidden or hard to tell apart in the SOURCE image?

    # 1. tagging sheets: source crop around the target, numbered, no edits, no verdicts
    python bench/hand_tags.py --sheet --out-dir runs/fresh/hand_tags
    # 2. fill runs/fresh/hand_tags/hand_tags.json (one entry per case, both flags true/false)
    # 3. score
    python bench/hand_tags.py --score runs/fresh/hand_tags/hand_tags.json

Scope: the 30 hold cases Sa2VA aimed (airforce excluded) (wear / on-body cases have no holding hand).
Flags, judged on the source only:
  occluded      the hand or forearm the edit needs is partly hidden (behind a person,
                an object, the frame edge) or not visible at all
  low_contrast  that hand/forearm is hard to separate from what is behind or beside it
                (same colour as the clothes, or touching another person's hand/arm)
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

CASES = "bench/attach_cases.json"
AIM = "runs/fresh/attach_sa2va.jsonl"
BLIND = "runs/fresh/attach_blind"
OK = {"clean", "off_target", "degraded"}
EXCLUDED = {"T-airforce-flag"}   # instruction error (user, 2026-11-20), excluded before tagging


def hold_cases():
    aim = {json.loads(l)["id"] for l in Path(AIM).read_text().splitlines()
           if l.strip() and json.loads(l)["cls"] == "target"}
    return [c for c in json.loads(Path(CASES).read_text())
            if c["edit_kind"] == "attach" and c.get("attach_kind") == "hold" and c["id"] in aim
            and c["id"] not in EXCLUDED]


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] with margins fixed:
    a = hard & fail, b = hard & ok, c = easy & fail, d = easy & ok."""
    n1, k, n = a + b, a + c, a + b + c + d
    tot = comb(n, k)
    return sum(comb(n1, x) * comb(n - n1, k - x) for x in range(a, min(n1, k) + 1)) / tot


def reading(hard_ok, hard_n, easy_ok, easy_n, p):
    if hard_n < 5 or easy_n < 5:
        return "inconclusive: fewer than 5 cases in a group"
    diff = easy_ok / easy_n - hard_ok / hard_n
    if diff >= 0.30 and p <= 0.05:
        return ("SUPPORTED: failures concentrate on hidden or low-contrast hands; "
                "part-level (amodal) body structure is the method target")
    if diff <= 0.10:
        return "NOT SUPPORTED: hand visibility does not explain the failures"
    return "inconclusive: report the cases"


def verdicts():
    led = {e["panel"]: e for e in json.loads(Path(f"{BLIND}/ledger.json").read_text())}
    out = {}
    for k in json.loads(Path(f"{BLIND}/key.json").read_text()):
        if k["set"] == "fresh" and k["arm"] == "cropbox" and k["edit_kind"] == "attach":
            out[k["id"]] = led[k["panel"]]["verdict"]
    return out


def sheet(out_dir: Path):
    from PIL import Image, ImageDraw  # noqa: PLC0415
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = hold_cases()
    tiles, template = [], {}
    for n, c in enumerate(cs):
        im = Image.open(c["image"]).convert("RGB")
        x0, y0, x1, y1 = c["target"]
        w, h = x1 - x0, y1 - y0
        box = (max(0, int(x0 - .5 * w)), max(0, int(y0 - .2 * h)),
               min(im.width, int(x1 + .5 * w)), min(im.height, int(y1 + .2 * h)))
        cr = im.crop(box)
        ImageDraw.Draw(cr).rectangle((x0 - box[0], y0 - box[1], x1 - box[0], y1 - box[1]),
                                     outline=(0, 255, 0), width=max(2, cr.width // 150))
        cr.thumbnail((520, 520))
        tile = Image.new("RGB", (540, 600), "white")
        tile.paste(cr, (10, 70))
        d = ImageDraw.Draw(tile)
        d.text((10, 6), f"{n + 1:02d}  {c['id']}", fill=(0, 0, 0))
        d.text((10, 26), c["instruction"][:85], fill=(0, 0, 0))
        d.text((10, 44), c["instruction"][85:170], fill=(0, 0, 0))
        tiles.append(tile)
        template[c["id"]] = {"occluded": None, "low_contrast": None, "note": ""}
    for s in range(0, len(tiles), 8):
        page = Image.new("RGB", (540 * 4, 600 * 2), "white")
        for i, t in enumerate(tiles[s:s + 8]):
            page.paste(t, ((i % 4) * 540, (i // 4) * 600))
        page.save(out_dir / f"sheet_{s // 8 + 1}.png")
    p = out_dir / "hand_tags.json"
    if not p.exists():
        p.write_text(json.dumps(template, indent=1))
    print(f"{len(cs)} hold cases -> {out_dir}/sheet_*.png; fill {p} (true/false)")


def score(tags_path: Path):
    tags = json.loads(tags_path.read_text())
    v = verdicts()
    ids = [c["id"] for c in hold_cases()]
    missing = [i for i in ids if tags.get(i, {}).get("occluded") is None
               or tags.get(i, {}).get("low_contrast") is None]
    if missing:
        raise SystemExit(f"untagged: {missing}")
    if any(v.get(i) is None for i in ids):
        raise SystemExit("some crop-box verdicts are missing")
    rows = {"hard": [0, 0], "easy": [0, 0]}
    for i in ids:
        g = "hard" if tags[i]["occluded"] or tags[i]["low_contrast"] else "easy"
        rows[g][1] += 1
        rows[g][0] += v[i] in OK
    (ho, hn), (eo, en) = rows["hard"], rows["easy"]
    p = fisher_one_sided(hn - ho, ho, en - eo, eo)
    print(f"hard (occluded or low-contrast): edit-right {ho}/{hn}")
    print(f"easy:                            edit-right {eo}/{en}")
    print(f"one-sided Fisher p = {p:.3f}")
    print("reading:", reading(ho, hn, eo, en, p))
    for flag in ("occluded", "low_contrast"):           # descriptive only
        a = [v[i] in OK for i in ids if tags[i][flag]]
        b = [v[i] in OK for i in ids if not tags[i][flag]]
        print(f"  {flag:<13} yes {sum(a)}/{len(a)}   no {sum(b)}/{len(b)}")
    return ho, hn, eo, en, p


def self_test():
    assert abs(fisher_one_sided(5, 0, 0, 5) - 1 / 252) < 1e-12
    assert fisher_one_sided(0, 5, 5, 0) == 1.0
    assert reading(2, 10, 15, 20, 0.01).startswith("SUPPORTED")
    assert reading(6, 10, 13, 20, 0.6).startswith("NOT")
    assert reading(4, 10, 14, 20, 0.2).startswith("inconclusive")
    assert reading(1, 3, 20, 28, 0.01).startswith("inconclusive: fewer")
    print("SELF-TEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out-dir", default="runs/fresh/hand_tags")
    ap.add_argument("--score", metavar="TAGS_JSON")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.sheet:
        sheet(Path(a.out_dir))
    elif a.score:
        score(Path(a.score))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
