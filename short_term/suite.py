#!/usr/bin/env python3
"""
Run the measured recipe over N cached MagicBrush items and write a SHEET.

WHY THIS EXISTS. Every number behind "the reference-guided recipe matches the
dataset's own edit" comes from ONE case -- a single balloon addition, chosen
because it was the most-changed item in the cache. n=1. README section 5.6
prescribes the fix ("5-10 cache cases, look at the DISTRIBUTION of d_in/tgt and
sat_in/tgt, not one case's absolute value"); this automates it and writes the
result as a table instead of a wall of terminal output.

    # 20 items, sampled across the usable population (not the easiest 20)
    python short_term/suite.py --cache /data/cache/<hash> --sample 20 \\
        --out runs/st/sheet20

    # the four arms, on a handful of items
    python short_term/suite.py --cache /data/cache/<hash> --top 5 \\
        --arms base,both,protect,selfy --out runs/st/arms

    # score whatever PNGs already exist; no cache, no model, no GPU
    python short_term/suite.py --cache /data/cache/<hash> --sample 20 \\
        --out runs/st/sheet20 --score-only

OUTPUT (all under --out)

    sheet.csv          one row per (item, arm). Opens in Excel.
    sheet.json         the full records, including each item's tgt row
    sheet_summary.txt  the cross-item verdict, per arm
    contact.png        src | tgt(GT) | one column per arm, one row per item
    items/<index>/     src.png, mask.png, tgt.png, meta.json, <arm>.png

THE ARMS

    base     --guide none                      whole-frame regeneration baseline
    both     --guide both (inside -> tgt)      the measured recipe: ORACLE Y
    protect  --guide source --invert-weight    background only
    selfy    --guide both (inside -> src)      experiment 3: inside pull toward
                                               the source, i.e. no WHAT source
                                               at all. Should barely edit.

WHY `both` IS CALLED THE ORACLE. Its inside target is the dataset's own edited
image, which you do not have at inference time. It measures the MECHANISM's
ceiling, not a deployable configuration. Deployable = `both` with a Y the user
supplies (line 1) or propose_y.py manufactures (line 2).

SELECTION MATTERS AND IS RECORDED. `--top` takes the most-changed items, which
is what the single-case study accidentally did and is therefore a biased sample.
`--sample` draws uniformly from the usable population and is the default. Which
one ran is written into the CSV, because "it generalises" means nothing without
knowing what it generalised over.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term import cache_item, edit, score as scoring  # noqa: E402

#: The recipe that was measured on one case. Both terms, constant kappa,
#: protection at the strength where it saturated (0.6), the inside pull at 1.0.
BOTH_ARGS = ["--guide", "both", "--kappa", "0.6", "--kappa-out", "1.0",
             "--kappa-mode", "flat"]
PROTECT_ARGS = ["--guide", "source", "--invert-weight", "--kappa", "1.0",
                "--kappa-mode", "flat"]

#: arm -> extra argv builder. `y` is "tgt" or "src" (a file under the fixture).
ARMS = {
    "base": lambda y: ["--guide", "none"],
    "both": lambda y: ["--guide-image", y] + BOTH_ARGS,
    "protect": lambda y: list(PROTECT_ARGS),
    "selfy": lambda y: ["--guide-image", y] + BOTH_ARGS,
}

COLUMNS = ["item", "arm", "instruction", "mask_frac", "moved",
           "d_in", "d_in_vs_tgt", "d_out_c", "d_out_c_vs_tgt",
           "contrast", "sat_in", "sat_in_vs_tgt", "exp_gain",
           "held", "valid", "tgt_d_in", "tgt_d_out_c", "png"]


# --------------------------------------------------------------- selection --

def pick_items(cache, indices, top, sample, seed, take_all=False):
    """-> (indices, rule). `rule` is recorded in the sheet: a result is only as
    general as the population it was drawn from.

    `take_all` is for a SUBSET cache (subset_cache.py), which is already the
    selection: re-ranking it here would silently drop items whose mask_frac the
    subset's author chose deliberately.
    """
    _meta, items = cache_item.load_index(cache)
    if indices:
        out = []
        for tok in indices.replace(",", " ").split():
            i = int(tok)
            if not 0 <= i < len(items):
                raise SystemExit(f"--indices {i} out of range "
                                 f"(cache has {len(items)} items)")
            out.append(i)
        return out, f"explicit:{indices}"
    if take_all:
        return list(range(len(items))), f"all{len(items)}_of_cache"
    cand = cache_item.rank(items)          # in-band, most-changed first
    if not cand:
        raise SystemExit(
            f"no item has mask_frac in [{cache_item.MASK_MIN}, "
            f"{cache_item.MASK_MAX}]; widen the band in cache_item.rank() or "
            "pass --indices")
    if top:
        return [items.index(r) for r in cand[:top]], f"top{top}_by_moved"
    if not sample:
        raise SystemExit("give --indices, --top or --sample")
    n = min(int(sample), len(cand))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), size=n, replace=False)
    return [items.index(cand[i]) for i in sorted(idx)], \
        f"sample{n}_of{len(cand)}_inband_seed{seed}"


# ---------------------------------------------------- transfer + manifest ---

def recover_items(out: Path) -> list[str]:
    """Item directory names that hold a complete fixture."""
    return sorted(p.name for p in (out / "items").glob("*")
                  if p.is_dir() and (p / "src.png").is_file()
                  and (p / "mask.png").is_file()
                  and (p / "tgt.png").is_file()
                  and (p / "meta.json").is_file())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out: Path, rule: str, cache: str | None) -> Path:
    """A hash per file, so a transfer can be PROVEN rather than assumed.

    WHY. This project has lost time twice to files that did not arrive: a
    partial sync made `sampler.py` absent and Python reported it as
    "No module named 'short_term.sampler'", pointing at the wrong thing; and
    Stage 0 lost three days to stale files read as findings. A 20-item fixture
    is ~60 files moved between machines, which is exactly the situation a
    manifest is for.

    It also pins the INPUT, which is what makes two servers comparable at all:
    src.png, mask.png and tgt.png are fixed bytes, so a run on machine B and a
    run on machine A differ only by the model and the flags -- not by an
    encode/decode round trip that each machine did its own way.
    """
    files: dict[str, dict] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(out)).replace("\\", "/")] = {
                "sha256": _sha256(p), "bytes": p.stat().st_size}
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection": rule,
        "cache": cache,
        "n_items": len(recover_items(out)),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
    }
    mp = out / "manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    return mp


def verify_manifest(out: Path) -> int:
    """Recompute and compare. Missing, truncated and modified all reported."""
    mp = out / "manifest.json"
    if not mp.is_file():
        print(f"no manifest at {mp}; nothing to verify")
        return 2
    man = json.loads(mp.read_text())
    missing, changed, ok = [], [], 0
    for rel, rec in man["files"].items():
        p = out / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if p.stat().st_size != rec["bytes"]:
            changed.append(f"{rel} (size {p.stat().st_size} != "
                           f"{rec['bytes']})")
            continue
        if _sha256(p) != rec["sha256"]:
            changed.append(f"{rel} (sha256 differs)")
            continue
        ok += 1
    print(f"manifest: {man['n_items']} item(s), {man['n_files']} file(s), "
          f"{man['total_bytes'] / 1e6:.1f} MB, selection {man['selection']}")
    print(f"  verified {ok}/{man['n_files']}")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for m in missing[:20]:
            print(f"    {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    if changed:
        print(f"  CHANGED ({len(changed)}):")
        for c in changed[:20]:
            print(f"    {c}")
    if not missing and not changed:
        print("  OK: every file matches the manifest. A partial transfer would "
              "show up here -- it did not.")
        return 0
    print("  FAILED: do not run a sheet on this directory until it matches.")
    return 1


# ----------------------------------------------------------------- scoring --

def score_item(item_dir: Path, out_png: Path) -> dict | None:
    """Score one output against the fixture's own target."""
    from short_term.backends import load_image, load_mask

    src = load_image(str(item_dir / "src.png"))
    mask = load_mask(str(item_dir / "mask.png"), src.shape[0])
    tgt = load_image(str(item_dir / "tgt.png"))
    out = load_image(str(out_png))
    if out.shape != src.shape:
        print(f"  !! {out_png.name}: shape {out.shape} != src {src.shape}; "
              "skipped rather than resized")
        return None
    s = scoring.score(src, out, mask, expect=tgt)      # --expect tgt
    s["_tgt"] = scoring.score(src, tgt, mask, expect=tgt)
    s["_path"] = str(out_png)
    return s


def _multiples(s: dict) -> tuple[float, float, float]:
    """(d_in/tgt, d_out_c/tgt, sat_in/tgt). The first is the one that decides
    whether the recipe generalises: 1.0 = the dataset's own amount of change."""
    st = s["_tgt"]
    din = s["d_inside"] / max(st["d_inside"], 1e-9)
    if st["d_out_content"] > 1e-6:
        dout = s["d_out_content"] / st["d_out_content"]
    else:
        dout = 0.0 if s["d_out_content"] <= 1e-6 else float("inf")
    sat = s["sat_in"] / max(st["sat_in"], 1e-9)
    return din, dout, sat


def _num(x) -> str:
    if x is None or not np.isfinite(x):
        return "inf" if x == float("inf") else "nan"
    return f"{x:.4f}"


# ------------------------------------------------------------ the contact ---

def contact_sheet(rows: list[dict], arms: list[str], out_png: Path,
                  cell: int = 192) -> None:
    """src | tgt(GT) | one column per arm; one row per item.

    README 6.1 and STAGE0_FINDINGS SS1 both say the images are not optional --
    three automated verdicts in Stage 0 were false and one look would have
    caught them. This is that look, for every item at once.
    """
    from PIL import Image, ImageDraw

    by_item: dict[str, dict[str, str]] = {}
    for r in rows:
        d = by_item.setdefault(r["item"], {})
        d[r["arm"]] = r["png"]
        # src and tgt live beside the arm PNGs, not in `rows` -- the first
        # version looked them up in `rows` and drew two empty columns with
        # placeholder diagonals for every single item.
        parent = Path(r["png"]).parent
        d["src"] = str(parent / "src.png")
        d["tgt(GT)"] = str(parent / "tgt.png")
    if not by_item:
        return
    cols = ["src", "tgt(GT)"] + arms
    head = 26
    label = 34
    W = len(cols) * (cell + 4) + 4
    H = head + len(by_item) * (cell + label + 4) + 4
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(cols):
        draw.text((4 + c * (cell + 4), 6), name, fill=(230, 230, 230))

    for i, (item, paths) in enumerate(sorted(by_item.items())):
        y0 = head + i * (cell + label + 4)
        for c, name in enumerate(cols):
            x0 = 4 + c * (cell + 4)
            p = paths.get(name)
            if not p or not Path(p).is_file():
                # Draw a small box rather than a full-cell diagonal: a missing
                # column should look like a missing column, not like artwork.
                draw.rectangle([x0 + cell // 4, y0 + cell // 4,
                                x0 + 3 * cell // 4, y0 + 3 * cell // 4],
                               outline=(90, 60, 60))
                draw.text((x0 + cell // 2 - 14, y0 + cell // 2 - 5), "missing",
                          fill=(150, 100, 100))
                continue
            im = Image.open(p)
            with im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell), Image.LANCZOS)
                canvas.paste(im, (x0 + (cell - im.width) // 2,
                                  y0 + (cell - im.height) // 2))
        r0 = next((r for r in rows if r["item"] == item), None)
        txt = f"{item}"
        if r0:
            txt += f"  |  {str(r0['instruction'])[:70]}"
            txt += (f"  |  d_in/tgt {r0['d_in_vs_tgt']:.2f}"
                    f"  sat/tgt {r0['sat_in_vs_tgt']:.2f}"
                    f"  exp {r0['exp_gain']:+.2f}")
        draw.text((6, y0 + cell + 4), txt, fill=(200, 205, 210))
        dy = y0 + cell + 18
        for r in rows:
            if r["item"] == item and r["arm"] != "base":
                draw.text((6, dy), f"  {r['arm']}: held={r['held']} "
                                   f"d_out_c/tgt={r['d_out_c_vs_tgt']}",
                          fill=(150, 155, 160))
                dy += 12
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


# -------------------------------------------------------------- the sheet ---

def write_sheet(rows: list[dict], arms: list[str], rule: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})
    (out / "sheet.json").write_text(json.dumps(
        {"selection": rule, "arms": arms, "n_rows": len(rows),
         "rows": rows}, indent=1))

    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add(f"SHEET  --  {len(rows)} row(s), selection: {rule}")
    add("=" * 78)
    items = sorted({r["item"] for r in rows})
    add(f"  items: {len(items)}   arms: {', '.join(arms)}")
    add("")
    add(f"  {'arm':<9}{'n':>4}{'d_in/tgt':>22}{'sat_in/tgt':>22}"
        f"{'exp_gain':>22}{'held':>8}")
    add(f"  {'':<9}{'':>4}{'median  range':>22}{'median  range':>22}"
        f"{'median  range':>22}")
    for arm in ["both", "protect", "selfy", "base"]:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        din = np.array([r["d_in_vs_tgt"] for r in rs], float)
        sat = np.array([r["sat_in_vs_tgt"] for r in rs], float)
        exp = np.array([r["exp_gain"] for r in rs], float)
        held = sum(bool(r["held"]) for r in rs)

        def cell(a):
            return (f"{np.median(a):+.2f} [{a.min():+.2f},{a.max():+.2f}]"
                    if arm == "base" else
                    f"{np.median(a):.2f} [{a.min():.2f},{a.max():.2f}]")
        add(f"  {arm:<9}{len(rs):>4}{cell(din):>22}{cell(sat):>22}"
            f"{cell(exp):>22}{str(held) + '/' + str(len(rs)):>8}")
    add("")

    both = [r for r in rows if r["arm"] == "both"]
    if both:
        din = np.array([r["d_in_vs_tgt"] for r in both], float)
        exp = np.array([r["exp_gain"] for r in both], float)
        held = sum(bool(r["held"]) for r in both)
        add("  VERDICT for the reference-guided recipe (`both`, ORACLE Y):")
        add(f"    d_in/tgt  median {np.median(din):.2f}  "
            f"range [{din.min():.2f}, {din.max():.2f}]")
        add(f"    exp_gain  median {np.median(exp):+.3f}  "
            f"range [{exp.min():+.3f}, {exp.max():+.3f}]")
        add(f"    background_held  {held}/{len(both)}")
        frac = float(np.mean(np.abs(din - 1.0) <= 0.5))
        add(f"    within +-50% of the truth's change: {frac * 100:.0f}% of "
            "items")
        if len(both) < 3:
            add("    !! n < 3: a pilot, not evidence that it generalises.")
        if np.median(din) > 2.0 or np.median(exp) <= 0:
            add("    READING: does NOT reproduce. Do not move to arbitrary "
                "photographs yet.")
        elif frac < 0.6:
            add("    READING: works on some items and not others. That is a "
                "different finding from 'it works' -- look at which items "
                "fail (edit type, mask size) before believing the median.")
        else:
            add("    READING: consistent with the single-case result across "
                "these items. Check contact.png before believing the table.")
        add("")
    add("  NOTE: `both` uses the DATASET'S OWN target as its inside Y. That is")
    add("  the mechanism's ceiling, not a deployable configuration. Deployable")
    add("  = line 1 (user supplies a reference crop) or line 2 (propose_y.py).")
    add(f"  contact sheet: {out / 'contact.png'}")

    (out / "sheet_summary.txt").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------- self-test ---

def _self_test() -> int:
    """CPU, no cache, no model, no GPU. Exercises the WHOLE sheet path --
    selection, scoring, CSV, JSON, summary, contact sheet -- on fabricated
    fixtures, leaving only the model call untested (which is labelled as such
    everywhere else too)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cache = tmp / "cache" / "cafe"
        cache.mkdir(parents=True)
        items = []
        for i, (mf, mv) in enumerate([(0.01, 0.9), (0.10, 0.5), (0.05, 0.8),
                                      (0.50, 0.9), (0.20, 0.1)]):
            np.savez_compressed(cache / f"{i:05d}.npz",
                                z_src=np.zeros((1, 2, 2), np.float32),
                                z_tgt=np.ones((1, 2, 2), np.float32),
                                mask=np.zeros((1, 2, 2), np.float32))
            items.append({"id": f"c{i}", "stem": f"{i:05d}", "instruction": "x",
                          "emb": None, "mask_frac": mf, "moved": mv})
        (cache / "meta.json").write_text(json.dumps({
            "config": {"size": 16, "latent_size": 2, "dtype": "float32",
                       "max_items": None, "context_length": 333},
            "n": len(items), "items": items}))

        # selection: --top ranks by moved INSIDE the mask_frac band, so item 0
        # (mask 0.01, below the 0.02 floor) and item 3 (mask 0.50, above the
        # 0.30 ceiling) are BOTH excluded. That leaves 1, 2, 4 with moved
        # 0.5, 0.8, 0.1 -> ranked [2, 1, 4]. (The first version of this test
        # expected [0, 2], forgetting the lower bound.)
        idx, rule = pick_items(cache, None, 2, None, 0)
        assert idx == [2, 1], (idx, rule)
        assert rule == "top2_by_moved", rule
        assert pick_items(cache, None, 99, None, 0)[0] == [2, 1, 4], \
            "the band has only 3 items; --top must cap at what exists"
        s1, _ = pick_items(cache, None, None, 4, 1)
        s2, rule2 = pick_items(cache, None, None, 4, 1)
        assert s1 == s2 and len(set(s1)) == 3, s1
        assert sorted(s1) == [1, 2, 4] and 0 not in s1 and 3 not in s1, s1
        # the rule records what ACTUALLY ran (3, capped from the requested 4),
        # not what was asked for -- a sheet must say what it sampled
        assert rule2 == "sample3_of3_inband_seed1", rule2
        assert pick_items(cache, "4 1", None, None, 0) == ([4, 1], "explicit:4 1")
        # --all takes the cache as given, which is what a subset cache needs:
        # re-ranking it would silently drop a deliberately-chosen item
        assert pick_items(cache, None, None, None, 0, True) == (
            [0, 1, 2, 3, 4], "all5_of_cache")
        for bad, why in (("99", "out of range"),):
            try:
                pick_items(cache, bad, None, None, 0)
            except SystemExit as exc:
                assert why in str(exc)
            else:
                raise AssertionError(f"{bad} must be refused")

        # fabricate fixtures + outputs, then run the real scoring path
        from short_term.backends import save_image
        out = tmp / "sheet"
        for i in idx:
            d = out / "items" / f"{i:05d}"
            d.mkdir(parents=True)
            src = np.full((64, 64, 3), 0.5, np.float32)
            m = np.zeros((64, 64), np.float32)
            m[16:48, 16:48] = 1.0
            tgt = src.copy()
            tgt[16:48, 16:48] = 0.9            # the truth's change: +0.4
            save_image(src, d / "src.png")
            save_image(m, d / "mask.png")
            save_image(tgt, d / "tgt.png")
            (d / "meta.json").write_text(json.dumps(
                {"instruction": "make it brighter", "mask_frac": 0.25,
                 "moved": 0.4, "id": f"c{i}"}))
            for arm, delta in (("base", 0.10), ("both", 0.40),
                               ("protect", 0.02)):
                o = src.copy()
                o[16:48, 16:48] += delta
                o[2:10, 2:10] += 0.004         # a little background drift
                save_image(o, d / f"{arm}.png")

        rows = []
        for i in idx:
            d = out / "items" / f"{i:05d}"
            for arm in ("base", "both", "protect"):
                s = score_item(d, d / f"{arm}.png")
                assert s is not None
                din, dout, sat = _multiples(s)
                rows.append({
                    "item": f"{i:05d}", "arm": arm,
                    "instruction": s.get("instruction", "make it brighter"),
                    "mask_frac": 0.25, "moved": 0.4,
                    "d_in": round(s["d_inside"], 4), "d_in_vs_tgt": din,
                    "d_out_c": round(s["d_out_content"], 4),
                    "d_out_c_vs_tgt": dout, "contrast": s["contrast"],
                    "sat_in": s["sat_in"], "sat_in_vs_tgt": sat,
                    "exp_gain": s["expect_gain"],
                    "held": bool(s["verdict"]["background_held"]),
                    "valid": bool(s["verdict"]["valid_wholeframe"]),
                    "tgt_d_in": s["_tgt"]["d_inside"],
                    "tgt_d_out_c": s["_tgt"]["d_out_content"],
                    "png": s["_path"]})
        # the `both` arm was constructed to change exactly as much as the
        # target (delta 0.40 both sides), so its multiple must be 1.0 and the
        # base arm's must be 0.25 -- this is the check that the multiple is
        # computed against the item's OWN tgt, not a shared constant
        b = [r for r in rows if r["arm"] == "both"]
        n = [r for r in rows if r["arm"] == "base"]
        assert b and all(abs(r["d_in_vs_tgt"] - 1.0) < 0.05 for r in b), \
            [r["d_in_vs_tgt"] for r in b]
        assert all(abs(r["d_in_vs_tgt"] - 0.25) < 0.05 for r in n), \
            [r["d_in_vs_tgt"] for r in n]

        write_sheet(rows, ["base", "both", "protect"], "top2_by_moved", out)
        assert (out / "sheet.csv").is_file()
        with (out / "sheet.csv").open(encoding="utf-8") as fh:
            got = list(csv.DictReader(fh))
        assert len(got) == len(rows) and list(got[0].keys()) == COLUMNS
        assert (out / "sheet.json").is_file()
        summary = (out / "sheet_summary.txt").read_text()
        assert "reference-guided recipe" in summary and "median" in summary
        assert "ORACLE Y" in summary, "the sheet must say the Y is the truth"

        contact_sheet(rows, ["base", "both", "protect"], out / "contact.png")
        from PIL import Image
        # `with`: on Windows an open handle makes TemporaryDirectory cleanup
        # fail with WinError 32, which looks like a test failure and is not one.
        with Image.open(out / "contact.png") as im:
            assert im.width > 0 and im.height > 0
            # 2 items + header + label rows
            assert im.height > 2 * 192, im.size

        # transfer integrity: the manifest must verify, must DETECT a missing
        # file and a modified one, and recover_items must ignore an incomplete
        # fixture (a directory whose tgt.png did not arrive)
        mp = write_manifest(out, "top2_by_moved", "/data/cache/fake")
        man = json.loads(mp.read_text())
        assert man["n_items"] == 2, man["n_items"]
        assert "items/00002/src.png" in man["files"], list(man["files"])[:3]
        assert verify_manifest(out) == 0
        (out / "items" / "00002" / "mask.png").unlink()
        assert verify_manifest(out) == 1, "a missing file must fail the check"
        assert recover_items(out) == ["00001"], "an incomplete fixture must be " \
            "ignored, not run"
        save_image(m, out / "items" / "00002" / "mask.png")   # restore, same bytes
        assert verify_manifest(out) == 0
        (out / "items" / "00002" / "mask.png").write_bytes(b"not the same")
        bad = verify_manifest(out)
        assert bad == 1
        (out / "manifest.json").unlink()
        assert verify_manifest(out) == 2, "no manifest is its own exit code"

    print("suite self-test OK")
    print("  selection: --top ranks inside the mask_frac band; --sample is")
    print("  seeded, without replacement, and the rule is recorded in the sheet")
    print("  d_in_vs_tgt is computed per item against that item's own tgt")
    print("  sheet.csv / sheet.json / sheet_summary.txt / contact.png all write")
    return 0


# ------------------------------------------------------------------ main ---

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None,
                    help="the cache HASH directory; not needed with "
                         "--score-only, which rebuilds a sheet from fixtures "
                         "already on disk")
    ap.add_argument("--indices", default=None,
                    help='explicit item indices, e.g. "3 17 44"')
    ap.add_argument("--top", type=int, default=None,
                    help="the N most-changed items in the band (BIASED toward "
                         "large edits; prefer --sample)")
    ap.add_argument("--sample", type=int, default=None,
                    help="N items drawn uniformly from the usable population")
    ap.add_argument("--all", dest="take_all", action="store_true",
                    help="use every item in the cache as given. This is what a "
                         "SUBSET cache (subset_cache.py) wants: the subset IS "
                         "the selection, so re-ranking it would drop items.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="base,both",
                    help="comma list from: base, both, protect, selfy")
    ap.add_argument("--out", default="runs/st/sheet")
    ap.add_argument("--model", default="BleachNick/SD3_UltraEdit_w_mask")
    ap.add_argument("--vae-model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", default="512")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--score-only", action="store_true",
                    help="score existing PNGs; no cache read, no model")
    ap.add_argument("--export-only", action="store_true",
                    help="export the selected fixtures + manifest.json, then "
                         "STOP. This is the transfer step: the fixtures are "
                         "self-contained, so the machine that runs them needs "
                         "no cache, only short_term/ and the model.")
    ap.add_argument("--verify", action="store_true",
                    help="check manifest.json against the files present and "
                         "exit. Run this on the receiving machine BEFORE "
                         "trusting the directory.")
    ap.add_argument("--no-contact", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.verify:
        return verify_manifest(out)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from "
                             f"{', '.join(ARMS)}")

    out.mkdir(parents=True, exist_ok=True)
    if args.score_only or (not args.cache and recover_items(out)):
        # Recover the item list from the fixture directory instead of the
        # cache, so a directory that has been MOVED here can still be run or
        # rescored. This is what makes "export there, run here" work without
        # the cache.
        idx = recover_items(out)
        rule = f"recovered_from_items_dir(n={len(idx)})"
        if not idx:
            raise SystemExit(
                f"no complete fixtures under {out / 'items'} (each needs "
                "src.png, mask.png, tgt.png, meta.json) and no --cache given")
    else:
        if not args.cache:
            raise SystemExit(
                "--cache is required unless --score-only, or unless "
                f"{out / 'items'} already holds complete fixtures")
        idx, rule = pick_items(args.cache, args.indices, args.top,
                               args.sample, args.seed, args.take_all)
    print(f"selection: {rule}")
    print(f"{len(idx)} item(s), arms: {', '.join(arms)}, out: {out}")

    if args.export_only:
        for i in idx:
            name = f"{i:05d}" if isinstance(i, int) else str(i)
            d = out / "items" / name
            if (d / "src.png").is_file() and (d / "tgt.png").is_file():
                continue
            cache_item.export(argparse.Namespace(
                cache=args.cache, index=int(i), id=None, out=str(d),
                no_decode=False, model=args.vae_model, device=args.device,
                quiet=True))
        mp = write_manifest(out, rule, args.cache)
        man = json.loads(mp.read_text())
        n = man["n_items"]
        print(f"\nexported {n} fixture(s), {man['n_files']} file(s), "
              f"{man['total_bytes'] / 1e6:.1f} MB")
        print(f"manifest: {mp}")
        if n < len(idx):
            print(f"  !! only {n} of {len(idx)} items are complete; the "
                  "export did not finish. Re-run before moving the directory.")
        print("\nTRANSFER: move this whole directory, then on the OTHER machine")
        print(f"  python short_term/suite.py --out {out} --verify")
        print(f"  python short_term/suite.py --out {out} --arms base,both,"
              f"protect,selfy")
        print("The receiving machine needs short_term/ and the model. It does "
              "NOT need the cache: edit.py takes PNGs and re-encodes them with "
              "its own VAE, and tgt.png is the fixed ground truth.")
        return 0

    editor = None
    rows: list[dict] = []
    print(f"\n{'item':<8}{'arm':<9}{'d_in':>8}{'d_in/tgt':>9}{'d_out_c':>9}"
          f"{'out/tgt':>9}{'sat/tgt':>9}{'exp_gain':>10}{'held':>6}")
    for i in idx:
        # `idx` is integers from the cache path, but directory NAMES (strings)
        # when rescoring offline -- the first version formatted both with :05d
        # and died on the string case.
        name = f"{i:05d}" if isinstance(i, int) else str(i)
        item_dir = out / "items" / name
        if not args.score_only:
            cache_item.export(argparse.Namespace(
                cache=args.cache, index=i, id=None, out=str(item_dir),
                no_decode=False, model=args.vae_model, device=args.device,
                quiet=True))
        meta_p = item_dir / "meta.json"
        if not meta_p.is_file():
            print(f"{i:<8}(no fixture at {item_dir})")
            continue
        meta = json.loads(meta_p.read_text())
        instr = meta.get("instruction", "edit the masked region")

        for arm in arms:
            png = item_dir / f"{arm}.png"
            if not args.score_only:
                if editor is None:
                    editor = edit.SD3IP2PEditor(
                        args.model, device=args.device, steps=args.steps)
                y = {"both": item_dir / "tgt.png", "selfy": item_dir / "src.png"}
                argv_ = ["--image", str(item_dir / "src.png"),
                         "--mask", str(item_dir / "mask.png"),
                         "--instruction", instr, "--size", str(args.size),
                         "--steps", str(args.steps), "--seed", str(args.seed),
                         "--device", args.device, "--model", args.model,
                         "--out", str(png)]
                argv_ += ARMS[arm](str(y.get(arm, item_dir / "tgt.png")))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = edit.main(argv_, editor=editor)
                for line in buf.getvalue().splitlines():
                    if "WARNING" in line or "step(s)" in line:
                        print(f"      [{i:05d}/{arm}] {line.strip()}")
                if rc != 0:
                    print(f"{i:<8}{arm:<9}edit.py returned {rc}; skipped")
                    continue
            if not png.is_file():
                print(f"{i:<8}{arm:<9}missing {png.name}")
                continue
            s = score_item(item_dir, png)
            if s is None:
                continue
            din, dout, sat = _multiples(s)
            rows.append({
                "item": f"{i:05d}" if isinstance(i, int) else str(i),
                "arm": arm, "instruction": instr,
                "mask_frac": meta.get("mask_frac"),
                "moved": meta.get("moved"),
                "d_in": round(s["d_inside"], 4), "d_in_vs_tgt": round(din, 4),
                "d_out_c": round(s["d_out_content"], 4),
                "d_out_c_vs_tgt": (dout if np.isfinite(dout)
                                   else float("inf")),
                "contrast": _num(s["contrast"]),
                "sat_in": round(s["sat_in"], 4),
                "sat_in_vs_tgt": round(sat, 4),
                "exp_gain": round(s["expect_gain"], 4),
                "held": bool(s["verdict"]["background_held"]),
                "valid": bool(s["verdict"]["valid_wholeframe"]),
                "tgt_d_in": round(s["_tgt"]["d_inside"], 4),
                "tgt_d_out_c": round(s["_tgt"]["d_out_content"], 4),
                "png": s["_path"]})
            print(f"{str(i):<8}{arm:<9}{s['d_inside']:>8.3f}{din:>9.2f}"
                  f"{s['d_out_content']:>9.4f}{_num(dout):>9}"
                  f"{sat:>9.2f}{s['expect_gain']:>+10.3f}"
                  f"{'PASS' if s['verdict']['background_held'] else 'FAIL':>6}")

    if not rows:
        print("\nno rows produced; nothing to write")
        return 2
    write_sheet(rows, arms, rule, out)
    write_manifest(out, rule, args.cache)
    if not args.no_contact:
        try:
            contact_sheet(rows, arms, out / "contact.png")
        except Exception as exc:  # noqa: BLE001
            print(f"[suite] contact sheet failed ({exc}); the CSV and summary "
                  "are still written")
    print()
    print((out / "sheet_summary.txt").read_text())
    print(f"wrote {out}/sheet.csv, sheet.json, sheet_summary.txt"
          + ("" if args.no_contact else ", contact.png"))
    print("LOOK AT contact.png. Pixel distance is a direction, not a verdict "
          "(README 6.1).")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
