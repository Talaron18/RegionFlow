"""Readable heat-map review panels for the klein grounding probe.

`bench/klein_grounding_probe.py` saved, per case and prompt kind (full / single /
swap), a 32x32 map of attention density from the instruction tokens to the
reference-image tokens (`{id}_{kind}.npy`) and an overlay PNG that divides the map
by its max.  One or two outlier tokens make that overlay nearly black.  This script
re-renders the saved maps on a scale that is readable and puts everything for a
case on one row:

    recovered source | plain klein edit | attention FULL | SINGLE | SWAP

The source images are not on disk.  They are recovered from the probe's `full` PNG,
which was written as `src * (0.35 + 0.65 * a)` (a = map/max, quantised to uint8,
bilinear-resized), by dividing that factor back out.  The recovery is LOSSY: about
+-3 grey levels (1.4 from the uint8 rounding divided by the 0.35 floor of the
factor, plus the probe's final truncation), more in dark regions.  Pixels that are
exactly pure green (0,255,0) / pure red (255,0,0) are the probe's outlines; they are
kept as the target / sibling masks, filled from neighbours in the source, and redrawn
on every tile.

Heat scale: map * 1024 is a multiple of the uniform density (1/1024 per token).  One
scale per case, 0 .. 99th percentile of the three maps pooled; anything above
saturates.  Numbers in captions are copied from the jsonl, never recomputed.

Outputs under --out: {id}.png per case, index.html, sheet_{outcome}_{points|misses}_{n}.png.

    python bench/grounding_heatmaps.py --self-test
    python bench/grounding_heatmaps.py            # defaults = paths in the repo
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

KINDS = ("full", "single", "swap")
TILE = 320
GAP = 4
HEAD_H = 26
CAP_H = 104
GREEN = (0, 255, 0)
RED = (255, 0, 0)
# (position, rgb) stops for the heat colour map; interpolated into a 256-entry LUT.
_STOPS = [(0.0, (40, 0, 130)), (0.25, (0, 90, 255)), (0.5, (0, 220, 200)),
          (0.75, (255, 235, 0)), (1.0, (255, 0, 0))]
LUT = np.stack([np.interp(np.linspace(0, 1, 256), [p for p, _ in _STOPS],
                          [c[i] for _, c in _STOPS]) for i in range(3)], axis=1)


# ------------------------------------------------------------------ io ---

def _quant_a(amap, size):
    """`a` exactly as klein_grounding_probe._save_overlay builds it. size = (w, h)."""
    a = amap / (amap.max() + 1e-12)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
        size, Image.BILINEAR)).astype(np.float64) / 255.0


def recover_source(png_rgb, amap):
    """-> (src float64 HxWx3 (outline pixels filled), target_mask, sibling_mask)."""
    px = np.asarray(png_rgb.convert("RGB")).astype(np.int32)
    h, w = px.shape[:2]
    tmask = (px == GREEN).all(-1)
    smask = (px == RED).all(-1)
    a = _quant_a(amap, (w, h))
    src = (px + 0.5) / (0.35 + 0.65 * a[..., None])
    known = ~(tmask | smask)
    src = _fill(src, known)
    return src.clip(0, 255), tmask, smask


def _fill(img, known, iters=6):
    img = img.copy()
    known = known.copy()
    for _ in range(iters):
        if known.all():
            break
        k = known.astype(np.float64)[..., None]
        num = np.zeros_like(img)
        den = np.zeros(img.shape[:2] + (1,))
        v = img * k
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                num += np.roll(np.roll(v, dy, 0), dx, 1)
                den += np.roll(np.roll(k, dy, 0), dx, 1)
        new = (den[..., 0] > 0) & ~known
        img[new] = num[new] / den[new]
        known |= new
    return img


def _read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_audits(specs):
    """specs: ['NAME:LEDGER:KEY', ...] -> {id: [(name, verdict), ...]}. Missing files skipped."""
    audits = {}
    for spec in specs:
        name, _, rest = spec.partition(":")
        parts = re.split(r"(?<=\.json):", rest, maxsplit=1)
        if len(parts) != 2:
            print(f"  audit spec not NAME:LEDGER.json:KEY.json, skipped: {spec}")
            continue
        lp, kp = Path(parts[0]), Path(parts[1])
        if not (lp.exists() and kp.exists()):
            print(f"  audit '{name}' missing files, no labels: {lp} / {kp}")
            continue
        with open(lp, encoding="utf-8") as fh:
            ledger = {e["panel"]: e.get("verdict") for e in json.load(fh)}
        with open(kp, encoding="utf-8") as fh:
            key = json.load(fh)
        for e in key:
            v = ledger.get(e["panel"])
            if v is not None:
                audits.setdefault(e["id"], []).append((name, v))
    return audits


# ---------------------------------------------------------------- render ---

def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text, font, width, max_lines):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "...", font=font) > width:
            last = last[:-1]
        lines[-1] = last + "..."
    return lines


def _fit(img, box=TILE):
    """Resize a PIL image to fit in box x box, centred on a dark tile."""
    w, h = img.size
    s = box / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    tile = Image.new("RGB", (box, box), (24, 24, 24))
    tile.paste(img.resize((nw, nh), Image.LANCZOS), ((box - nw) // 2, (box - nh) // 2))
    return tile, (nw, nh, (box - nw) // 2, (box - nh) // 2)


def _mask_to_tile(mask, geom):
    nw, nh, ox, oy = geom
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    full = np.zeros((TILE, TILE), bool)
    full[oy:oy + nh, ox:ox + nw] = np.asarray(m) > 25
    return full


def _paint_outlines(tile, tm, sm):
    arr = np.asarray(tile).copy()
    arr[sm] = RED
    arr[tm] = GREEN
    return Image.fromarray(arr)


def heat_scale(maps):
    """Pooled 99th percentile of the three maps, as multiples of uniform."""
    pooled = np.concatenate([np.asarray(m, np.float64).ravel() * 1024.0 for m in maps])
    return max(float(np.percentile(pooled, 99)), 1e-6)


def render_heat(gray, amap, scale):
    """gray: HxW float 0..255 at source size. -> (RGB uint8 HxWx3, alpha HxW float)."""
    h, w = gray.shape
    up = np.asarray(Image.fromarray((np.asarray(amap, np.float32) * 1024.0)).resize(
        (w, h), Image.BILINEAR)).astype(np.float64)
    v = (up / scale).clip(0, 1)
    color = LUT[(v * 255).round().astype(int)]
    alpha = (v * 1.15).clip(0, 0.85)
    base = np.repeat(gray[..., None], 3, -1) * 0.85
    out = base * (1 - alpha[..., None]) + color * alpha[..., None]
    return out.clip(0, 255).astype(np.uint8), alpha


def _caption(draw, x, y, lines, font, fill=(235, 235, 235)):
    for ln in lines:
        draw.text((x, y), ln, font=font, fill=fill)
        y += 14


def build_panel(cid, src, tmask, smask, maps, recs, edit_img, outcome, audit, scale):
    """Return the one-row PIL panel for a case. maps/recs: dict kind -> map / jsonl rec."""
    h, w = src.shape[:2]
    src_img = Image.fromarray(src.round().astype(np.uint8))
    gray = np.asarray(src_img.convert("L")).astype(np.float64)
    _, geom = _fit(src_img)
    tm, sm = _mask_to_tile(tmask, geom), _mask_to_tile(smask, geom)

    tiles = [(_paint_outlines(_fit(src_img)[0], tm, sm), "source (recovered)", None, None)]
    if edit_img is not None:
        e = edit_img.convert("RGB").resize((w, h), Image.LANCZOS)
        tiles.append((_paint_outlines(_fit(e)[0], tm, sm), "plain klein edit", None, None))
    else:
        tiles.append((Image.new("RGB", (TILE, TILE), (24, 24, 24)), "plain klein edit",
                      None, ["(no edit image)"]))
    for k in KINDS:
        if k in maps:
            rgb, _ = render_heat(gray, maps[k], scale)
            t = _paint_outlines(_fit(Image.fromarray(rgb))[0], tm, sm)
        else:
            t = Image.new("RGB", (TILE, TILE), (24, 24, 24))
        tiles.append((t, k.upper(), k, None))

    W = 5 * TILE + 4 * GAP
    panel = Image.new("RGB", (W, HEAD_H + TILE + CAP_H), (12, 12, 12))
    d = ImageDraw.Draw(panel)
    f, fb = _font(12), _font(13)
    full = recs.get("full", {})
    head = f"{cid}   plain outcome: {outcome}   n_instances: {full.get('n_instances', '?')}"
    if audit:
        head += "   audit: " + ", ".join(f"{n}: {v}" for n, v in audit)
    d.text((6, 5), head, font=fb, fill=(255, 255, 255))
    for i, (tile, label, kind, note) in enumerate(tiles):
        x = i * (TILE + GAP)
        panel.paste(tile, (x, HEAD_H))
        y = HEAD_H + TILE + 3
        d.text((x + 3, y), label, font=fb, fill=(255, 255, 160))
        y += 16
        if kind in KINDS:
            r = recs.get(kind)
            if r is None:
                _caption(d, x + 3, y, ["(no jsonl record)"], f)
                continue
            lines = _wrap(d, r.get("instruction", ""), f, TILE - 6, 3)
            lines.append("points at target: " + {True: "yes", False: "no"}.get(
                r.get("pointing"), "n/a"))
            tp = r.get("target_pref")
            lines.append("target_pref " + (f"{tp:.2f}" if isinstance(tp, (int, float)) else "n/a"))
            lines.append(f"red = {scale:.1f}x uniform")
            _caption(d, x + 3, y, lines, f)
        elif note:
            _caption(d, x + 3, y, note, f)
        elif label == "source (recovered)":
            _caption(d, x + 3, y, ["green = target", "red = siblings"], f)
    return panel


# ------------------------------------------------------------------ main ---

def _group_name(rec):
    return "points" if rec and rec.get("pointing") is True else "misses"


def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    maps_dir, edits_dir = Path(a.maps), Path(a.edits)
    grounding = {}
    for r in _read_jsonl(a.grounding):
        grounding[(r["id"], r["prompt_kind"])] = r
    outcomes = {}
    if Path(a.outcomes).exists():
        for r in _read_jsonl(a.outcomes):
            outcomes.setdefault(r["id"], r.get("outcome", "unknown"))
    else:
        print(f"  outcomes file missing, all 'unknown': {a.outcomes}")
    audits = load_audits(a.audit)

    ids = sorted(p.name[:-len("_full.npy")] for p in maps_dir.glob("*_full.npy"))
    if a.ids:
        want = set(a.ids)
        ids = [i for i in ids if i in want]
    skipped, cases = [], []
    for cid in ids:
        npys = {k: maps_dir / f"{cid}_{k}.npy" for k in KINDS}
        missing = [k for k, p in npys.items() if not p.exists()]
        pngp = maps_dir / f"{cid}_full.png"
        if missing:
            skipped.append((cid, f"missing maps: {missing}"))
            continue
        if not pngp.exists():
            skipped.append((cid, "missing full overlay png"))
            continue
        maps = {k: np.load(p).astype(np.float64) for k, p in npys.items()}
        with Image.open(pngp) as im:
            src, tm, sm = recover_source(im, maps["full"])
        recs = {k: grounding.get((cid, k)) for k in KINDS}
        recs = {k: v for k, v in recs.items() if v is not None}
        if "full" not in recs:
            skipped.append((cid, "no full record in grounding jsonl"))
            continue
        ep = edits_dir / f"{cid}_full.png"
        edit = None
        if ep.exists():
            with Image.open(ep) as e:
                edit = e.convert("RGB")
        scale = heat_scale(list(maps.values()))
        panel = build_panel(cid, src, tm, sm, maps, recs, edit,
                            outcomes.get(cid, "unknown"), audits.get(cid), scale)
        panel.save(out / f"{cid}.png")
        cases.append({"id": cid, "outcome": outcomes.get(cid, "unknown"),
                      "group": _group_name(recs["full"]),
                      "pref": recs["full"].get("target_pref") or 0.0,
                      "scale": scale, "has_edit": edit is not None})

    # grouping: outcome -> points/misses -> sorted by full target_pref (high first)
    groups = {}
    for c in cases:
        groups.setdefault(c["outcome"], {"points": [], "misses": []})[c["group"]].append(c)
    sheets = []
    for oc in sorted(groups):
        for g in ("points", "misses"):
            lst = sorted(groups[oc][g], key=lambda c: -c["pref"])
            groups[oc][g] = lst
            for n, i in enumerate(range(0, len(lst), 12), 1):
                chunk = lst[i:i + 12]
                imgs = [Image.open(out / f"{c['id']}.png") for c in chunk]
                sheet = Image.new("RGB", (imgs[0].width, sum(m.height for m in imgs)))
                y = 0
                for m in imgs:
                    sheet.paste(m, (0, y))
                    y += m.height
                name = f"sheet_{oc}_{g}_{n}.png"
                sheet.save(out / name)
                sheets.append((oc, g, name))
    _write_index(out, groups, sheets)
    n_miss_edit = sum(not c["has_edit"] for c in cases)
    print(f"panels: {len(cases)}  sheets: {len(sheets)}  skipped: {len(skipped)}"
          f"  panels without edit image: {n_miss_edit}")
    for cid, why in skipped:
        print(f"  skipped {cid}: {why}")
    return cases, sheets, skipped


def _write_index(out, groups, sheets):
    esc = html.escape
    p = ["<!doctype html><html><head><meta charset='utf-8'><title>klein grounding heat maps</title>",
         "<style>body{font:14px sans-serif;background:#161616;color:#ddd;margin:16px}"
         "img{max-width:100%;display:block;margin:6px 0}h2{margin-top:32px}"
         "h3{margin:18px 0 4px;color:#aaa}a{color:#8cf}.leg{background:#222;padding:8px 14px;"
         "border-radius:6px}</style></head><body><h1>Klein grounding: attention from "
         "instruction to reference tokens</h1><div class='leg'><b>Legend.</b> Green outline = "
         "target instance; red outlines = sibling instances. Each row has five tiles: "
         "recovered source (lossy, about +-3 grey levels) | plain klein edit | attention for "
         "the FULL instruction | SINGLE | SWAP. Heat = attention density as a multiple of "
         "uniform (1/1024 per token), one scale per case (0 to the pooled 99th percentile of "
         "the three maps, shown under each tile as &quot;red = N.Nx uniform&quot;; higher "
         "saturates), so the three heat tiles of a case are comparable but different cases "
         "are not. Low-attention areas stay visible as dimmed greyscale. Cases are sorted by "
         "the full instruction's target_pref, highest first.</div>"]
    for oc in sorted(groups):
        tot = sum(len(v) for v in groups[oc].values())
        p.append(f"<h2>{esc(oc)} ({tot})</h2>")
        for g, title in (("points", "full instruction points at target"),
                         ("misses", "full instruction does not point at target")):
            lst = groups[oc][g]
            if not lst:
                continue
            links = " ".join(f"<a href='{esc(n)}'>{esc(n)}</a>" for o, gg, n in sheets
                             if o == oc and gg == g)
            p.append(f"<h3>{esc(oc)} / {esc(title)} ({len(lst)})</h3><div>sheets: {links}</div>")
            for c in lst:
                p.append(f"<img loading='lazy' src='{esc(c['id'])}.png' alt='{esc(c['id'])}'>")
    p.append("</body></html>")
    (out / "index.html").write_text("\n".join(p), encoding="utf-8")


# ------------------------------------------------------------- self-test ---

def _outline(mask, px=2):
    m = np.asarray(mask, bool)
    er = m.copy()
    for _ in range(px):
        e = er.copy()
        e[1:] &= er[:-1]; e[:-1] &= er[1:]; e[:, 1:] &= er[:, :-1]; e[:, :-1] &= er[:, 1:]
        er = e
    return m & ~er


def _probe_overlay(src_arr, amap, tmask, smasks):
    """Copy of klein_grounding_probe._save_overlay, returning the image."""
    h, w = src_arr.shape[:2]
    src = src_arr.astype(np.float64)
    out = src * (0.35 + 0.65 * _quant_a(amap, (w, h))[..., None])
    out[_outline(tmask)] = GREEN
    for s in smasks:
        out[_outline(s)] = RED
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def self_test():
    rng = np.random.default_rng(0)
    H = W = 512
    yy, xx = np.mgrid[0:H, 0:W]
    src = np.stack([60 + 120 * xx / W, 40 + 150 * yy / H,
                    120 + 60 * np.sin(xx / 25.0)], -1)
    src = (src + rng.integers(0, 6, src.shape)).clip(20, 235).astype(np.uint8)
    amap = np.full((32, 32), 0.5 / 1024, np.float32)
    amap[8:16, 8:16] = 8.0 / 1024            # known hot region
    amap[28, 28] = 400.0 / 1024              # extreme outlier token
    tmask = np.zeros((H, W), bool); tmask[120:260, 120:260] = True
    smask = np.zeros((H, W), bool); smask[330:450, 300:460] = True
    png = _probe_overlay(src, amap, tmask, [smask])

    rec, rt, rs = recover_source(png, amap)
    out_m = _outline(tmask) | _outline(smask)
    err = np.abs(rec - src.astype(np.float64))[~out_m]
    assert err.max() <= 4, f"recovered source error {err.max()}"
    assert (rt == _outline(tmask)).all() and (rs == _outline(smask)).all(), "outline masks"
    print(f"  (a) max recovery error off outlines: {err.max():.2f} grey levels")

    maps = {k: amap.astype(np.float64) for k in KINDS}
    scale = heat_scale(list(maps.values()))
    gray = np.asarray(Image.fromarray(src).convert("L")).astype(np.float64)
    _, alpha = render_heat(gray, amap, scale)
    hot = alpha[8 * 16 + 24:16 * 16 - 24, 8 * 16 + 24:16 * 16 - 24].mean()
    cold = alpha[256:400, 40:200].mean()
    assert hot > 0.6 and hot > 5 * cold, f"hot alpha {hot:.2f} vs outside {cold:.2f}"
    print(f"  (c) heat alpha hot {hot:.2f} vs outside {cold:.2f} (scale {scale:.1f}x, outlier 400x)")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        md = td / "maps"; md.mkdir()
        for k in KINDS:
            np.save(md / f"case-1_{k}.npy", amap)
        png.save(md / "case-1_full.png")
        with open(td / "g.jsonl", "w", encoding="utf-8") as fh:
            for k in KINDS:
                fh.write(json.dumps({"id": "case-1", "prompt_kind": k, "instruction":
                                     "Make the thing " * 8, "pointing": k == "full",
                                     "target_pref": 1.44, "n_instances": 2}) + "\n")
        with open(td / "o.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": "case-1", "outcome": "measurable"}) + "\n")
        args = argparse.Namespace(maps=str(md), grounding=str(td / "g.jsonl"),
                                  outcomes=str(td / "o.jsonl"), edits=str(td / "none"),
                                  audit=[f"plain:{td / 'nope.json'}:{td / 'nope2.json'}"],
                                  out=str(td / "out"), ids=None)
        cases, sheets, skipped = run(args)
        o = td / "out"
        for f in ("case-1.png", "index.html", "sheet_measurable_points_1.png"):
            assert (o / f).exists(), f"not written: {f}"
        assert len(cases) == 1 and not skipped
        assert "case-1.png" in (o / "index.html").read_text(encoding="utf-8")
        print("  (d) files written: case-1.png, index.html, sheet_measurable_points_1.png")
    print("SELF-TEST OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = "runs/gate_c/"
    ap.add_argument("--maps", default=g + "klein_grounding_maps")
    ap.add_argument("--grounding", default=g + "klein_grounding.jsonl")
    ap.add_argument("--outcomes", default=g + "klein4b_none.jsonl")
    ap.add_argument("--edits", default=g + "img_klein4b_none")
    ap.add_argument("--audit", action="append", default=None,
                    help="NAME:LEDGER.json:KEY.json (repeatable); default: plain + paste-back")
    ap.add_argument("--out", default=g + "klein_grounding_heat")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if a.audit is None:
        a.audit = [f"plain:{g}blind_klein_ledger.json:{g}blind_klein_key.json",
                   f"paste-back:{g}blind_comp_ledger.json:{g}blind_comp_key.json"]
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
