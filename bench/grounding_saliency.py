"""Which image features does klein's instruction attention follow when the
instruction cannot pick out the referent?  (GROUNDING_DIAGNOSTIC.md, "Feature check")

Seen in the heat maps: under a swapped instruction, and often the single one,
attention sits on the larger, nearer, brighter or more colourful instance.
This puts a number on that. No model, no GPU.

    python bench/grounding_saliency.py --self-test
    python bench/grounding_saliency.py --extract          # needs /data (COCO + bench)
    python bench/grounding_saliency.py --report runs/gate_c/klein_grounding_features.jsonl

--extract rebuilds each case's masks exactly as the probe did (run_gate_c.
build_masks; target first, then siblings) and measures, per instance:

    area   fraction of the image the mask covers
    low    centroid height, 0 = top edge, 1 = bottom edge (lower ~ nearer)
    lum    mean luminance inside the mask
    sat    mean HSV saturation inside the mask

BEFORE TRUSTING A FEATURE it recomputes the attention densities from the saved
map with these masks and compares them with what the probe wrote. If the masks
or their order differ from the probe's, the features would be attached to the
wrong instance and every number below would be wrong; the script stops instead.

--report asks, for each instruction kind and each feature: is the instance with
the highest attention DENSITY also the one that is largest / lowest / brightest
/ most saturated? Only the target and its same-category siblings are compared;
attention on other objects or people in the scene is not tested here.

SIZE IS PARTLY BUILT INTO DENSITY. `pointing` weights each token cell by how
much of it the mask covers. For a map that is flat on objects and flat on
background, an instance's density is  a_bg + (a_obj - a_bg) * sum(c^2)/sum(c)
with c the per-cell coverage. That factor ("fill") is near 1 for a large compact
mask and small for one a couple of cells wide, so equal attention per object
pixel still ranks the larger instance higher. Two controls are therefore
stored and printed: `fill` per instance, and INTERIOR densities, the mean over
cells the mask covers at least 90%, which do not have this bias (nan for an
instance with no such cell).

PRIMARY (fixed in GROUNDING_DIAGNOSTIC.md before any run): swapped instruction
whose source case has a DIFFERENT category, two-instance cases, chance 50%.
The interior-only line under it decides whether an `area` (or `low`) result
can be called size. Everything else printed is context.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

FEATURES = (("area", "larger"), ("low", "lower in the frame"),
            ("lum", "brighter"), ("sat", "more saturated"))
KINDS = ("swap", "single", "full")


# ----------------------------------------------------------- pure helpers ---

def instance_features(rgb: np.ndarray, masks: list[np.ndarray]) -> dict:
    """rgb: (H, W, 3) uint8. masks: bool (H, W), target first. -> {name: [per instance]}."""
    a = np.asarray(rgb, dtype=np.float64) / 255.0
    H, W = a.shape[:2]
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 1e-9, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    ys = (np.arange(H) + 0.5) / H
    out = {k: [] for k, _ in FEATURES}
    for m in masks:
        m = np.asarray(m) > 0
        if m.shape != (H, W):
            raise ValueError(f"mask {m.shape} does not match image {(H, W)}")
        n = int(m.sum())
        if n == 0:
            for k in out:
                out[k].append(float("nan"))
            continue
        out["area"].append(n / m.size)
        out["low"].append(float((m.sum(axis=1) * ys).sum() / n))
        out["lum"].append(float(lum[m].mean()))
        out["sat"].append(float(sat[m].mean()))
    return out


def top(vals) -> int | None:
    """Index of the single largest value; None on a tie or a nan."""
    v = [float(x) for x in vals]
    if any(math.isnan(x) for x in v):
        return None
    best = max(v)
    idx = [i for i, x in enumerate(v) if x == best]
    return idx[0] if len(idx) == 1 else None


def count_p(k: int, ps: list[float]) -> float:
    """Exact two-sided p for k successes when trial i succeeds with probability
    ps[i] (plain binomial if they are all equal). Sums every outcome no more
    likely than the observed one."""
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for j, d in enumerate(dist):
            nxt[j] += d * (1 - p)
            nxt[j + 1] += d * p
        dist = nxt
    return min(1.0, sum(d for d in dist if d <= dist[k] * (1 + 1e-9)))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided p for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 in (0, n) or c1 in (0, n):
        return 1.0

    def pr(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)

    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, r1 - (n - c1)), min(r1, c1) + 1)
                        if pr(x) <= p0 * (1 + 1e-9)))


def fill_factor(pm: np.ndarray) -> float:
    """sum(c^2)/sum(c) over the pooled mask: how much of the mask's weight sits
    in cells it fully covers. See the module docstring."""
    s = float(pm.sum())
    return float((pm ** 2).sum() / s) if s > 1e-9 else float("nan")


def interior_density(amap: np.ndarray, pm: np.ndarray, thr: float = 0.9) -> float:
    """Mean attention over the cells the mask covers at least `thr`; nan if none."""
    cells = pm >= thr
    if not cells.any():
        return float("nan")
    return float(np.asarray(amap, dtype=np.float64)[cells].mean())


def check_densities(amap, masks, recorded, pool_mask, densities, tol=1e-3):
    """Largest relative gap between the probe's recorded densities and the ones
    these masks give on the saved map. None if they cannot be compared."""
    if len(recorded) != len(masks):
        return float("inf")
    gh, gw = amap.shape
    got = densities(np.asarray(amap, dtype=np.float64),
                    [pool_mask(m, gh, gw) for m in masks])
    gap = 0.0
    for g, r in zip(got, recorded):
        if math.isnan(g) or math.isnan(r):
            return float("inf")
        gap = max(gap, abs(g - r) / max(abs(r), 1e-12))
    return gap


def _read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


# ---------------------------------------------------------------- extract ---

def extract(args) -> None:
    from PIL import Image  # noqa: PLC0415

    from klein_grounding_probe import densities, pool_mask  # noqa: PLC0415
    from refedit_siblings import bench_image, load_bench, load_coco  # noqa: PLC0415
    from run_gate_c import build_masks  # noqa: PLC0415

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    coco_index, _c, _u = load_coco(Path(args.coco_annotations),
                                   tuple(args.coco_splits.split(",")),
                                   cache=Path(args.coco_cache))
    rows = {r["index"]: r for r in load_bench(args.bench)}
    full = {r["id"]: r for r in _read_jsonl(args.grounding) if r["prompt_kind"] == "full"}
    maps_dir = Path(args.maps)

    recs, skipped, checked, worst, bad = [], [], 0, 0.0, []
    for case in cases:
        cid = case["id"]
        row = rows.get(int(cid.rsplit("-", 1)[1]))
        if row is None:
            skipped.append((cid, "no bench row"))
            continue
        tmask, smasks = build_masks(case, coco_index)
        if tmask is None:
            skipped.append((cid, "masks unavailable"))
            continue
        src = bench_image(row["input_image"])
        if src.size != (512, 512):                       # as the probe does
            src = src.resize((512, 512), Image.BILINEAR)
        masks = [np.asarray(tmask) > 0] + [np.asarray(s) > 0 for s in smasks]
        feats = instance_features(np.asarray(src.convert("RGB")), masks)

        g, mp = full.get(cid), maps_dir / f"{cid}_full.npy"
        if g is not None and mp.exists():
            gap = check_densities(np.load(mp), masks, g["agg_densities"],
                                  pool_mask, densities)
            checked += 1
            worst = max(worst, gap)
            if gap > args.tol:
                bad.append((cid, gap))

        # controls for the size bias in coverage-weighted density
        interior, pooled = {}, None
        for kind in KINDS:
            kp = maps_dir / f"{cid}_{kind}.npy"
            if not kp.exists():
                continue
            amap = np.load(kp)
            if pooled is None:
                pooled = [pool_mask(m, *amap.shape) for m in masks]
            interior[kind] = [interior_density(amap, pm) for pm in pooled]
        if pooled is None:
            gh, gw = (g or {}).get("grid", (32, 32))
            pooled = [pool_mask(m, gh, gw) for m in masks]
        recs.append({"id": cid, "n_instances": len(masks), **feats,
                     "fill": [fill_factor(pm) for pm in pooled],
                     "interior": interior})

    print(f"{len(recs)} cases measured, {len(skipped)} skipped"
          + "".join(f"\n  skipped {c}: {w}" for c, w in skipped))
    print(f"mask check against the probe's densities: {checked} cases compared, "
          f"largest relative gap {worst:.2e} (tolerance {args.tol:.0e})")
    if not checked:
        print("  NOTHING WAS COMPARED (no saved maps or no grounding records). The "
              "features are unverified against the probe's masks.")
    if bad and not args.no_check:
        raise SystemExit(
            f"{len(bad)} cases disagree with the probe (e.g. {bad[:3]}). The masks "
            "or their order are not the probe's, so features would be attached to "
            "the wrong instance. Nothing written. (--no-check to write anyway.)")
    n_int = sum(1 for r in recs if r["interior"].get("swap")
                and top(r["interior"]["swap"]) is not None)
    print(f"interior densities (swap map): usable in {n_int} of {len(recs)} cases "
          "(every instance has a cell it covers at least 90%)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    print(f"wrote {out}\n  next: python bench/grounding_saliency.py --report {out}")


# ----------------------------------------------------------------- report ---

def report(feats: list[dict], grounding: list[dict], categories: dict | None = None,
           outcomes: dict | None = None, verbose: bool = True) -> dict:
    """feats: one record per case. grounding: probe records. Returns the numbers
    it prints, keyed for the self-test."""
    say = print if verbose else (lambda *a, **k: None)
    F = {r["id"]: r for r in feats}
    G = {}
    for r in grounding:
        G.setdefault(r["prompt_kind"], {})[r["id"]] = r
    res = {}

    # which case each swapped instruction came from, and whether it shares the category
    by_instr = {}
    for cid, r in G.get("full", {}).items():
        by_instr.setdefault(r["instruction"], []).append(cid)
    diff_cat = set()
    if categories:
        for cid, r in G.get("swap", {}).items():
            src = by_instr.get(r["instruction"], [])
            if len(src) == 1 and src[0] in categories and cid in categories \
                    and categories[src[0]] != categories[cid]:
                diff_cat.add(cid)

    def dens(kind, cid, interior=False):
        if interior:
            return F[cid].get("interior", {}).get(kind) or [float("nan")]
        return G[kind][cid]["agg_densities"]

    def usable(kind, interior=False):
        out = []
        for cid, g in G.get(kind, {}).items():
            f = F.get(cid)
            if f is None or len(g["agg_densities"]) != f["n_instances"]:
                continue
            d = dens(kind, cid, interior)
            if len(d) != f["n_instances"] or top(d) is None:
                continue
            out.append(cid)
        return out

    def match(kind, ids, feat, interior=False):
        """(k, n, ps): attention's top instance is also the feature's top instance."""
        k, ps = 0, []
        for cid in ids:
            ft = top(F[cid][feat])
            if ft is None:
                continue
            ps.append(1.0 / F[cid]["n_instances"])
            k += top(dens(kind, cid, interior)) == ft
        return k, len(ps), ps

    def line(label, kind, ids, interior=False):
        cells = []
        for feat, _ in FEATURES:
            k, n, ps = match(kind, ids, feat, interior)
            p = count_p(k, ps) if n else float("nan")
            res[(label, kind, feat)] = (k, n, p)
            cells.append(f"{k:>3}/{n:<3} {100 * k / n if n else float('nan'):5.1f}% p={p:<7.4f}")
        chance = np.mean([1.0 / F[c]["n_instances"] for c in ids]) if ids else float("nan")
        say(f"  {label:<34} chance {100 * chance:4.1f}%  " + "  ".join(cells))

    say("=" * 100)
    say("Does attention's top instance coincide with the top instance on each feature?")
    say("  columns: " + " | ".join(f"{f} ({w})" for f, w in FEATURES))
    say("  p: exact two-sided, against each case's own 1/n_instances. FOUR features are "
        "tested;\n  the pre-set bar is p < 0.0125 (0.05 / 4). Only the target and its "
        "same-category siblings are compared.")
    say("=" * 100)

    if "swap" in G:
        ids = usable("swap")
        prim = [c for c in ids if c in diff_cat and F[c]["n_instances"] == 2]
        say("\nPRIMARY  swapped instruction, source case of a different category, "
            "two instances")
        if not categories:
            say("  (no --cases file: partner category unknown, primary subset empty)")
        line("primary", "swap", prim)
        res["primary_n"] = len(prim)
        keep = set(usable("swap", interior=True))
        prim_int = [c for c in prim if c in keep]
        line("primary, interior cells only", "swap", prim_int, interior=True)
        res["primary_interior_n"] = len(prim_int)
        both = [c for c in prim if top(F[c]["area"]) is not None
                and top(F[c].get("fill", [float("nan")])) is not None]
        same = sum(top(F[c]["area"]) == top(F[c]["fill"]) for c in both)
        res["area_fill"] = (same, len(both))
        say(f"  size control: interior line uses {len(prim_int)} of {len(prim)} primary "
            f"cases; the larger instance is also the\n  higher-fill one in {same} of "
            f"{len(both)}. An `area` or `low` result counts as size or position only if "
            "the\n  interior-only line agrees with it.")

    for kind in KINDS:
        if kind not in G:
            continue
        ids = usable(kind)
        say(f"\n{kind.upper()} instruction")
        line("two-instance cases", kind, [c for c in ids if F[c]["n_instances"] == 2])
        line("all cases", kind, ids)
        if kind == "swap" and categories:
            line("all, different-category partner", kind, [c for c in ids if c in diff_cat])
            line("all, same-category partner", kind, [c for c in ids if c not in diff_cat])

    # features travel together (big objects are also low in the frame): show it,
    # and where two features disagree, show which one attention goes with.
    if "swap" in G:
        ids = [c for c in usable("swap") if F[c]["n_instances"] == 2
               and all(top(F[c][f]) is not None for f, _ in FEATURES)]
        say(f"\nWHEN TWO FEATURES DISAGREE  (swapped instruction, two-instance cases, n={len(ids)})")
        say("  'agree' = the same instance is top on both. Where they disagree, which "
            "does attention follow?")
        names = [f for f, _ in FEATURES]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                agree = [c for c in ids if top(F[c][a]) == top(F[c][b])]
                dis = [c for c in ids if top(F[c][a]) != top(F[c][b])]
                ka = sum(top(G["swap"][c]["agg_densities"]) == top(F[c][a]) for c in dis)
                p = count_p(ka, [0.5] * len(dis)) if dis else float("nan")
                res[("disagree", a, b)] = (ka, len(dis))
                say(f"  {a:>4} vs {b:<4}  agree {len(agree):>3}   disagree {len(dis):>3}: "
                    f"attention follows {a} in {ka}, {b} in {len(dis) - ka}   p={p:.4f}")

    # the same question from the target's side: is "points at the target" mostly
    # "the target happens to be the big / low / bright one"?
    say("\nPOINTING AT THE TARGET, split by whether the target is top on the feature")
    for kind in KINDS:
        if kind not in G:
            continue
        ids = usable(kind)
        for feat, word in FEATURES:
            yes = [c for c in ids if top(F[c][feat]) == 0]
            no = [c for c in ids if top(F[c][feat]) not in (0, None)]
            py = sum(top(G[kind][c]["agg_densities"]) == 0 for c in yes)
            pn = sum(top(G[kind][c]["agg_densities"]) == 0 for c in no)
            p = fisher(py, len(yes) - py, pn, len(no) - pn)
            res[("target", kind, feat)] = (py, len(yes), pn, len(no), p)
            say(f"  {kind:<6} {feat:<4} target is {word:<19} {py:>3}/{len(yes):<3}"
                f" ({100 * py / max(len(yes), 1):3.0f}%)   is not {pn:>3}/{len(no):<3}"
                f" ({100 * pn / max(len(no), 1):3.0f}%)   p={p:.4f}")

    if outcomes:
        say("\nPLAIN EDIT OFF TARGET (mis-targeted or off-instance), same split")
        ids = [c for c in F if c in outcomes]
        for feat, word in FEATURES:
            yes = [c for c in ids if top(F[c][feat]) == 0]
            no = [c for c in ids if top(F[c][feat]) not in (0, None)]
            oy = sum(outcomes[c] != "measurable" for c in yes)
            on = sum(outcomes[c] != "measurable" for c in no)
            p = fisher(oy, len(yes) - oy, on, len(no) - on)
            res[("edit", feat)] = (oy, len(yes), on, len(no), p)
            say(f"  {feat:<4} target is {word:<19} off target {oy:>3}/{len(yes):<3}"
                f" ({100 * oy / max(len(yes), 1):3.0f}%)   is not {on:>3}/{len(no):<3}"
                f" ({100 * on / max(len(no), 1):3.0f}%)   p={p:.4f}")

    say("\nReading is in GROUNDING_DIAGNOSTIC.md ('Feature check'), fixed before this ran.")
    return res


# -------------------------------------------------------------- self-test ---

def self_test() -> None:
    # exact tests against values worked by hand
    assert abs(count_p(5, [0.5] * 10) - 1.0) < 1e-12
    assert abs(count_p(9, [0.5] * 10) - 22 / 1024) < 1e-12        # 0,1,9,10
    assert abs(count_p(2, [0.5, 0.5]) - 0.5) < 1e-12              # 0 and 2
    assert abs(fisher(3, 1, 1, 3) - 34 / 70) < 1e-12
    assert top([1.0, 2.0, 0.5]) == 1 and top([1.0, 1.0]) is None
    assert top([float("nan"), 1.0]) is None
    print("  exact tests and tie handling OK")

    # features on an image whose answers are known by construction
    H = W = 64
    img = np.zeros((H, W, 3), np.uint8)
    big = np.zeros((H, W), bool)
    big[36:60, 8:56] = True                    # large, low, dark grey
    small = np.zeros((H, W), bool)
    small[4:12, 4:12] = True                   # small, high, bright pure red
    img[big] = (40, 40, 40)
    img[small] = (255, 0, 0)
    f = instance_features(img, [big, small])
    assert f["area"][0] > f["area"][1] and abs(f["area"][0] - 24 * 48 / 4096) < 1e-12
    assert f["low"][0] > f["low"][1] and abs(f["low"][1] - 8 / 64) < 1e-12
    assert f["lum"][1] > f["lum"][0] and abs(f["lum"][0] - 40 / 255) < 1e-9
    assert f["sat"][1] == 1.0 and f["sat"][0] == 0.0
    assert math.isnan(instance_features(img, [np.zeros((H, W), bool)])["lum"][0])
    try:
        instance_features(img, [np.zeros((8, 8), bool)])
        raise AssertionError("shape mismatch not refused")
    except ValueError:
        pass
    print("  features OK (area, height, luminance, saturation; empty mask; shape guard)")

    # the mask check must pass for the probe's own masks and catch a swapped order
    from klein_grounding_probe import densities, pool_mask  # noqa: PLC0415
    rng = np.random.default_rng(0)
    amap = rng.random((8, 8))
    amap /= amap.sum()
    masks = [big, small]
    recorded = densities(amap, [pool_mask(m, 8, 8) for m in masks])
    assert check_densities(amap.astype(np.float32), masks, recorded,
                           pool_mask, densities) < 1e-5
    assert check_densities(amap, masks[::-1], recorded, pool_mask, densities) > 1e-2
    assert check_densities(amap, masks[:1], recorded, pool_mask, densities) == float("inf")
    print("  mask check OK (passes on the probe's masks, catches a swapped order)")

    # the size bias, and the control for it: a map that is flat on objects gives
    # the larger block the higher coverage-weighted density, and equal interior
    # densities.
    small2 = np.zeros((H, W), bool)
    small2[4:28, 4:28] = True                  # 2 x 2 full cells with a half-covered rim
    pb, ps = pool_mask(big, 8, 8), pool_mask(small2, 8, 8)
    flat = (pb + ps) / (pb + ps).sum()
    d_big, d_small = densities(flat, [pb, ps])
    assert abs(fill_factor(pb) - 15 / 18) < 1e-12
    assert abs(fill_factor(ps) - 6.25 / 9) < 1e-12
    assert d_big > d_small * 1.15, (d_big, d_small)               # the bias
    assert abs(d_big / d_small - fill_factor(pb) / fill_factor(ps)) < 1e-9
    assert abs(interior_density(flat, pb) - interior_density(flat, ps)) < 1e-15
    assert math.isnan(interior_density(flat, pool_mask(small, 8, 8)))  # no full cell
    print(f"  size bias reproduced (flat map: density ratio {d_big / d_small:.2f} = "
          "fill ratio) and removed by interior densities")

    # report: attention that follows AREA by construction under 'swap', and the
    # target under 'full'. Brightness is independent of both.
    feats, grounding, cats = [], [], {}
    n = 200
    for i in range(n):
        cid = f"t-{i}"
        area = rng.permutation([0.3, 0.1]).tolist()
        # interior densities are a coin flip here: the planted effect is ALL fill
        # bias. In the last 20 cases the small instance has no interior cell.
        coin = rng.random(2).tolist()
        feats.append({"id": cid, "n_instances": 2, "area": area,
                      "low": rng.random(2).tolist(), "lum": rng.random(2).tolist(),
                      "sat": rng.random(2).tolist(),
                      "fill": [0.9 if a > 0.2 else 0.5 for a in area],
                      "interior": {"swap": coin if i < n - 20 else [1.0, float("nan")]}})
        cats[cid] = "a" if i % 2 else "b"
        other = f"t-{(i + 101) % n}"            # odd offset: always the other category
        grounding.append({"id": cid, "prompt_kind": "full", "instruction": f"edit {cid}",
                          "agg_densities": [2.0, 1.0]})
        grounding.append({"id": cid, "prompt_kind": "swap", "instruction": f"edit {other}",
                          "agg_densities": [a + 1e-3 * rng.random() for a in area]})
    res = report(feats, grounding, cats, verbose=False)
    assert res["primary_n"] == n, res["primary_n"]
    k, m, p = res[("primary", "swap", "area")]
    assert (k, m) == (n, n) and p < 1e-20, (k, m, p)
    k, m, p = res[("primary", "swap", "lum")]
    assert 0.35 * n < k < 0.65 * n and p > 0.001, (k, m, p)
    assert res["primary_interior_n"] == n - 20, res["primary_interior_n"]
    k, m, p = res[("primary, interior cells only", "swap", "area")]
    assert m == n - 20 and 0.35 * m < k < 0.65 * m and p > 0.001, (k, m, p)
    assert res["area_fill"] == (n, n)
    py, ny, pn, nn, p = res[("target", "full", "area")]
    assert py == ny and pn == nn                 # full always points at the target
    py, ny, pn, nn, p = res[("target", "swap", "area")]
    assert py == ny and pn == 0 and p < 1e-20    # swap points at it only when it is big
    # same-category partners must fall out of the primary subset
    res2 = report(feats, grounding, {c: "a" for c in cats}, verbose=False)
    assert res2["primary_n"] == 0
    print("  report OK (recovers a planted area effect, not a brightness one; shows it "
          "vanish on\n  interior densities when it is all fill bias; "
          "partner-category filter)")
    print("SELF-TEST OK")


# ------------------------------------------------------------------- main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="bench/cases_refedit_final.json")
    p.add_argument("--bench",
                   default="/data/datasets/RefEdit-Bench/data/test-00000-of-00001.parquet")
    p.add_argument("--coco-annotations", default="/data/datasets/coco/annotations")
    p.add_argument("--coco-splits", default="train2014")
    p.add_argument("--coco-cache", default="/data/datasets/coco/desc")
    p.add_argument("--grounding", default="runs/gate_c/klein_grounding.jsonl")
    p.add_argument("--maps", default="runs/gate_c/klein_grounding_maps",
                   help="the probe's --save-maps directory, for the mask check")
    p.add_argument("--outcomes", default="runs/gate_c/klein4b_none.jsonl")
    p.add_argument("--out", default="runs/gate_c/klein_grounding_features.jsonl")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--no-check", action="store_true",
                   help="write features even if the mask check fails")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--report", default=None, metavar="FEATURES_JSONL")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
    elif args.extract:
        extract(args)
    elif args.report:
        cats = None
        if Path(args.cases).exists():
            cats = {c["id"]: c["category"] for c in
                    json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]}
        outcomes = None
        if Path(args.outcomes).exists():
            outcomes = {r["id"]: r["outcome"] for r in _read_jsonl(args.outcomes)
                        if r.get("prompt_kind", "full") == "full" and r.get("outcome")}
        report(_read_jsonl(args.report), _read_jsonl(args.grounding), cats, outcomes)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
