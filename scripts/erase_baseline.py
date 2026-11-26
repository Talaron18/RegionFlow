"""How well does an instruction-free "erase the source" predictor score on
Gate A's cosine metrics?

    python scripts/erase_baseline.py --cache /data/cache/13061d51aee4 \
        --bias-npy runs/bias.npy

WHY. The training target is v = z_tgt - z_src, which contains -z_src inside
the mask. A predictor that never reads the instruction and only removes what
is there already correlates with v. The text ablation (STATUS_REVIEW.md 4b)
found both trained arms ignore the instruction, and the synthetic red->blue
test scores cos 0.66-0.75 in-mask while producing grey. This measures how
much of the reported cosine such a predictor gets for free.

Three predictors, none of which sees the instruction or the target item:

    erase     u = (c - z_src) * M     c = per-channel mean of in-mask TARGET
                                      latents, fit on the other half of the
                                      cache (two-fold, so no item scores
                                      against a constant it helped fit)
    flatten   u = (mean_in_mask(z_src) - z_src) * M
                                      removes spatial detail, keeps the
                                      region's own average. pooled/centered
                                      are undefined (its pooled value is 0).
    constant  u = b * M               b = per-channel mean pooled displacement
                                      from the other half. No source content
                                      at all: the "same shift everywhere"
                                      predictor.

Scored exactly as transport/pairflow.py::direction_position_diagnostics
scores a checkpoint (raw / pooled / centered), so the numbers sit next to
`gate_a_transport.py diagnose` output directly. CPU and numpy only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den > 1e-8 else float("nan")


def pool(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-channel mean of x (C,H,W) inside m (1,H,W)."""
    return (x * m).sum(axis=(1, 2)) / max(float(m.sum()), 1e-8)


def score(u: np.ndarray, v: np.ndarray, m: np.ndarray,
          bias: np.ndarray | None) -> dict:
    up, vp = pool(u, m), pool(v, m)
    out = {"raw": _cos(u, v), "pooled": _cos(up, vp)}
    if bias is not None:
        out["centered"] = _cos(up - bias, vp - bias)
    return out


def predictors(z_src: np.ndarray, m: np.ndarray, c: np.ndarray,
               b: np.ndarray) -> dict:
    C = z_src.shape[0]
    return {
        "erase": (c.reshape(C, 1, 1) - z_src) * m,
        "flatten": (pool(z_src, m).reshape(C, 1, 1) - z_src) * m,
        "constant": b.reshape(C, 1, 1) * m,
    }


def evaluate(items: list[dict], bias: np.ndarray | None, seed: int = 0) -> dict:
    """items: dicts with z_src, z_tgt (C,H,W) and mask (1,H,W)."""
    items = [it for it in items if it["mask"].sum() > 0]
    order = np.random.default_rng(seed).permutation(len(items))
    folds = [order[::2], order[1::2]]
    acc: dict = {}
    for k in (0, 1):
        fit = [items[i] for i in folds[1 - k]]
        c = np.mean([pool(it["z_tgt"], it["mask"]) for it in fit], axis=0)
        b = np.mean([pool(it["z_tgt"] - it["z_src"], it["mask"]) for it in fit],
                    axis=0)
        for i in folds[k]:
            it = items[i]
            v = it["z_tgt"] - it["z_src"]
            for name, u in predictors(it["z_src"], it["mask"], c, b).items():
                for key, val in score(u, v, it["mask"], bias).items():
                    acc.setdefault(name, {}).setdefault(key, []).append(val)
    summary = {}
    for name, d in acc.items():
        summary[name] = {}
        for key, vals in d.items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            summary[name][key] = ({"n": int(arr.size),
                                   "mean": float(arr.mean()),
                                   "median": float(np.median(arr))}
                                  if arr.size else {"n": 0})
    return summary


def load_cache(root: Path) -> list[dict]:
    meta = json.loads((root / "meta.json").read_text())
    out = []
    for rec in meta["items"]:
        d = np.load(root / f"{rec['stem']}.npz")
        if "mask" not in d.files:
            continue
        m = d["mask"].astype(np.float64)
        out.append({"z_src": d["z_src"].astype(np.float64),
                    "z_tgt": d["z_tgt"].astype(np.float64),
                    "mask": m.reshape(1, *m.shape[-2:])})
    return out


def report(summary: dict) -> None:
    print(f"\n{'predictor':<10}{'n':>6}{'raw':>9}{'pooled':>9}{'centered':>10}"
          "   (median cosine)")
    for name in ("erase", "flatten", "constant"):
        d = summary[name]

        def _m(key):
            e = d.get(key, {})
            if not e.get("n"):
                return "--"
            # a pooled vector of ~0 gives a meaningless cosine
            if name == "flatten" and key != "raw":
                return "--"
            return f"{e['median']:+.3f}"
        print(f"{name:<10}{d['raw']['n']:>6}{_m('raw'):>9}{_m('pooled'):>9}"
              f"{_m('centered'):>10}")
    print("\ncompare with `gate_a_transport.py diagnose` on the same cache "
          "(trained arms: raw ~0.54, pooled ~0.77, centered ~0.67). If "
          "'erase' is close to those, the cosine was mostly measuring "
          "removal of the source content, not the requested edit.")


def self_test() -> None:
    rng = np.random.default_rng(0)
    C, H = 4, 8
    m = np.zeros((1, H, H)); m[:, 2:6, 2:6] = 1

    def make(kind: str) -> list[dict]:
        items = []
        fill = rng.normal(size=C)
        for _ in range(40):
            zs = rng.normal(size=(C, H, H))
            if kind == "repaint":      # region replaced by one flat fill
                zt = zs * (1 - m) + fill.reshape(C, 1, 1) * m
            else:                      # item-specific shift, source kept
                zt = zs + rng.normal(size=C).reshape(C, 1, 1) * m
            items.append({"z_src": zs, "z_tgt": zt, "mask": m})
        return items

    rep = evaluate(make("repaint"), None)
    # erase is exactly right when every target is the same flat fill
    assert rep["erase"]["raw"]["median"] > 0.99, rep["erase"]
    assert rep["flatten"]["raw"]["median"] > 0.5, rep["flatten"]
    shift = evaluate(make("shift"), None)
    # with the source kept and a random per-item shift, removing the source
    # points the wrong way: erase must NOT look good here
    assert shift["erase"]["raw"]["median"] < 0.2, shift["erase"]
    assert abs(shift["constant"]["raw"]["median"]) < 0.5, shift["constant"]
    # scoring matches the definition on a hand case
    u = np.ones((C, H, H)) * m
    s = score(u, 2 * u, m, bias=np.zeros(C))
    assert abs(s["raw"] - 1) < 1e-9 and abs(s["pooled"] - 1) < 1e-9
    assert abs(s["centered"] - 1) < 1e-9
    assert abs(score(u, -u, m, None)["raw"] + 1) < 1e-9
    print("erase_baseline self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--bias-npy", default=None,
                    help="same file diagnose uses; enables the centered column")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.cache is None:
            ap.error("--cache is required (or pass --self-test)")
        bias = np.load(args.bias_npy).astype(np.float64) if args.bias_npy else None
        items = load_cache(args.cache)
        print(f"{len(items)} cached item(s) with a mask from {args.cache}")
        report(evaluate(items, bias, seed=args.seed))
