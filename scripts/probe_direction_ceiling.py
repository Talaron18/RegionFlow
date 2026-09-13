"""
Is cos ~ 0.5 the learner's limit, or the data's?

THE RELATION THAT MAKES THIS TESTABLE. Suppose the target displacement for a
given conditioning decomposes as

    v = mu + e,        e independent noise, E[e] = 0

where mu is the conditional mean -- the best any deterministic MSE predictor
can output. Then for two independent draws under the same conditioning:

    E[cos(v_i, v_j)]  ~  ||mu||^2 / ||v||^2
    cos(mu, v)        ~  ||mu||   / ||v||

so the CEILING on the model's alignment is

    cos_max  ~  sqrt( E[cos(v_i, v_j)] )

That inverts into a concrete prediction. The run sits at cos ~ 0.5, so:

    pairwise ~ 0.25   the model is AT the statistical ceiling. More capacity,
                      more data and longer schedules cannot help; the target
                      direction is genuinely ambiguous given (z, c, M), and
                      a deterministic regression is the wrong object.
    pairwise ~ 0.5    ceiling ~ 0.71. Real headroom; the limit is the
                      architecture, the conditioning pathway or optimisation.
    pairwise ~ 0.8    ceiling ~ 0.89. Large headroom; the model is
                      substantially underperforming what the data supports.

WHAT DIRECTION MEANS HERE. Comparing raw v tensors across examples would
mostly measure whether their masks overlap spatially, which is not the
question. The object compared is the MASK-POOLED displacement: the mean of v
over the mask, giving one 16-dim vector per example in latent channel space.
That is exactly the "attribute direction" the Stage 1 hypothesis is about --
z + Delta z with Delta z conditioned on the edit -- and it is
location-invariant, so two "make it red" edits in different parts of different
images are comparable.

The spatial pattern within the mask is reported separately, for examples whose
masks do overlap, so the two are not conflated.

NO GPU, NO MODEL. Everything comes from the cached latents, so this runs while
training continues.

    python scripts/probe_direction_ceiling.py --cache /data/cache/<hash>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.data import TripletCache  # noqa: E402


def normalise_instruction(s: str) -> str:
    """Loose grouping key: lowercase, strip articles and punctuation.

    'change the green ball to blue' and 'Change green ball to blue.' should
    land in the same bucket. This is deliberately crude -- it defines the
    conditioning we are testing agreement UNDER, and a crude key gives a
    conservative (low) estimate of agreement, which biases against the
    "data is ambiguous" conclusion rather than toward it.
    """
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(the|a|an|please|make|let|it|be|to|into|this|that)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pooled_direction(z_src, z_tgt, mask, eps=1e-8):
    """(16,) unit vector: mean displacement inside the mask, channel space."""
    v = z_tgt - z_src                       # (16, h, w)
    m = mask[0] if mask.ndim == 3 else mask  # (h, w)
    sel = m > 0.5
    if sel.sum() < 4:
        return None, 0.0
    d = v[:, sel].mean(axis=1)              # (16,)
    n = float(np.linalg.norm(d))
    if n < eps:
        return None, 0.0
    return d / n, n


def pairwise_cos(vecs, rng, max_pairs=4000):
    """Mean cosine over random pairs from a group."""
    k = len(vecs)
    if k < 2:
        return None, 0
    V = np.stack(vecs)
    if k * (k - 1) // 2 <= max_pairs:
        G = V @ V.T
        iu = np.triu_indices(k, 1)
        vals = G[iu]
    else:
        i = rng.integers(0, k, max_pairs)
        j = rng.integers(0, k, max_pairs)
        keep = i != j
        vals = (V[i[keep]] * V[j[keep]]).sum(1)
    return float(vals.mean()), len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--min-group", type=int, default=4,
                    help="smallest instruction group to report")
    ap.add_argument("--model-cos", type=float, default=0.50,
                    help="the model's achieved cos, for the comparison")
    ap.add_argument("--knn", type=int, default=0,
                    help="group by conditioning-space neighbours instead of "
                         "exact instruction text: for each example, the K "
                         "nearest by pooled text embedding. Covers the whole "
                         "set rather than the ~1%% with repeated instructions.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = TripletCache(args.cache)
    print(f"cache {args.cache}: {len(cache)} triplets")

    dirs, norms, keys, ids, kept_idx = [], [], [], [], []
    for i in range(len(cache)):
        d = cache[i]
        u, n = pooled_direction(d["z_src"], d["z_tgt"], d["mask"])
        if u is None:
            continue
        dirs.append(u)
        kept_idx.append(i)
        norms.append(n)
        keys.append(normalise_instruction(d["meta"]["instruction"]))
        ids.append(d["meta"]["id"])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(cache)}")

    dirs = np.stack(dirs)
    norms = np.array(norms)
    rng = np.random.default_rng(0)
    print(f"\nusable: {len(dirs)}")

    # --- baseline: unconditional agreement --------------------------------
    uncond, n_uncond = pairwise_cos(list(dirs), rng)
    print(f"\nUNCONDITIONAL pairwise cos over {n_uncond} random pairs: "
          f"{uncond:+.3f}")
    print("  This is the floor. If it is already high, the displacements share")
    print("  a global direction (e.g. a VAE or dataset bias) and the")
    print("  conditional numbers below must be read against it, not against 0.")

    # --- conditional: k nearest neighbours in conditioning space ----------
    # Covers the whole set instead of the ~1% with repeated instruction text.
    knn_result = None
    if args.knn:
        # Index by KEPT items only. dirs skips examples whose mask was too
        # small or whose displacement was degenerate, so pooled[i] and dirs[i]
        # are not the same example unless nothing was skipped. Slicing the
        # first len(dirs) entries -- which the first version did -- silently
        # misaligns the two whenever anything is dropped.
        pooled = []
        for i in kept_idx:
            d = cache[i]
            if "pooled" not in d:
                raise KeyError(
                    f"item {i} has no pooled embedding. --knn needs the cache "
                    "to carry text embeddings; rebuild it with "
                    "precompute_cache.py, which writes them."
                )
            pooled.append(d["pooled"])

        P = np.stack(pooled)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        sub = rng.choice(len(P), size=min(1500, len(P)), replace=False)

        # Calibrate the similarity scale. Pooled CLIP embeddings are
        # anisotropic -- random pairs are not near 0 -- so a neighbour
        # similarity of 0.70 means nothing until the random baseline is known.
        ri = rng.integers(0, len(P), 4000)
        rj = rng.integers(0, len(P), 4000)
        keep = ri != rj
        rand_sim = float((P[ri[keep]] * P[rj[keep]]).sum(1).mean())

        # Sweep the similarity FLOOR. Tighter conditioning -> higher ceiling.
        # k=8 with no threshold mixes loosely-related instructions and
        # UNDERSTATES the ceiling; exact-text grouping reaches only ~1% of the
        # data and OVERSTATES it by selecting easy repeated edits. Thresholded
        # neighbours keep whole-set coverage while tightening conditioning, so
        # the trend as thr -> 1 is the honest estimate.
        floor0 = float(np.sqrt(max(uncond, 0.0)))
        sweep = []
        for thr in (0.0, 0.75, 0.85, 0.90, 0.95):
            vals, sims = [], []
            for i in sub:
                sim = P @ P[i]
                sim[i] = -2.0
                cand = np.where(sim >= thr)[0] if thr > 0 else \
                    np.argpartition(-sim, args.knn)[: args.knn]
                if thr > 0:
                    if len(cand) == 0:
                        continue
                    if len(cand) > args.knn:
                        cand = cand[np.argsort(-sim[cand])[: args.knn]]
                sims.extend(float(sim[j]) for j in cand)
                vals.extend(float(dirs[i] @ dirs[j]) for j in cand)
            if len(vals) < 200:
                sweep.append({"thr": thr, "n_pairs": len(vals),
                              "note": "too few pairs"})
                continue
            pc_ = float(np.mean(vals))
            ceil_ = float(np.sqrt(max(pc_, 0.0)))
            sweep.append({
                "thr": thr, "n_pairs": len(vals),
                "n_anchors_used": int(len(set())) or None,
                "mean_neighbour_sim": float(np.mean(sims)),
                "pairwise_cos": pc_, "ceiling": ceil_,
                "skill": (args.model_cos - floor0) / max(ceil_ - floor0, 1e-6),
            })

        knn_result = {"k": args.knn, "n_anchors": int(len(sub)),
                      "random_text_sim": rand_sim, "sweep": sweep}

        print(f"\n[knn] k<={args.knn}, {len(sub)} anchors")
        print(f"      random-pair text similarity {rand_sim:+.3f}  "
              "(the scale's zero point)")
        print(f"\n{'sim floor':>10}{'pairs':>8}{'mean sim':>10}"
              f"{'pair cos':>10}{'ceiling':>9}{'skill':>8}")
        for r in sweep:
            if "note" in r:
                print(f"{r['thr']:>10.2f}{r['n_pairs']:>8}   {r['note']}")
                continue
            print(f"{r['thr']:>10.2f}{r['n_pairs']:>8}"
                  f"{r['mean_neighbour_sim']:>10.3f}{r['pairwise_cos']:>10.3f}"
                  f"{r['ceiling']:>9.3f}{r['skill']:>8.2f}")
        print("\n      Read the TREND, not any single row. If the ceiling keeps")
        print("      climbing as the floor tightens, the loose-kNN number was")
        print("      an underestimate and the model has real headroom. If it")
        print("      flattens near the model's own cos, the model is at the")
        print("      ceiling and a deterministic objective cannot do better.")

    # --- conditional: same instruction ------------------------------------
    groups = defaultdict(list)
    for u, k in zip(dirs, keys):
        groups[k].append(u)
    big = {k: v for k, v in groups.items() if len(v) >= args.min_group}
    print(f"\ninstruction groups with >= {args.min_group} members: {len(big)} "
          f"covering {sum(len(v) for v in big.values())} examples")

    rows = []
    for k, v in sorted(big.items(), key=lambda kv: -len(kv[1])):
        c, npairs = pairwise_cos(v, rng)
        rows.append({"instruction": k, "n": len(v), "pairwise_cos": c})

    print(f"\n{'instruction':<40}{'n':>5}{'pair cos':>10}{'ceiling':>9}")
    for r in rows[:20]:
        ceil = np.sqrt(max(r["pairwise_cos"], 0.0))
        print(f"{r['instruction'][:39]:<40}{r['n']:>5}"
              f"{r['pairwise_cos']:>+10.3f}{ceil:>9.3f}")

    w = np.array([r["n"] for r in rows], dtype=float)
    cond = float(np.average([r["pairwise_cos"] for r in rows], weights=w))
    ceiling = float(np.sqrt(max(cond, 0.0)))

    # --- magnitude: how much is there to predict? -------------------------
    lo, hi = np.percentile(norms, [10, 90])
    print(f"\n||v|| spread: median {np.median(norms):.3f}, "
          f"p10-p90 {lo:.3f}-{hi:.3f}, cv {norms.std()/norms.mean():.2f}")
    print("  A high cv means per-example magnitude IS predictable in "
          "principle; corr(||v_hat||, ||v||) across examples would be worth "
          "logging. A low cv means there is little magnitude signal to learn "
          "and the amplitude question is settled by the cos relation alone.")

    # --- verdict ----------------------------------------------------------
    # SUBTRACT THE FLOOR. The unconditional pairwise agreement is not zero, so
    # a model that emits ONE FIXED DIRECTION for every input already scores
    # sqrt(unconditional). Comparing the model against the ceiling alone
    # credits it for that free baseline. The first version of this script did
    # exactly that and reported "71% of ceiling" for a model that had captured
    # 21% of the learnable signal.
    floor = float(np.sqrt(max(uncond, 0.0)))
    skill = (args.model_cos - floor) / max(ceiling - floor, 1e-6)

    print("\n" + "=" * 62)
    print(f"unconditional pairwise    {uncond:+.3f}"
          f"   -> constant-direction baseline cos = {floor:.3f}")
    print(f"conditional pairwise      {cond:+.3f}"
          f"   -> ceiling cos                     = {ceiling:.3f}")
    print(f"model achieved                          "
          f"                          = {args.model_cos:.3f}")
    print(f"\n  fraction of ceiling (MISLEADING)      = "
          f"{args.model_cos / max(ceiling, 1e-6):.2f}")
    print(f"  SKILL above the free baseline         = {skill:.2f}")
    print("=" * 62)

    n_grouped = sum(r["n"] for r in rows)
    print(f"\nSAMPLE: the conditional figure rests on {n_grouped} examples in "
          f"{len(rows)} groups,\n  out of {len(dirs)} "
          f"({100*n_grouped/max(len(dirs),1):.1f}%). Group values range "
          f"{min(r['pairwise_cos'] for r in rows):+.2f} to "
          f"{max(r['pairwise_cos'] for r in rows):+.2f}.")
    print("  Instructions repeated often enough to group are the COMMON, EASY\n"
          "  edits (recolours, removals). The ceiling is estimated on those\n"
          "  while the model's cos is measured across everything, so the\n"
          "  comparison is not like-for-like and flatters the ceiling. Use\n"
          "  --knn to group by conditioning-space neighbours instead, which\n"
          "  covers the whole set.")

    if skill >= 0.7:
        print("\nREADING: the model captures most of the learnable directional\n"
              "signal. Remaining error is close to irreducible under a\n"
              "deterministic objective.")
    elif skill >= 0.35:
        print("\nREADING: meaningful but partial. An architecture ablation\n"
              "(rank, injection sites, conditioning pathway) is worth one run\n"
              "before any claim about the latent space.")
    else:
        print("\nREADING: the model is barely above a CONSTANT-DIRECTION\n"
              "predictor. Most of its apparent alignment needs no conditioning\n"
              "at all. Before concluding anything about SD3's latent geometry,\n"
              "establish that the instruction is reaching the displacement:\n"
              "  - train with cond_dropout=1.0 (instruction fully removed). If\n"
              "    cos barely drops, the conditioning pathway is not being\n"
              "    used and that is the bug, not the latent space.\n"
              "  - compare against a fixed-direction baseline explicitly.\n"
              "This is a LEARNER-side diagnosis, and it is testable cheaply.")

    print("\nCAVEAT: grouping is by instruction text alone, so this conditions\n"
          "on c but not on (z, M). Conditioning on more can only INCREASE\n"
          "agreement, so the ceiling reported here is a LOWER bound. If the\n"
          "model already exceeds it, that is not a contradiction -- it means\n"
          "the model is using source and mask information this grouping\n"
          "ignores, and the test needs conditioning-space neighbours rather\n"
          "than exact instruction matches.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "unconditional_pairwise_cos": uncond,
            "conditional_pairwise_cos": cond,
            "implied_ceiling": ceiling,
            "model_cos": args.model_cos,
            "constant_direction_baseline": float(np.sqrt(max(uncond, 0.0))),
            "skill_above_baseline": skill,
            "n_grouped_examples": sum(r["n"] for r in rows),
            "knn": knn_result,
            "norm_stats": {"median": float(np.median(norms)),
                           "cv": float(norms.std() / norms.mean())},
            "groups": rows,
        }, indent=1))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
