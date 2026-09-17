"""
Target ambiguity probe, GATE_A_REOPENING_PLAN.md SS7 "The corrected
ambiguity probe" -- the one thing both the proposal and its critique agree
should run before any Gate A retrain. numpy over the existing triplet
cache (transport/data.py), no GPU.

WHAT THIS TESTS, AND WHY THIS FORM NOT THE ORIGINAL ONE. The original probe
idea -- cos(v_i, v_j) across samples "sharing similar (z,c,M)" -- is
confounded: a low cosine could mean genuine mode ambiguity (same intent,
different valid edit) or just that the two images are different. The cache
has no fixed (z,c,M) with multiple recorded targets, so p(v|z,c,M) isn't
directly measurable that way.

What IS measurable is a contrast between two groupings of v = z_tgt - z_src:

  same_instruction  -- same instruction string, different source images.
                       If the instruction pinned down the edit, these
                       should be reasonably aligned regardless of the
                       source (a "make it black and white" displacement
                       should point a similar direction in latent space
                       across different photos).
  same_source       -- same source image, different instructions. This is
                       the null/control: different instructions SHOULD
                       produce different directions. Low cosine here is
                       expected and uninformative on its own.

THE ACTUAL CLAIM THIS TESTS. Gate A's failure story assumes the
conditioning (z, c, M) underdetermines the edit -- "turn the dog brown"
doesn't pin down WHICH brown. If that's right, same_instruction cosine
should still be noticeably higher than same_source cosine (the instruction
carries real directional information, just not complete information). If
same_instruction and same_source come out comparable, the instruction is
carrying close to NO directional information at all, which is a different
and more severe finding than "some residual ambiguity" -- it would mean
conditioning itself, not merely mode multiplicity, is the primary problem.

USAGE
    python ambiguity_probe.py --self-test
    python ambiguity_probe.py --cache /path/to/cache/<hash>
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def build_groups(ids: list[str], instruction_of: dict, source_of: dict):
    """Group ids by instruction string and by source-image id.

    source_of[id] is the part of the cache item id before "_t<turn>"
    (scripts/precompute_cache.py: item_id = f"{row['img_id']}_t{turn}").
    Turn is always 1 in this cache (docstring: "turn_index == 1 only"), so
    this recovers the underlying MagicBrush image id.
    """
    by_instruction = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for i in ids:
        by_instruction[instruction_of[i]].append(i)
        by_source[source_of[i]].append(i)
    return by_instruction, by_source


def sample_pairs(groups: dict, vecs: dict, rng: np.random.Generator,
                 max_pairs: int, label: str) -> np.ndarray:
    pairs = []
    for members in groups.values():
        if len(members) < 2:
            continue
        pairs.extend(itertools.combinations(members, 2))
    n_groups = sum(1 for m in groups.values() if len(m) >= 2)
    print(f"  {label}: {len(pairs)} candidate pairs across {n_groups} "
          f"groups with >=2 members")
    if not pairs:
        return np.array([])
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    return np.array([cosine(vecs[a], vecs[b]) for a, b in pairs])


def sample_unrelated_pairs(ids: list[str], instruction_of: dict,
                           vecs: dict, rng: np.random.Generator,
                           n_pairs: int) -> np.ndarray:
    """Null baseline: random pairs with DIFFERENT instructions (source may
    or may not match -- with same-source empty in practice this is moot,
    but excluded explicitly anyway so the null can't accidentally include
    a same-source pair the main run couldn't find).

    Added because the same-source contrast this probe was designed around
    can come back empty (MagicBrush: at most one turn-1 edit per source
    image in this cache) -- without SOME null, the same-instruction cosine
    has nothing to be compared against and can't be interpreted.
    """
    n = len(ids)
    got, tries, max_tries = [], 0, n_pairs * 50
    while len(got) < n_pairs and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j:
            continue
        a, b = ids[i], ids[j]
        if instruction_of[a] == instruction_of[b]:
            continue
        got.append((a, b))
    print(f"  unrelated (different instruction, random source): "
          f"{len(got)} pairs sampled")
    return np.array([cosine(vecs[a], vecs[b]) for a, b in got])


def report(name: str, arr: np.ndarray) -> None:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        print(f"  {name}: no pairs (nothing to report)")
        return
    print(f"  {name}: n={arr.size}  mean {arr.mean():+.3f}  "
          f"median {np.median(arr):+.3f}  "
          f"p10 {np.percentile(arr, 10):+.3f}  "
          f"p90 {np.percentile(arr, 90):+.3f}")


def run(cache_root: Path, max_pairs: int, seed: int) -> None:
    meta = json.loads((cache_root / "meta.json").read_text())
    items = meta["items"]
    print(f"{len(items)} cached triplets in {cache_root}")

    vecs, instruction_of, source_of = {}, {}, {}
    for rec in items:
        d = np.load(cache_root / f"{rec['stem']}.npz")
        vecs[rec["id"]] = (d["z_tgt"].astype(np.float64)
                           - d["z_src"].astype(np.float64))
        instruction_of[rec["id"]] = rec["instruction"]
        source_of[rec["id"]] = rec["id"].rsplit("_t", 1)[0]

    ids = list(vecs)
    by_instruction, by_source = build_groups(ids, instruction_of, source_of)
    rng = np.random.default_rng(seed)

    print("\npair availability:")
    same_instr = sample_pairs(by_instruction, vecs, rng, max_pairs,
                              "same instruction, different source")
    same_src = sample_pairs(by_source, vecs, rng, max_pairs,
                            "same source, different instruction")

    print("\ncos(v_i, v_j):")
    report("same-instruction", same_instr)
    report("same-source     ", same_src)

    if same_src.size == 0:
        print("\nno same-source pairs at all: this MagicBrush cache has at "
              "most one turn-1 edit per source image, so the same-source "
              "contrast can't be run against it as built. Falling back to "
              "an unrelated-pairs null (different instruction, random "
              "source) so same-instruction has SOMETHING to be compared "
              "against.")
        unrelated = sample_unrelated_pairs(
            ids, instruction_of, vecs, rng, min(max_pairs, len(same_instr) or max_pairs))
        report("unrelated       ", unrelated)
        null = unrelated
        null_label = "unrelated"
    else:
        null = same_src
        null_label = "same-source"

    if same_instr.size and null.size:
        gap = np.median(same_instr[~np.isnan(same_instr)]) - \
              np.median(null[~np.isnan(null)])
        print(f"\nmedian gap (same-instruction minus {null_label}): {gap:+.3f}")
        try:
            from scipy import stats  # noqa: PLC0415
            u, p = stats.mannwhitneyu(
                same_instr[~np.isnan(same_instr)],
                null[~np.isnan(null)])
            print(f"Mann-Whitney U: U={u:.0f}  p={p:.4g}")
        except ImportError:
            print("(scipy not installed -- compare the medians above "
                  "directly, or install scipy for a significance test)")


# --------------------------------------------------------------- self-test --

def self_test() -> None:
    assert abs(cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-9
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9
    assert abs(cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) + 1.0) < 1e-9
    assert np.isnan(cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])))

    # Synthetic cache: 2 instructions x 3 sources, plus one source with two
    # different instructions (so same_source has something to group).
    # v aligned within an instruction (small perpendicular jitter), and
    # orthogonal-ish across instructions -- a controlled case where the
    # ambiguity story is TRUE, to check the probe recovers that shape.
    rng = np.random.default_rng(0)
    instr_dir = {"A": np.array([1.0, 0.0, 0.0]), "B": np.array([0.0, 1.0, 0.0])}
    vecs, instruction_of, source_of = {}, {}, {}
    idx = 0
    specs = [("src0", "A"), ("src1", "A"), ("src2", "A"),
             ("src0", "B"), ("src3", "B"), ("src4", "B")]
    for src, instr in specs:
        jitter = rng.normal(scale=0.05, size=3)
        v = instr_dir[instr] + jitter
        item_id = f"{src}_t1_{idx}"
        idx += 1
        vecs[item_id] = v
        instruction_of[item_id] = instr
        source_of[item_id] = src

    ids = list(vecs)
    by_instruction, by_source = build_groups(ids, instruction_of, source_of)
    assert len(by_instruction["A"]) == 3
    assert len(by_instruction["B"]) == 3
    # src0 appears under both A and B -> the only same-source group with 2+.
    assert len(by_source["src0"]) == 2
    assert len(by_source["src1"]) == 1

    rng2 = np.random.default_rng(1)
    same_instr = sample_pairs(by_instruction, vecs, rng2, 1000, "test")
    same_src = sample_pairs(by_source, vecs, rng2, 1000, "test")
    assert same_instr.size == 3 + 3  # C(3,2) within A, C(3,2) within B
    assert same_src.size == 1        # only src0 has 2 members
    assert same_instr.mean() > 0.9, same_instr.mean()   # near-aligned by construction
    assert same_src.mean() < 0.3, same_src.mean()        # near-orthogonal (A vs B dirs)

    unrelated = sample_unrelated_pairs(ids, instruction_of, vecs, rng2, 100)
    # every unrelated pair by construction spans A-vs-B (only 2 instructions
    # exist), so it should look like same_src: near-orthogonal.
    assert unrelated.size > 0
    assert unrelated.mean() < 0.3, unrelated.mean()

    print("ambiguity_probe self-test OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--max-pairs", type=int, default=5000,
                   help="cap on sampled pairs per group, for cost")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.cache is None:
        p.error("--cache is required (or pass --self-test)")
    run(args.cache, args.max_pairs, args.seed)


if __name__ == "__main__":
    main()
