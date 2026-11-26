"""Paired comparison of two arms' VLM-judged quality (edit_quality_vlm.py output).

    python scripts/compare_judged_arms.py \
        --treat runs/gate_c/vlm_quality_gate_a_dino.jsonl \
        --control runs/gate_c/vlm_quality_gate_a_ctrl.jsonl

Pairs on (id, prompt_kind), reports the median of `overall` per arm and an
exact two-sided sign test on the paired differences (ties dropped). The
decision rule it prints is the one pre-registered in DINO_EXPERIMENT.md.
"""
import argparse
import json
from math import comb
from statistics import median


def load(path: str, kind: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["prompt_kind"] == kind:
                    out[r["id"]] = float(r["overall"])
    return out


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value at p = 0.5."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(treat: dict, control: dict, min_gain: float, alpha: float) -> dict:
    ids = sorted(set(treat) & set(control))
    diffs = [treat[i] - control[i] for i in ids]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    p = sign_test(wins, losses)
    gain = (median([treat[i] for i in ids]) - median([control[i] for i in ids])
            if ids else 0.0)
    return {"n_paired": len(ids), "wins": wins, "losses": losses,
            "ties": len(ids) - wins - losses,
            "median_treat": median([treat[i] for i in ids]) if ids else None,
            "median_control": median([control[i] for i in ids]) if ids else None,
            "median_gain": gain, "sign_test_p": p,
            "pass": bool(wins > losses and p < alpha and gain >= min_gain)}


def self_test() -> None:
    assert abs(sign_test(5, 5) - 1.0) < 1e-12
    assert abs(sign_test(10, 0) - 2 / 2 ** 10) < 1e-12
    assert abs(sign_test(0, 10) - sign_test(10, 0)) < 1e-12
    a = {f"c{i}": 6.0 for i in range(40)}
    b = {f"c{i}": 4.0 for i in range(40)}
    assert compare(a, b, 1.0, 0.05)["pass"]
    assert not compare(b, a, 1.0, 0.05)["pass"]      # worse must not pass
    assert not compare(a, a, 1.0, 0.05)["pass"]      # identical must not pass
    print("compare_judged_arms self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treat")
    ap.add_argument("--control")
    ap.add_argument("--prompt-kind", default="full", choices=("full", "single"))
    ap.add_argument("--min-gain", type=float, default=1.0,
                    help="required gain in median overall (0-10 scale)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        res = compare(load(args.treat, args.prompt_kind),
                      load(args.control, args.prompt_kind),
                      args.min_gain, args.alpha)
        print(json.dumps(res, indent=1))
