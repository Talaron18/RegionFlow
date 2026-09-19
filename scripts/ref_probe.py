"""Direct test: use another sample's z_tgt as z_ref, predict v_i.

This is what an actual z_ref conditioning path would provide -- not
z_tgt itself, but a different sample's target. If the cos is higher than
baseline (0.615), z_ref has information. If not, z_ref adds nothing
beyond what z_src + c already carry.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transport.data import TripletCache


def mean_pool_masked(v, m):
    denom = m.sum().clamp(min=1.0)
    return (v * m).sum(dim=(1, 2)) / denom


def main():
    cache = TripletCache("/data/cache/13061d51aee4")
    n = len(cache)

    # group by instruction
    by_instr = defaultdict(list)
    for i in range(n):
        rec = cache[i]
        instr = rec["meta"]["instruction"].strip().lower()
        by_instr[instr].append(i)

    # load pooled signatures
    v_pooled, z_src_pooled, z_tgt_pooled = {}, {}, {}
    for i in range(n):
        rec = cache[i]
        m = torch.from_numpy(rec["mask"]).float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        z_src = torch.from_numpy(rec["z_src"]).float()
        z_tgt = torch.from_numpy(rec["z_tgt"]).float()
        v_pooled[i] = mean_pool_masked(z_tgt - z_src, m)
        z_src_pooled[i] = mean_pool_masked(z_src, m)
        z_tgt_pooled[i] = mean_pool_masked(z_tgt, m)

    def cos(a, b):
        na, nb = a.norm(), b.norm()
        if na < 1e-8 or nb < 1e-8:
            return float("nan")
        return float((a * b).sum() / (na * nb))

    # the direct test: for each (i, j) same instruction, predict v_i
    # using z_ref = z_tgt_j
    rows = []
    for instr, idxs in by_instr.items():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(len(idxs)):
                if a == b:
                    continue
                i, j = idxs[a], idxs[b]
                # baseline: v_i vs v_j (what we already have)
                v_sim = cos(v_pooled[i], v_pooled[j])
                # z_ref test: predict v_i's direction from z_tgt_j - z_src_i
                # IMPORTANT: v_i = z_tgt_i - z_src_i, and z_ref_j - z_src_i
                # uses i's own source with j's target. This measures whether
                # a target hint helps direction.
                v_pred_ref = z_tgt_pooled[j] - z_src_pooled[i]
                v_ref_cos = cos(v_pooled[i], v_pred_ref)
                rows.append({
                    "i": i, "j": j,
                    "v_sim": v_sim,
                    "v_ref_cos": v_ref_cos,
                })

    v_arr = np.array([r["v_sim"] for r in rows])
    ref_arr = np.array([r["v_ref_cos"] for r in rows])
    valid = ~(np.isnan(v_arr) | np.isnan(ref_arr))
    v_arr, ref_arr = v_arr[valid], ref_arr[valid]

    print(f"n = {len(v_arr)}")
    print(f"BASELINE (predict from same instruction only):")
    print(f"  v_sim: mean {v_arr.mean():+.4f}  ceiling {np.sqrt(max(v_arr.mean(), 0)):.4f}")
    print(f"\nWITH z_ref (other sample's z_tgt as target hint):")
    print(f"  v_ref_cos: mean {ref_arr.mean():+.4f}  "
          f"sqrt(mean) {np.sqrt(max(ref_arr.mean(), 0)):.4f}")
    print(f"\nDelta ceiling: "
          f"{np.sqrt(max(ref_arr.mean(), 0)) - np.sqrt(max(v_arr.mean(), 0)):+.4f}")


if __name__ == "__main__":
    main()