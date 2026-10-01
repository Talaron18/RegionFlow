#!/usr/bin/env bash
# Does a sampler that STARTS LOWER than t = 1 see signal?
# (H0_H0PRIME_SPEC.md Sec.1.9 items 7-8; this is fix (A), the free one.)
#
# WHY. The training t-marginal has a HOLE at the top -- measured from the
# sampler's own code by scripts/probe_delta_mass.py:
#     [0.8,0.9) 6.36%   [0.9,0.95) 0.86%   [0.95,0.99) 0.09%   [0.99,1] 0.00%
# and joint (t>=0.9 AND Delta>0.2) = 0.219%. Every sampler's FIRST step is at
# t ~ 1, and the two cells measured there are dead: cos +0.001 at (1.0,0.75) and
# +0.180 at (1.0,0.5) -- while (0.875,0.625) reads +0.642. This maps WHERE the
# signal dies, which is exactly what picks sigma_max. No training, no code change.
#
# PRE-REGISTERED PREDICTION (written before the run):
#   If the dead noise end is limited by TRAINING SUPPORT, cos should stay high up
#   to about t = 0.90-0.925 and collapse above ~0.95.
#   If it collapses EARLIER (already by t = 0.90), support is NOT the explanation
#   and the noise end is a capacity / conditioning limit instead -- then fix (A)
#   is dead and only (B) (train at high t) or the layer-1/2 work remains.
#
#   bash scripts/t_ladder.sh                    # Delta=0.25, t = 0.900 ... 1.000
#   DELTA=0.5 bash scripts/t_ladder.sh          # the 2-step ladder
#   TS="0.85 0.875 0.9" bash scripts/t_ladder.sh
#
# Cheap: one model load per cell (~1 min), both checkpoints inside each load.
set -u

cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/t_ladder

CACHE="${CACHE:-/data/cache/13061d51aee4}"
PY="${PY:-python}"
DELTA="${DELTA:-0.25}"
TS="${TS:-0.900 0.925 0.950 0.975 1.000}"
CKPTS="${CKPTS:-/data/runs/curr_a0_control_s1/ckpt_final.pt runs/a4_md75/ckpt_final.pt}"

for name in $CKPTS; do
    [ -f "$name" ] || { echo "[MISS] $name"; exit 1; }
done

for T in $TS; do
    # eval_interval_field.py takes ONE --t-center; the cell (t, r) is recovered
    # from t_center = (t+r)/2 and delta = t - r, so tc = t - delta/2.
    TC=$($PY -c "print(f'{$T - $DELTA / 2:.4f}')")
    R=$($PY -c "print(f'{$T - $DELTA:.4f}')")
    out="runs/t_ladder/t${T}_d${DELTA}.json"
    if [ -f "$out" ]; then echo "[skip] $out"; continue; fi
    echo "[run ] t=$T  r=$R  t_center=$TC  delta=$DELTA -> $out"
    # shellcheck disable=SC2086
    $PY -u scripts/eval_interval_field.py --cache "$CACHE" --ckpt $CKPTS \
        --deltas "$DELTA" --t-center "$TC" --samples 32 --substeps 16 \
        --rank 32 --seed 0 --out "$out" > "${out%.json}.stdout" 2>&1 \
        || echo "[FAIL] t=$T -- see ${out%.json}.stdout"
done

$PY - <<'PY'
import glob, json

rows = []
for path in glob.glob("runs/t_ladder/t*_d*.json"):
    try:
        blob = json.load(open(path))
    except Exception as exc:                       # noqa: BLE001
        print(f"[warn] unreadable {path}: {exc}")
        continue
    cfg = blob["config"]
    d = float(cfg["deltas"][0])
    t = float(cfg["t_center"]) + d / 2.0
    for name, res in blob["results"].items():
        for r in res["rows"]:
            rows.append((t, t - d, name.split("/")[0], r))

if not rows:
    print("no runs/t_ladder/*.json yet -- nothing to tabulate")
    raise SystemExit(0)

rows.sort(key=lambda x: (-x[0], x[2]))
print(f"\n{'t':>6}{'r':>7}  {'ckpt':<24}{'m/b':>8}{'cos':>8}{'corr_err':>10}"
      f"{'D_r_ratio':>10}")
for t, r_, name, r in rows:
    print(f"{t:>6.3f}{r_:>7.3f}  {name:<24}{r.get('vs_base', float('nan')):>8.3f}"
          f"{r.get('cos_correction', float('nan')):>+8.3f}"
          f"{r.get('corr_err', float('nan')):>10.3f}"
          f"{r.get('D_r_ratio', float('nan')):>10.2f}")

print("\nWHERE THE SIGNAL DIES (this is the sigma_max argument):")
by = {}
for t, r_, name, r in rows:
    by.setdefault(name, []).append((t, r.get("cos_correction", 0.0),
                                    r.get("vs_base", float("nan"))))
for name, v in sorted(by.items()):
    v.sort()
    good = [t for t, c, m in v if c > 0.4]
    last = max(good) if good else None
    print(f"  {name:<24} highest t with cos > 0.4 : "
          + (f"{last:.3f}   <- a sampler starting at sigma_max ~ {last:.2f}"
             if last is not None else "NONE in the swept range"))
print("\nRead: if cos holds to ~0.90-0.925 and collapses above ~0.95, the dead")
print("noise end is a TRAINING-SUPPORT hole and fix (A) works (start the sampler")
print("at sigma_max ~ 0.9, no retraining). If it collapses by 0.90, support is")
print("not the explanation and the noise end is a capacity/conditioning limit.")
PY
