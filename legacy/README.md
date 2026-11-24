# legacy/

Stale root-level duplicates of `bench/inversion.py` and `scripts/smoke_edit.py`
from an earlier layout. They do NOT contain the Stage 0 fixes (fp32 ODE state,
σ retracing, fp32 timestep). Kept for reference only.

Do not import from here. Use `bench/` and `scripts/`.

## Moved here 2026-11-24
One-off patches and helper scripts from the root: `*.patch`, `apply_throughput.py`,
`fix_loader.py`, `gate_b_meanflow.project_patched.py`, `train.project.py`,
`recompute_outcomes.py`, `scan_corruption.py`, `score_all_arms.sh`. Not imported by current
code. `ambiguity_probe.py`, `delta_curriculum.py`, `can.jpg` and `synthetic_*.png` stay at the
root because code imports or reads them there.
