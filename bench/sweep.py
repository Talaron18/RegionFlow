"""
Stage-0 sweep driver.

Two sweeps in order, because running them together confounds:

  A. lambda pre-sweep at fixed r. Finds an operating point for (lam_in,
     lam_out). These are logit-space and interact with the guidance scale, so
     without this the r-sweep measures attention strength, not tiering.
  B. r-sweep at the chosen lambda. This is the tiering claim. r=0 is the
     binary-mask control, i.e. the published-prior equivalent.

Nothing here needs training. Both fields are priors, so Stage 0 runs on any
off-the-shelf inversion and either validates the idea or kills it before the
BiFM reimplementation is attempted.

The confound guard is the important part. Raising r slightly grows the count of
tokens at full lam_in, because smoothstep is near 1 just outside the boundary.
Field mass is logged per case so a tiering effect can be separated from a mild
dilation effect. Without it the sweep is uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from region.alpha_field import RetentionConfig, build_retention_field, paired_configs
from region.lambda_field import RegionFieldConfig, build_region_field

from .metrics import (
    CaseResult,
    background_change,
    colour_bleed,
    instance_leakage,
    masked_psnr,
)


R_GRID = (0.0, 1.0, 2.0, 4.0)
LAM_IN_GRID = (2.0, 4.0, 8.0)
LAM_OUT_GRID = (0.0, -2.0, -4.0)


@dataclass
class BenchCase:
    """One evaluation item.

    other_instance_masks is what makes the leakage metric possible and is not
    present in MAG-Bench or PIE-Bench as shipped. Annotating it for the
    multi-instance subset is a small job and is arguably a contribution in
    itself -- both benchmarks stratify by mask size, neither by instance
    multiplicity.
    """

    case_id: str
    image: np.ndarray
    source_prompt: str
    target_prompt: str
    target_mask: np.ndarray
    box: tuple | None = None
    other_instance_masks: list | None = None
    mask_bucket: str = "unknown"
    source_attribute: str | None = None
    target_attribute: str | None = None


def field_mass(field: np.ndarray, cfg: RegionFieldConfig) -> dict:
    """Confound guard. Log this for every case at every r."""
    core = float((field > cfg.lam_in * 0.999).sum())
    touched = float((field > cfg.lam_out + 1e-3).sum())
    # Total positive drive, the scale-free summary that should be reported
    # alongside r.
    mass = float(np.clip(field - cfg.lam_out, 0, None).sum())
    return {"core_tokens": core, "touched_tokens": touched, "field_mass": mass}


def evaluate_case(
    case: BenchCase,
    edited: np.ndarray,
    r: float,
    lam_in: float,
    lam_out: float,
    leak_threshold: float = 0.02,
) -> CaseResult:
    """Score one edited output."""
    others = case.other_instance_masks or []
    leak = instance_leakage(
        case.image, edited, case.target_mask, others, leak_threshold
    )
    return CaseResult(
        case_id=case.case_id,
        r=r,
        lam_in=lam_in,
        lam_out=lam_out,
        mask_bucket=case.mask_bucket,
        n_instances=1 + len(others),
        bg_change=background_change(case.image, edited, case.target_mask),
        bg_psnr=masked_psnr(case.image, edited, case.target_mask, inside=False),
        target_change=leak["target_change"],
        leak_rate=leak["leak_rate"],
        selectivity=leak["selectivity"],
        bleed=colour_bleed(case.image, edited, case.target_mask),
    )


def calibrate_leak_threshold(
    cases: Iterable[BenchCase], edit_fn: Callable, n: int = 20
) -> float:
    """Measure VAE round-trip noise on a no-op edit and set the threshold above it.

    Skipping this means measuring reconstruction error and reporting it as
    instance leakage. Run it once per backbone.
    """
    diffs = []
    for case in list(cases)[:n]:
        recon = edit_fn(case, field=None, tau=None)  # identity edit
        d = np.abs(
            np.asarray(case.image, np.float32) - np.asarray(recon, np.float32)
        )
        if d.max() > 1.5:
            d = d / 255.0
        diffs.append(float(d.mean()))
    noise = float(np.mean(diffs)) if diffs else 0.0
    return max(3.0 * noise, 0.005)


def run_lambda_presweep(
    cases, edit_fn, r: float = 2.0, leak_threshold: float = 0.02
) -> dict:
    """Sweep A. Returns {(lam_in, lam_out): summary} plus the chosen point.

    Selection rule: highest selectivity subject to background change staying
    within 10% of the best observed. Picking on selectivity alone rewards
    configurations that simply edit nothing.
    """
    from .metrics import aggregate

    results = []
    for lam_in in LAM_IN_GRID:
        for lam_out in LAM_OUT_GRID:
            cfg_l, cfg_t = paired_configs(r, lam_in, lam_out)
            for case in cases:
                field = build_region_field(case.target_mask, cfg_l)
                tau = build_retention_field(case.target_mask, cfg_t)
                edited = edit_fn(case, field=field, tau=tau)
                results.append(
                    evaluate_case(case, edited, r, lam_in, lam_out, leak_threshold)
                )

    summary = aggregate(results, by=("lam_in", "lam_out"))
    bg_best = min(
        (s["bg_change"] for s in summary.values() if np.isfinite(s["bg_change"])),
        default=float("inf"),
    )
    eligible = {
        k: s for k, s in summary.items() if s["bg_change"] <= bg_best * 1.10
    } or summary
    chosen = max(eligible, key=lambda k: eligible[k]["selectivity"])
    return {"summary": summary, "chosen": chosen, "results": results}


def run_r_sweep(
    cases, edit_fn, lam_in: float, lam_out: float, leak_threshold: float = 0.02
) -> dict:
    """Sweep B -- the tiering claim. r=0 is the binary-mask control."""
    from .metrics import aggregate

    results, guards = [], {}
    for r in R_GRID:
        cfg_l, cfg_t = paired_configs(r, lam_in, lam_out)
        masses = []
        for case in cases:
            field = build_region_field(case.target_mask, cfg_l)
            tau = build_retention_field(case.target_mask, cfg_t)
            masses.append(field_mass(field, cfg_l))
            edited = edit_fn(case, field=field, tau=tau)
            results.append(
                evaluate_case(case, edited, r, lam_in, lam_out, leak_threshold)
            )
        guards[r] = {
            k: float(np.mean([m[k] for m in masses])) for k in masses[0]
        }

    return {
        "by_r": aggregate(results, by=("r",)),
        "by_r_bucket": aggregate(results, by=("r", "mask_bucket")),
        "confound_guard": guards,
        "results": results,
    }


def report(sweep: dict) -> str:
    lines = ["r      n   bg_change  selectivity  leak_rate   bleed   core  mass"]
    for (r,), s in sorted(sweep["by_r"].items()):
        g = sweep["confound_guard"][r]
        lines.append(
            f"{r:<5} {s['n']:>3}  {s['bg_change']:>9.4f}  {s['selectivity']:>11.2f}"
            f"  {s['leak_rate']:>9.2f}  {s['bleed']:>6.3f}"
            f"  {g['core_tokens']:>5.0f} {g['field_mass']:>7.0f}"
        )
    lines.append("")
    lines.append("by mask bucket (small bucket has no room for a graded field):")
    for (r, bucket), s in sorted(sweep["by_r_bucket"].items(), key=lambda x: str(x[0])):
        lines.append(
            f"  r={r:<4} {bucket:<7} n={s['n']:<3} "
            f"sel={s['selectivity']:.2f} bg={s['bg_change']:.4f}"
        )
    return "\n".join(lines)
