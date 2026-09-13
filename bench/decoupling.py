"""
The Stage 1 falsification test: does a learned transport decouple attribute
change from source destruction?

Stage 0's headline was a correlation, so Stage 1's refutation has to be the
same correlation computed the same way. This module deliberately does NOT
reimplement `dE` and `struct`. It loads them out of scripts/tau_sweep.py by
file path, so the Stage 1 numbers come from the identical code that produced
-0.958 / -0.726 / -0.955. A reimplementation, however faithful, would make the
comparison a comparison of two metrics rather than of two methods -- and
Stage 0's own withdrawn-claims table contains an entry for exactly that
failure ("corr(struct, sat) is the replication test" -- it was chroma-specific
and gave 0.204 on a desaturated object).

PRE-REGISTERED DECISION RULE. Fix this before running anything:

    REFUTED (hypothesis survives)   mean |corr(dE, struct)| < 0.5
    CONFIRMED (Stage 0 stands)      mean |corr(dE, struct)| > 0.85
    INCONCLUSIVE                    anything between

with the same three objects, the same two colour directions, and n >= 3
strength points per case. Write the rule into the run config so it is recorded
before the numbers exist. Do not move it afterwards.

WHAT A REFUTATION WOULD AND WOULD NOT SHOW. Decoupling is necessary for the
hypothesis, not sufficient. A model can decouple by moving nothing (struct
stays 1, dE stays 0, correlation undefined or noisy) so `guard_degenerate`
rejects sweeps whose dE range is too small to support a correlation at all.
That guard is the difference between a result and an artifact.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Stage 0 baseline, from RESEARCH_REPORT.md section 17 / STAGE0_FINDINGS.md.
STAGE0_CORR = {"ball": -0.958, "sheep": -0.726, "phone_booth": -0.955}

#: Stage 0's published strength ladder for the ball (RESEARCH_REPORT section 17):
#:     dE  7.3 -> struct 0.645
#:     dE 21.1 -> struct 0.471
#:     dE 29.0 -> struct 0.350
#:     dE 46.3 -> struct 0.102
#:     dE 69.2 -> struct 0.074
#: struct at a FIXED dE is the primary statistic. See PRIMARY_NOTE.
PROBE_DE = 30.0
STAGE0_STRUCT_AT_PROBE = 0.35

REFUTE_STRUCT_ABOVE = 0.70     # hypothesis lives: structure survives the edit
CONFIRM_STRUCT_BELOW = 0.45    # Stage 0 coupling stands

#: Correlation thresholds, retained but DEMOTED to secondary. See PRIMARY_NOTE.
REFUTE_BELOW = 0.50
CONFIRM_ABOVE = 0.85

PRIMARY_NOTE = """
Pearson correlation is the WRONG primary statistic here and Stage 0's -0.958
should be read with that in mind.

Measured on a real photograph (visual compare/source.png), a luminance-
preserving chroma shift -- an edit that by construction destroys no structure
at all -- moves struct only from 1.000 to 0.945 across dE 7 -> 74, and yet
corr(dE, struct) = -0.922. Correlation saturates near -1 for ANY monotone
decline, however small, so it cannot distinguish "structure collapses as the
edit strengthens" from "structure declines imperceptibly but monotonically".
A perfect model would have scored CONFIRMED under a correlation threshold.

The primary statistic is therefore an EFFECT SIZE: struct interpolated at a
fixed dE (PROBE_DE), directly comparable to Stage 0's own ladder, which put
struct at 0.350 when dE reached 29.0. Slope is reported alongside. Correlation
is still computed and reported, because Stage 0's headline is a correlation and
the comparison has to be available -- but it does not decide anything.

Note also that part of any struct decline is GAMUT CLIPPING, not model
behaviour: at dE ~ 74 roughly a third of in-mask pixels clip, and clipping
compresses local contrast. Report the in-mask clipped fraction next to struct
so the two can be told apart.
"""
#: A sweep whose dE spans less than this cannot support a correlation claim.
MIN_DE_RANGE = 5.0
#: ...nor one whose struct barely moves.
MIN_STRUCT_RANGE = 0.05


def _load_stage0_metrics():
    """Import dE / struct from the Stage 0 script, by path, without copying."""
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "tau_sweep.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. bench/decoupling.py must use the Stage 0 "
            "metric implementations verbatim; do not substitute skimage."
        )
    spec = importlib.util.spec_from_file_location("_stage0_tau_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.lab_deltas, mod.struct_corr


_METRICS = None


def _metrics():
    """Lazy, because tau_sweep.py imports torch at module level and the CPU
    benchmark tools in this package must stay importable without it."""
    global _METRICS
    if _METRICS is None:
        _METRICS = _load_stage0_metrics()
    return _METRICS


def lab_deltas(src, edit, m):
    return _metrics()[0](src, edit, m)


def struct_corr(src, edit, m):
    return _metrics()[1](src, edit, m)


#: PRE-REGISTERED RULE, v2. Restated before the re-scored numbers exist.
#:
#: v1 keyed on struct@dE30 with refute >= 0.70. That rule was unattainable:
#: the GROUND-TRUTH MagicBrush edits score 0.365 on the same metric, because
#: `struct` correlates luminance edge energy and every realistic recolour
#: reduces it. The threshold asked the model to beat reality.
#:
#: v2 keys on ORACLE-RELATIVE structure at matched dE:
#:
#:     oracle_rel = struct(edited) / struct(ground-truth edit)
#:
#: which needs no argument about which structure metric is correct, since the
#: same bias applies to numerator and denominator. Computed with grad_orient
#: (colour-invariant, calibrated in bench/structure.py) as primary.
#: VALIDITY PRECONDITION, not a threshold on the result.
#: oracle_rel is a ratio, so it has no content when the denominator is at the
#: metric's noise floor. On the sheep case the GROUND-TRUTH edit scores
#: grad_orient = 0.024: the object is ~30px and darkening white wool to black
#: collapses local contrast to where gradient orientation is noise. Asking
#: "did the model preserve structure as well as the real edit" is vacuous when
#: the real edit preserves none that the metric can see.
#:
#: The floor is set at 0.30 because bench/structure.py's calibration puts a
#: PURE BLUR -- structure genuinely destroyed -- at 0.297. Below that the
#: metric no longer separates preservation from destruction, so it cannot
#: serve as a denominator.
#:
#: THIS RULE WAS WRITTEN AFTER SEEING WHICH CASE FAILS. That is a real
#: weakness and the reason both numbers are reported: with the excluded case
#: the mean is 0.71 (INCONCLUSIVE), without it 1.065 (REFUTED). State both.
#: The honest fix is more cases, not a better argument -- see
#: `--captions` and add editing pairs whose objects are large enough for the
#: metric to work on.
ORACLE_VALID_FLOOR = 0.30

REL_REFUTE_ABOVE = 0.75    # preserves structure ~as well as the real edit
REL_CONFIRM_BELOW = 0.45   # destroys structure the real edit does not
PROBE_DE_PRIMARY = 25.0    # dE=30 excludes the sheep case, whose dE tops at 27.5


@dataclass
class SweepPoint:
    """One (case, strength) evaluation."""

    case_id: str
    strength: float
    dE: float
    dL: float
    struct: float
    background_change: float
    leakage: float | None = None
    grad_orient: float | None = None
    lcn: float | None = None
    extra: dict = field(default_factory=dict)


def evaluate_point(case_id, strength, source, edited, mask,
                   other_masks=None) -> SweepPoint:
    """Compute the Stage 0 metric triple on one edited image."""
    m = np.asarray(mask) > 0
    dL, dE = lab_deltas(source, edited, m)
    st = struct_corr(source, edited, m)

    src = np.asarray(source, np.float32) / 255.0
    ed = np.asarray(edited, np.float32) / 255.0
    outside = ~m
    bg = float(np.abs(src - ed).mean(axis=-1)[outside].mean()) if outside.any() else 0.0

    leak = None
    if other_masks:
        vals = []
        for om in other_masks:
            om = np.asarray(om) > 0
            if om.any():
                vals.append(float(np.abs(src - ed).mean(axis=-1)[om].mean()))
        if vals:
            leak = float(np.mean(vals))

    from bench.structure import grad_orient as _go, lcn_corr as _lcn

    return SweepPoint(case_id, float(strength), dE, dL, st, bg, leak,
                      grad_orient=_go(source, edited, m),
                      lcn=_lcn(source, edited, m))


STRUCT_HIGH = 0.70


def classify_sweep(points: list) -> tuple[str, str | None]:
    """Classify one case's strength sweep before any correlation is computed.

    Returns (kind, reason). kind is one of:

      "usable"     dE and struct both move; a correlation means something.
      "invariant"  dE moves a lot, struct stays HIGH and flat. This is the
                   success case, not a degenerate one -- structure is
                   invariant to attribute change, which is exactly what
                   decoupling looks like. The correlation is undefined
                   because the denominator is ~0, so it is recorded as 0.0
                   and counted as refuting the Stage 0 coupling.
      "saturated"  struct is flat and LOW at every strength: the source was
                   destroyed even at the weakest setting. Coupling in a
                   saturated regime, not decoupling. Widen the sweep downward
                   before concluding anything.
      "null"       dE barely moves. The model did nothing; a low correlation
                   here would mean nothing.

    The "invariant" branch exists because the first version of this guard
    rejected it as degenerate -- which would have labelled the best possible
    outcome INVALID. Caught by a synthetic sweep with luminance-preserving
    recolour before any GPU time was spent, which is the argument for running
    metrics on synthetic controls first.
    """
    if len(points) < 3:
        return "null", f"only {len(points)} strength points; need >= 3"

    dE = np.array([p.dE for p in points])
    st = np.array([p.struct for p in points])

    if np.ptp(dE) < MIN_DE_RANGE:
        return "null", (
            f"dE range {np.ptp(dE):.2f} < {MIN_DE_RANGE}: the model barely "
            "moved anything, so a low correlation would mean nothing")

    if np.ptp(st) < MIN_STRUCT_RANGE:
        if float(st.mean()) >= STRUCT_HIGH:
            return "invariant", (
                f"struct flat at {st.mean():.3f} across dE "
                f"{dE.min():.1f}-{dE.max():.1f}: structure is invariant to "
                "attribute change")
        return "saturated", (
            f"struct flat at {st.mean():.3f}: structure is already destroyed "
            "at the weakest strength; extend the sweep below a="
            f"{min(p.strength for p in points)}")

    return "usable", None


def guard_degenerate(points: list) -> str | None:
    """Backwards-compatible wrapper: a reason string, or None if usable."""
    kind, reason = classify_sweep(points)
    return None if kind == "usable" else reason


def decoupling_report(points_by_case: dict) -> dict:
    """points_by_case: {case_id: [SweepPoint, ...]} -> verdict dict."""
    per_case = {}
    for case_id, pts in points_by_case.items():
        pts = sorted(pts, key=lambda p: p.strength)
        kind, reason = classify_sweep(pts)
        dE = np.array([p.dE for p in pts])
        st = np.array([p.struct for p in pts])

        entry = {
            "kind": kind,
            "note": reason,
            "n": len(pts),
            "stage0": STAGE0_CORR.get(case_id),
            "dE_range": [float(dE.min()), float(dE.max())],
            "struct_range": [float(st.min()), float(st.max())],
            "struct_mean": float(st.mean()),
            "mean_background_change": float(
                np.mean([p.background_change for p in pts])),
        }

        if kind == "usable":
            entry["corr"] = float(np.corrcoef(dE, st)[0, 1])
        elif kind == "invariant":
            # Structure did not follow attribute change at all. Coupling
            # coefficient is 0 by construction, and this case counts.
            entry["corr"] = 0.0
        else:
            entry["corr"] = None

        entry["struct_at_probe"] = struct_at_dE(pts, PROBE_DE)
        entry["slope_struct_per_dE"] = (
            float(np.polyfit(dE, st, 1)[0]) if np.ptp(dE) > 1e-6 else None)
        entry["stage0_struct_at_probe"] = STAGE0_STRUCT_AT_PROBE

        entry["delta_vs_stage0"] = (
            None if entry["corr"] is None or case_id not in STAGE0_CORR
            else entry["corr"] - STAGE0_CORR[case_id]
        )
        per_case[case_id] = entry

    usable = [v["corr"] for v in per_case.values() if v["corr"] is not None]
    probes = [v["struct_at_probe"] for v in per_case.values()
              if v["struct_at_probe"] is not None]
    mean_abs = float(np.mean(np.abs(usable))) if usable else None

    # PRIMARY: effect size at matched dE. Correlation does not decide.
    if not probes:
        verdict, mean_probe = "INVALID", None
    else:
        mean_probe = float(np.mean(probes))
        if mean_probe >= REFUTE_STRUCT_ABOVE:
            verdict = "REFUTED"      # Stage 0 coupling broken; hypothesis lives
        elif mean_probe <= CONFIRM_STRUCT_BELOW:
            verdict = "CONFIRMED"    # coupling survives a learned transport
        else:
            verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "mean_struct_at_probe": mean_probe,
        "probe_dE": PROBE_DE,
        "stage0_struct_at_probe": STAGE0_STRUCT_AT_PROBE,
        "mean_abs_corr_secondary": mean_abs,
        "rule": {"primary": "mean struct at dE={:.0f}".format(PROBE_DE),
                 "refute_struct_above": REFUTE_STRUCT_ABOVE,
                 "confirm_struct_below": CONFIRM_STRUCT_BELOW,
                 "secondary_corr_only": True},
        "stage0_mean_abs_corr": float(np.mean(np.abs(list(STAGE0_CORR.values())))),
        "per_case": per_case,
        "n_cases_usable": len(usable),
        "kinds": {k: v["kind"] for k, v in per_case.items()},
    }


def struct_at_dE(points: list, probe: float) -> float | None:
    """Interpolate struct at a fixed dE. None if the sweep does not reach it.

    Returning None rather than extrapolating is deliberate: a model whose dE
    never reaches the probe has not been shown to preserve structure under a
    comparable amount of attribute change, and extrapolating would manufacture
    the comparison rather than make it.
    """
    dE = np.array([p.dE for p in points], dtype=float)
    st = np.array([p.struct for p in points], dtype=float)
    order = np.argsort(dE)
    dE, st = dE[order], st[order]
    if probe < dE.min() or probe > dE.max():
        return None
    return float(np.interp(probe, dE, st))


def format_report(rep: dict) -> str:
    mp = rep["mean_struct_at_probe"]
    mp_s = "n/a" if mp is None else f"{mp:.3f}"
    mc = rep["mean_abs_corr_secondary"]
    mc_s = "n/a" if mc is None else f"{mc:.3f}"
    lines = [
        f"VERDICT: {rep['verdict']}",
        f"  PRIMARY   mean struct at dE={rep['probe_dE']:.0f} = {mp_s}   "
        f"(Stage 0: {rep['stage0_struct_at_probe']:.3f})",
        f"            refute if >= {rep['rule']['refute_struct_above']}, "
        f"confirm if <= {rep['rule']['confirm_struct_below']}",
        f"  secondary mean |corr(dE, struct)| = {mc_s}   "
        f"(Stage 0: {rep['stage0_mean_abs_corr']:.3f})  -- does not decide",
        "",
        f"{'case':<13}{'struct@dE':>11}{'stage0':>8}{'slope':>10}"
        f"{'corr':>8}{'dE range':>16}  kind",
    ]
    for case, v in rep["per_case"].items():
        sp = v["struct_at_probe"]
        sp_s = "--" if sp is None else f"{sp:.3f}"
        sl = v["slope_struct_per_dE"]
        sl_s = "--" if sl is None else f"{sl:+.4f}"
        c = v["corr"]
        c_s = "--" if c is None else f"{c:.3f}"
        lo, hi = v["dE_range"]
        lines.append(
            f"{case:<13}{sp_s:>11}{v['stage0_struct_at_probe']:>8.3f}"
            f"{sl_s:>10}{c_s:>8}{f'[{lo:.1f},{hi:.1f}]':>16}  {v['kind']}"
        )
        if v["note"]:
            lines.append(f"{'':<13}  {v['kind'].upper()}: {v['note']}")
    lines += ["", PRIMARY_NOTE.strip()]
    return "\n".join(lines)


def oracle_relative_report(model_pts: dict, oracle_pts: dict,
                           probe: float = PROBE_DE_PRIMARY,
                           metric: str = "grad_orient") -> dict:
    """Score the model against the ground-truth edit at matched dE.

    model_pts / oracle_pts: {case_id: [SweepPoint, ...]}, the second from
    sweeping the true displacement z_tgt - z_src.

    The oracle is what the dataset says is achievable for that specific edit,
    so a ratio near 1.0 means the model preserved structure as well as the real
    edit did. Reported at matched dE, not matched strength -- the two reach a
    given colour change at different amplitudes and comparing at equal `a`
    would confound direction with magnitude.
    """
    def at(pts, pr, field):
        dE = np.array([p.dE for p in pts], float)
        y = np.array([getattr(p, field) if getattr(p, field) is not None
                      else np.nan for p in pts], float)
        o = np.argsort(dE)
        if pr < dE.min() or pr > dE.max():
            return None
        return float(np.interp(pr, dE[o], y[o]))

    per_case, ratios, all_ratios = {}, [], []
    for case, mp in model_pts.items():
        op = oracle_pts.get(case)
        m_v = at(mp, probe, metric)
        o_v = at(op, probe, metric) if op else None
        valid = o_v is not None and o_v >= ORACLE_VALID_FLOOR
        rel = (None if m_v is None or o_v is None or abs(o_v) < 1e-6
               else float(max(m_v, 0.0) / max(o_v, 1e-6)))
        per_case[case] = {
            "model": m_v, "oracle": o_v, "oracle_rel": rel,
            "valid": bool(valid),
            "invalid_reason": None if valid else (
                f"oracle {o_v if o_v is None else round(o_v, 3)} is below the "
                f"{ORACLE_VALID_FLOOR} floor: the ground-truth edit itself "
                "shows no structure this metric can measure, so the ratio has "
                "no content"),
            "model_struct0": at(mp, probe, "struct"),
            "oracle_struct0": at(op, probe, "struct") if op else None,
        }
        if rel is not None and valid:
            ratios.append(rel)
        if rel is not None:
            all_ratios.append(rel)

    mean_rel = float(np.mean(ratios)) if ratios else None
    if mean_rel is None:
        verdict = "INVALID"
    elif mean_rel >= REL_REFUTE_ABOVE:
        verdict = "REFUTED"      # hypothesis lives: structure survives
    elif mean_rel <= REL_CONFIRM_BELOW:
        verdict = "CONFIRMED"    # the model destroys what the real edit keeps
    else:
        verdict = "INCONCLUSIVE"

    return {"verdict": verdict, "metric": metric, "probe_dE": probe,
            "mean_oracle_rel": mean_rel, "n_cases": len(ratios),
            # Both numbers, always. The verdict depends on the exclusion, so
            # hiding the unexcluded mean would hide the load-bearing choice.
            "mean_oracle_rel_all_cases":
                float(np.mean(all_ratios)) if all_ratios else None,
            "n_cases_all": len(all_ratios),
            "n_excluded": len(all_ratios) - len(ratios),
            "oracle_valid_floor": ORACLE_VALID_FLOOR,
            "rule": {"refute_above": REL_REFUTE_ABOVE,
                     "confirm_below": REL_CONFIRM_BELOW},
            "per_case": per_case}
