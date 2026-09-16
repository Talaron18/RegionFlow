"""
Off-instance change: the measure for the THIRD failure mode.

WHY THIS EXISTS. Gate C's design anticipated two failure modes -- leak onto a
sibling, edit the wrong instance. The 40-case audit found a third, and it is
the whole of the metric-human disagreement (12/12):

    refedit-29   "parrot on the shoulder of the chef"  -> parrot on the STOVE
    refedit-165  "red dots on the person's shirt"      -> edited the TV
    refedit-121  "wine glass in front of the man on the right" -> the man LEFT
    refedit-49   "remove the black car"                -> the parking meter

The model edited OUTSIDE every mask the benchmark tracks, so target, siblings
and their comparison all stayed quiet. `no_change` fired and the case was read
as model inaction when it was metric blindness.

`run_gate_c.outcome_class` now reads `off_instance_peak` / `off_instance_ambient`
(OFF_INSTANCE_EDIT_RATIO below) and SPLITS what used to be one bucket into
`no_tracked_change` (genuinely nothing happened) and `off_instance_edit` (something
happened, just not anywhere tracked) -- so the four cases above no longer read
as inaction anywhere in the pipeline, not just in this module's own output.
`instance_selectivity` itself, and everything in gate_c_note.md Sec.9, stays
untouched; only outcome_class's classification of the no-tracked-signal bucket
changed.

WHY `background_change` CANNOT SEE IT. `background_change` is the MEAN over
everything outside the instance set. RefEdit-SD3 regenerates the whole frame,
so its ambient change is ~0.06 everywhere. A localised edit on one 64px object
moves that mean by well under its own noise. The signal is not in the mean, it
is in the PEAK: one compact region changed far more than its surroundings.

    ambient  = MEDIAN off-instance change   (robust to the hot spot itself)
    peak     = max over sliding windows of mean off-instance change
    ratio    = peak / ambient

A frame-wide tone shift moves ambient and peak together: ratio ~ 1. A repaint
of one off-target object moves peak alone: ratio >> 1.

RELATIONSHIP TO THE FROZEN METRIC. Nothing here touches `instance_selectivity`.
No threshold in gate_c_note.md Sec.9 is read, changed, or reinterpreted, and
`valid` / `leakage_ratio` / `sibling_excess_over_bg` are computed exactly as
before. This is an ADDITIONAL, SEPARATELY NAMED set of keys. That matters
because the frozen set was frozen before any model was scored, and Gate C has
now been scored -- so the honest move is to add a measure, not to revise one.

WHAT IT STILL CANNOT DO. It reports WHERE an off-instance change is, never
whether that change was wanted. A model legitimately asked to "add a hat to
the man" must change pixels outside every COCO instance mask. Read the ratio
as "something compact happened off-instance", then LOOK at the crop. The
returned bounding box is there so that looking is one line of code, because
this project's automated verdicts have been wrong every time they fired.

USAGE
    from off_instance import off_instance_change
    r = off_instance_change(src, edited, tmask, smasks)
    r["off_instance_ratio"]         # peak / ambient
    r["off_instance_over_target"]   # peak / target change  <- the discriminator
    r["off_instance_bbox"]          # (y0, x0, y1, x1), for the crop

    python off_instance.py --self-test
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- constants --
# Frozen BEFORE any arm is scored, and chosen from the benchmark's geometry
# rather than from a sweep:
#
# WINDOW 64 on 512px. The off-target objects in the four audited failures are
# ordinary COCO things -- a stove, a TV, a parking meter. The sibling area
# floor is min_sib_area 0.01 = 5243px, i.e. a ~72px square. A 64px window sits
# just under that, so it can resolve the smallest object the benchmark is
# willing to call an instance, without averaging it away inside a larger box.
#
# STRIDE 16. Four samples per window width. Dense enough that a 64px object
# lands near-centred in some window; cheap enough to be free.
#
# PURITY 0.90. A window touching the target's edge would read the target's own
# change as off-instance. Requiring 90% of the window to lie outside the
# instance set keeps boundary windows out. Combined with DILATE below.
#
# DILATE 8. COCO polygons are tight and the models resample; a few pixels of
# edge bleed around the target is not an off-instance edit.
WINDOW = 64
STRIDE = 16
PURITY = 0.90
DILATE = 8

# OFF_INSTANCE_EDIT_RATIO = 10.0. The threshold run_gate_c.outcome_class uses
# to promote a case out of "no_tracked_change" into "off_instance_edit": a
# peak more than 10x the off-instance ambient. Not a fresh number -- it
# reuses instance_selectivity's own K=10 "how far above the noise floor
# counts as real" convention (the same K the `valid` gate applies to
# target_change vs background_change), applied here to off_instance_peak vs
# off_instance_ambient instead. The self-test below already brackets it from
# both sides at this window/stride/purity: the genuine off-instance edit
# (control 3) reads > 20, and the false-positive controls -- clean target
# edit (4), boundary bleed (5) -- read < 2. 10 sits in the gap with margin on
# both sides.
#
# NOT calibrated against the 12 known metric/human disagreements from the
# blind audit -- that data lives in runs/gate_c/, off this machine. Check
# this threshold against runs/gate_c/blind_key.json's 12 disagreement cases
# before trusting it on a real scoring run; if it misses any of them, that
# is a finding about this constant, not about the model being scored.
OFF_INSTANCE_EDIT_RATIO = 10.0


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r, via a separable max filter. No scipy dependency:
    this module is imported by the scoring path and should not add one."""
    if r <= 0:
        return mask
    m = mask
    for axis in (0, 1):
        out = m.copy()
        for s in range(1, r + 1):
            out |= np.roll(m, s, axis=axis)
            out |= np.roll(m, -s, axis=axis)
        m = out
    return m


def _integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row/col, so a box sum is 4 lookups."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s


def _box_sums(integ: np.ndarray, win: int, stride: int):
    """Sums of every `win`x`win` box on a `stride` grid, plus their origins."""
    h, w = integ.shape[0] - 1, integ.shape[1] - 1
    if h < win or w < win:
        return None, None, None
    ys = np.arange(0, h - win + 1, stride)
    xs = np.arange(0, w - win + 1, stride)
    y0, x0 = np.meshgrid(ys, xs, indexing="ij")
    y1, x1 = y0 + win, x0 + win
    sums = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
    return sums, y0, x0


def off_instance_change(source, edited, target_mask, sibling_masks,
                        window: int = WINDOW, stride: int = STRIDE,
                        purity: float = PURITY, dilate: int = DILATE,
                        eps: float = 1e-9) -> dict:
    """Peak compact change strictly outside the tracked instance set.

    Returns keys all prefixed `off_instance_` so they cannot be confused with
    the frozen metric's output when both are spliced into one record.

      off_instance_ambient    median change off-instance -- the frame's floor
      off_instance_peak       highest windowed mean change off-instance
      off_instance_ratio      peak / ambient. ~1 = uniform; >>1 = a hot spot
      off_instance_over_target  peak / target_change. THE discriminator: >1
                              means more happened somewhere untracked than
                              happened where the instruction pointed
      off_instance_bbox       (y0, x0, y1, x1) of the peak window, for looking
      off_instance_n_windows  windows that passed the purity test
    """
    src = np.asarray(source, np.float32) / 255.0
    ed = np.asarray(edited, np.float32) / 255.0
    diff = np.abs(src - ed).mean(axis=-1)

    inst = np.asarray(target_mask) > 0
    for sm in sibling_masks:
        inst = inst | (np.asarray(sm) > 0)
    off = ~_dilate(inst, dilate)

    tm = np.asarray(target_mask) > 0
    tgt_change = float(diff[tm].mean()) if tm.any() else 0.0

    if not off.any():
        return _empty(tgt_change)

    ambient = float(np.median(diff[off]))

    masked = np.where(off, diff, 0.0)
    sums, y0, x0 = _box_sums(_integral(masked), window, stride)
    counts, _, _ = _box_sums(_integral(off.astype(np.float64)), window, stride)
    if sums is None:
        return _empty(tgt_change, ambient=ambient)

    keep = counts >= purity * window * window
    if not keep.any():
        return _empty(tgt_change, ambient=ambient)

    means = np.where(keep, sums / np.maximum(counts, 1.0), -np.inf)
    k = int(np.argmax(means))
    peak = float(means.flat[k])
    py, px = int(y0.flat[k]), int(x0.flat[k])

    return {
        "off_instance_ambient": ambient,
        "off_instance_peak": peak,
        "off_instance_ratio": peak / (ambient + eps),
        "off_instance_over_target": peak / (tgt_change + eps),
        "off_instance_bbox": [py, px, py + window, px + window],
        "off_instance_n_windows": int(keep.sum()),
    }


def _empty(tgt_change: float, ambient: float = 0.0) -> dict:
    return {
        "off_instance_ambient": ambient,
        "off_instance_peak": 0.0,
        "off_instance_ratio": 0.0,
        "off_instance_over_target": 0.0,
        "off_instance_bbox": None,
        "off_instance_n_windows": 0,
    }


# ------------------------------------------------------------------ controls --

def self_test() -> None:
    """Controls with known answers, run BEFORE the measure is trusted -- the
    same battery discipline that caught the missing sibling-only control in
    the frozen metric's calibration.
    """
    rng = np.random.default_rng(0)
    H = W = 512
    src = rng.integers(40, 200, (H, W, 3), dtype=np.uint8)

    t = np.zeros((H, W), bool); t[100:200, 100:200] = True
    s = np.zeros((H, W), bool); s[100:200, 300:400] = True

    # 1. IDENTITY -> no hot spot, ratio is 0/0-safe.
    r = off_instance_change(src, src, t, [s])
    assert r["off_instance_peak"] == 0.0, r
    assert r["off_instance_over_target"] == 0.0, r

    # 2. GLOBAL TONE SHIFT -> peak and ambient move TOGETHER. This is the
    #    control the frozen metric's `leakage_ratio` cannot pass (it reads
    #    1.000 here); the ratio must stay near 1.
    shift = np.clip(src.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    r = off_instance_change(src, shift, t, [s])
    assert 0.9 < r["off_instance_ratio"] < 1.1, r["off_instance_ratio"]

    # 3. OFF-INSTANCE EDIT, the third failure mode. Repaint a 64px block that
    #    is neither target nor sibling, and leave both instances alone. This
    #    is `refedit-49`: the parking meter, not the car.
    ed = src.copy()
    ed[350:414, 60:124] = 255
    r = off_instance_change(src, ed, t, [s])
    assert r["off_instance_ratio"] > 20, r["off_instance_ratio"]
    assert r["off_instance_over_target"] > 20, r["off_instance_over_target"]
    y0, x0, y1, x1 = r["off_instance_bbox"]
    assert 330 <= y0 <= 360 and 40 <= x0 <= 70, r["off_instance_bbox"]

    # 4. CLEAN TARGET EDIT -> the change is INSIDE the target, so the
    #    off-instance peak must stay at ambient. A measure that fired here
    #    would flag every successful edit.
    ed = src.copy()
    ed[100:200, 100:200] = 255
    r = off_instance_change(src, ed, t, [s])
    assert r["off_instance_ratio"] < 1.5, r["off_instance_ratio"]
    assert r["off_instance_over_target"] < 0.1, r["off_instance_over_target"]

    # 5. BOUNDARY BLEED -> a few pixels of resampling slop around the target
    #    must NOT read as an off-instance edit. This is what DILATE buys.
    ed = src.copy()
    ed[96:204, 96:204] = 255
    r = off_instance_change(src, ed, t, [s])
    assert r["off_instance_ratio"] < 2.0, r["off_instance_ratio"]

    print("off_instance self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
