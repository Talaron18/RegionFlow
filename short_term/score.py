#!/usr/bin/env python3
"""
Score a local edit run, with the background metric that is not diluted.

WHY THIS EXISTS, AND WHY IT IS NOT JUST `off_instance`.

The first tier A run on the GPU box came back with these two numbers for the
same output:

    mean |delta| outside the mask, whole frame   0.0513   (looks fine)
    mean |delta| on NON-WHITE content outside    0.5789   (the other object
                                                            was destroyed)

11.3x apart. The test image is mostly flat white, a flat white field
regenerates to a flat white field, and the "background" term averages over
210k pixels that could not have changed. `bench/off_instance.py` exists for
exactly this reason at the benchmark scale ("RefEdit-SD3 regenerates the whole
frame"); this is the per-run version, so that a beta/kappa sweep is compared on
the honest number rather than on the diluted one.

WHAT IT REPORTS

    d_inside            mean |delta| inside the edit mask
    d_out_all           mean |delta| over the whole frame minus the mask
    d_out_content       mean |delta| over non-background pixels outside the mask
    d_out_dilution      d_out_content / d_out_all  -- how much the flat
                        background is hiding. Report this, always.
    contrast            d_inside / d_out_content
    sat_in              out/src saturation ratio inside the mask. Below 1 is
                        the Stage 0 signature: stronger edit, greyer result
                        (STAGE0_FINDINGS SS1: edit strength and source
                        destruction are the same axis).
    struct_in           out/src pixel std inside the mask, i.e. did the object
                        survive or dissolve into a wash.

VERDICTS (thresholds copied from scripts/stage0_real_edit.py's pre-registered
criteria, so a local run and a Stage 0 run mean the same thing):

    target_changed      d_inside        > 0.06
    background_held     d_out_content   < 0.02
    contrast_ok         contrast        > 4.0
    valid_wholeframe    d_inside        > 10 * d_out_all     (Gate C's own
                                                              definition, which
                                                              is the diluted one)

    python short_term/score.py --src synthetic_src.png --mask synthetic_mask.png \\
        --out runs/st/a_none_b5_k0.png

    # a whole sweep, ranked, with the knobs read off the .json sidecars
    python short_term/score.py --src synthetic_src.png --mask synthetic_mask.png \\
        --out "runs/st/d_target_*.png" --rank

    python short_term/score.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term.backends import load_image, load_mask  # noqa: E402

#: From scripts/stage0_real_edit.py's "SUCCESS IS DECLARED BEFORE THE RUN"
#: block. Copied, not re-chosen: a local run and a Stage 0 run have to mean the
#: same thing or the comparison is worthless.
TARGET_CHANGED = 0.06
BACKGROUND_HELD = 0.02
CONTRAST_MIN = 4.0
VALID_RATIO = 10.0

#: A pixel counts as background if it is near-white AND near-flat. Both, so a
#: white object in a photograph is not mistaken for empty background.
WHITE_MIN = 0.90

#: Coarse named colours for --expect. Deliberately crude: the point is not
#: colour science, it is to turn "the region changed" into "the region changed
#: in the requested direction". A metric that cannot tell those apart is how
#: the first tier A "test" got scored on an image with no shirt in it.
NAMED_COLOURS = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 0.6, 0.0), "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0), "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0), "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 0.5), "pink": (1.0, 0.7, 0.8),
    "brown": (0.5, 0.3, 0.1), "grey": (0.5, 0.5, 0.5),
    "gray": (0.5, 0.5, 0.5), "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
}


def parse_expect(text: str | None):
    """A colour name, 'R,G,B', or the special value 'tgt'.

    'tgt' means: the expected content is the DATASET'S OWN edited target, taken
    from --tgt. That is the general form and the only one that works for edits
    with no colour in them -- "A hot air balloon should also be floating" has
    nothing for a colour name to name, and without this the scorer can only say
    that the region changed, not whether it changed toward the right content.
    """
    if text is None:
        return None
    key = text.strip().lower()
    if key == "tgt":
        return "tgt"
    if key in NAMED_COLOURS:
        return np.array(NAMED_COLOURS[key], np.float32)
    parts = [float(v) for v in key.replace(",", " ").split()]
    if len(parts) != 3:
        raise SystemExit(
            f"--expect needs a colour name ({', '.join(sorted(NAMED_COLOURS))}), "
            f"three numbers, or 'tgt' for the dataset's own target; got "
            f"{text!r}")
    vals = np.array(parts, np.float32)
    if vals.max() > 1.5:
        vals = vals / 255.0
    return vals


def _sat(a: np.ndarray) -> np.ndarray:
    return a.max(axis=2) - a.min(axis=2)


def score(src: np.ndarray, out: np.ndarray, mask: np.ndarray,
          expect: np.ndarray | None = None) -> dict:
    m = mask > 0.5
    d = np.abs(out - src).mean(axis=2)
    inside = m
    # BACKGROUND = non-white. A flat white field cannot change, so including it
    # in the denominator is what produced the 11.3x gap.
    background = (~m) & (src.min(axis=2) >= WHITE_MIN)
    content_out = (~m) & (~background)
    s_src, s_out = _sat(src), _sat(out)

    def mean(a, sel):
        return float(a[sel].mean()) if sel.any() else float("nan")

    d_in = mean(d, inside)
    d_all = mean(d, ~m)
    d_con = mean(d, content_out) if content_out.any() else 0.0
    # A ratio to zero is not a big number, it is undefined. Reporting it as
    # 5.6e8 blows the column up and reads like a spectacular win; a mask-only
    # edit with a bit-identical background IS the ideal case, and `inf` says so.
    if d_con > 0.0:
        contrast = d_in / d_con
    else:
        contrast = float("inf") if d_in > 0.0 else float("nan")
    sat_in = (mean(s_out, inside) / max(mean(s_src, inside), 1e-9)
              if inside.any() else float("nan"))
    struct_in = (float(out[inside].std()) / max(float(src[inside].std()), 1e-9)
                 if inside.any() else float("nan"))

    rec = {
        "d_inside": d_in,
        "d_out_all": d_all,
        "d_out_content": d_con,
        "d_out_dilution": (d_con / d_all) if d_all > 0.0 else float("nan"),
        "contrast": contrast,
        "sat_in": sat_in,
        "struct_in": struct_in,
        "mask_frac": float(m.mean()),
        "content_out_frac": float(content_out.mean()),
        "mean_rgb_in_src": [round(float(v), 3) for v in src[inside].mean(0)]
        if inside.any() else None,
        "mean_rgb_in_out": [round(float(v), 3) for v in out[inside].mean(0)]
        if inside.any() else None,
        "verdict": {
            "target_changed": bool(d_in > TARGET_CHANGED),
            "background_held": bool(d_con < BACKGROUND_HELD),
            "contrast_ok": bool(contrast > CONTRAST_MIN),
            "valid_wholeframe": bool(d_in > VALID_RATIO * d_all),
        },
    }

    # "DID IT DO THE RIGHT THING", not just "did something happen". Without
    # this, the first tier A run on an image with no shirt in it scored
    # target_changed=PASS while the region went from solid red to a pale wash.
    #
    # `expect` is either a colour (3,) or a full target IMAGE (H,W,3), the
    # latter being --expect tgt. Both reduce to "distance from this pixel to the
    # expected content, averaged over the mask".
    if expect is not None and inside.any():
        e = np.asarray(expect, np.float32)
        # Accept exactly two shapes: a colour of C values, or a full image
        # matching the source. Anything else is refused with the shape printed
        # -- a silent reshape here would compare against nonsense.
        if e.ndim == 3 and e.shape[:2] == src.shape[:2]:
            if e.shape[2] != src.shape[2]:
                raise SystemExit(
                    f"--expect image has {e.shape[2]} channels; the source has "
                    f"{src.shape[2]}")
            as_image = True
        elif e.ndim <= 1 or (e.ndim >= 1 and int(np.prod(e.shape)) ==
                             src.shape[2]):
            if int(np.prod(e.shape)) != src.shape[2]:
                raise SystemExit(
                    f"--expect colour has {int(np.prod(e.shape))} values; the "
                    f"image has {src.shape[2]} channels")
            e = e.reshape(1, 1, -1)
            as_image = False
        else:
            raise SystemExit(
                f"--expect shape {e.shape} is neither a {src.shape[2]}-value "
                f"colour nor an image matching the source {src.shape}")
        d_src_e = mean(np.abs(src - e).mean(axis=2), inside)
        d_out_e = mean(np.abs(out - e).mean(axis=2), inside)
        rec["d_expect_src"] = d_src_e
        rec["d_expect_out"] = d_out_e
        rec["expect_kind"] = "tgt-image" if as_image else "colour"
        rec["expect_name"] = "the dataset's target" if as_image else None
        # > 0 moved toward the expected content, ~0 no-op, < 0 moved AWAY.
        rec["expect_gain"] = 1.0 - d_out_e / max(d_src_e, 1e-9)
        rec["verdict"]["toward_expect"] = bool(rec["expect_gain"] > 0.0)
        # VACUITY GUARD. Two ways a case can be unable to answer the question:
        #   colour form: the requested colour is already the region's nearest
        #     named colour, so a correct answer is "no change";
        #   tgt form: the dataset's own target is nearly the source inside the
        #     mask, so there is no edit to measure and every run scores ~0.
        mean_src = src[inside].mean(0).astype(np.float32)
        if as_image:
            rec["expect_distance"] = d_src_e
            rec["expect_nearest_other"] = None
            rec["expect_vacuous"] = bool(d_src_e < 0.01)
        else:
            exp = e.reshape(-1)
            near_expect = float(np.abs(mean_src - exp).mean())
            near_any = min(
                (float(np.abs(mean_src - np.array(c, np.float32)).mean())
                 for c in NAMED_COLOURS.values() if not np.allclose(c, exp)),
                default=float("inf"))
            rec["expect_distance"] = near_expect
            rec["expect_nearest_other"] = near_any
            rec["expect_vacuous"] = bool(near_expect <= near_any + 1e-6)
    return rec


def criterion_margins(st: dict) -> list[tuple[str, float]]:
    """How many TIMES OVER each threshold a (ground-truth) score sits.

    A criterion the data itself clears by only a hair cannot discriminate a good
    run from a bad one, and must not be reported as a verdict. Measured on the
    MagicBrush balloon case the truth clears `valid_wholeframe` by 1.4% (ratio
    10.15 against a threshold of 10) and `target_changed` by 1.17x (d_inside
    0.070 against 0.06). Without this check, a run within 7% of the truth on the
    inside change reads FAIL on a criterion calibrated to nothing -- which is
    exactly how the both-terms result would have been misread.
    """
    ratio = st["d_inside"] / max(st["d_out_all"], 1e-9)
    return [
        ("target_changed", st["d_inside"] / TARGET_CHANGED),
        ("background_held", BACKGROUND_HELD / max(st["d_out_content"], 1e-9)),
        ("contrast_ok", st["contrast"] / CONTRAST_MIN),
        ("valid_wholeframe", ratio / VALID_RATIO),
    ]


def calibration_warnings(st: dict, factor: float = 1.5
                         ) -> list[tuple[str, float]]:
    """The subset of criteria with no margin, given the ground truth."""
    return [(n, m) for n, m in criterion_margins(st) if m < factor]


def parse_terms(rec: dict) -> dict:
    """Normalise edit.py's `terms` field, which has TWO shapes.

    One guidance term is a dict; `--guide both` writes a LIST of them. The first
    version of `sidecar` did `rec.get("terms", {}).get("guide")` and died with
    "'list' object has no attribute 'get'" the first time a two-term run was
    scored -- the second time a schema change in edit.py broke this reader, so
    the parsing now lives in one tested function instead of inline.
    """
    terms = rec.get("terms")
    if isinstance(terms, dict):
        terms = [terms]
    elif not isinstance(terms, list):
        terms = []
    terms = [t for t in terms if isinstance(t, dict)]
    guide = terms[0].get("guide") if terms else None
    weights = [t.get("weight") for t in terms]
    names = [t.get("term") for t in terms]
    # Two-term runs need both strengths in the table, or a reader cannot tell
    # which of the two kappas produced the row.
    k_in = rec.get("kappa")
    k_out = rec.get("kappa_out_used")
    kappa = k_in
    if k_out is not None and k_in is not None and k_out != k_in:
        kappa = f"{k_in:g}/{k_out:g}"
    return {
        "guide": guide,
        "terms_n": len(terms),
        "term_names": names or None,
        "term_weights": weights or None,
        "kappa": kappa,
    }


def sidecar(out_path: Path) -> dict:
    """Read edit.py's provenance record next to the image, if there is one."""
    p = out_path.with_suffix(".json")
    if not p.is_file():
        return {}
    rec = json.loads(p.read_text())
    out = {
        "beta": rec.get("beta"),
        "mode": rec.get("mode"),
        "cfg_factor": rec.get("cfg_factor"),
        "guided_steps": rec.get("guided_steps"),
        "scheduler_dev": rec.get("scheduler_dev"),
    }
    out.update(parse_terms(rec))
    return out


def report_one(path: Path, src: np.ndarray, mask: np.ndarray, expect=None,
               verbose=True) -> dict:
    out = load_image(path)
    if out.shape != src.shape:
        print(f"  !! {path.name}: shape {out.shape} != source {src.shape}; "
              "resizing would silently change every number, so this is being "
              "skipped rather than guessed.")
        return {}
    s = score(src, out, mask, expect=expect)
    s["_path"] = str(path)
    s.update(sidecar(path))
    if not verbose:
        return s
    print(f"\n--- {path.name} ---")
    if s.get("guide") is not None:
        print(f"  guide={s['guide']} beta={s['beta']} kappa={s['kappa']} "
              f"mode={s['mode']}")
        if s.get("guided_steps") is not None:
            print(f"  guided_steps={s['guided_steps']} "
                  f"cfg_factor={s['cfg_factor']} "
                  f"scheduler_dev={s['scheduler_dev']}")
    print(f"  mask covers {100 * s['mask_frac']:.1f}% of the frame; "
          f"{100 * s['content_out_frac']:.1f}% is non-white content outside it")
    print(f"  d_inside        {s['d_inside']:.4f}      "
          f"(target_changed  {'PASS' if s['verdict']['target_changed'] else 'FAIL'})")
    print(f"  d_out_content   {s['d_out_content']:.4f}      "
          f"(background_held {'PASS' if s['verdict']['background_held'] else 'FAIL'})")
    print(f"  d_out_all       {s['d_out_all']:.4f}      "
          f"<- DILUTED by {s['d_out_dilution']:.1f}x, do not read this alone")
    print(f"  contrast        {s['contrast']:.2f}      "
          f"(contrast_ok     {'PASS' if s['verdict']['contrast_ok'] else 'FAIL'})")
    print(f"  valid (Gate C)  {'PASS' if s['verdict']['valid_wholeframe'] else 'FAIL'}"
          f"  = d_inside > 10 * d_out_all ({s['d_inside']:.3f} vs "
          f"{VALID_RATIO * s['d_out_all']:.3f})")
    print(f"  in-mask saturation  {s['sat_in']:.3f} x source"
          f"   ({'desaturated' if s['sat_in'] < 1 else 'saturated'})")
    print(f"  in-mask structure   {s['struct_in']:.3f} x source"
          f"   ({'dissolved' if s['struct_in'] < 1 else 'sharpened'})")
    print(f"  in-mask RGB  {s['mean_rgb_in_src']} -> {s['mean_rgb_in_out']}")
    if "expect_gain" in s:
        what = s.get("expect_name") or "the requested colour"
        print(f"  expect_gain     {s['expect_gain']:+.3f}      "
              f"(distance to {what} "
              f"{s['d_expect_src']:.3f} -> {s['d_expect_out']:.3f})"
              f"  {'TOWARD' if s['verdict']['toward_expect'] else 'AWAY/NO-OP'}")
        if s["expect_vacuous"]:
            if s.get("expect_kind") == "tgt-image":
                print("  !! --expect tgt VACUOUS: the dataset's own target is "
                      f"nearly identical to the source inside the mask "
                      f"(distance {s['expect_distance']:.4f}), so there is no "
                      "edit here to measure and every run will score ~0. Pick "
                      "a cache item with a real change (see cache_item.py "
                      "--list).")
            else:
                print("  !! --expect VACUOUS: the requested colour is already "
                      "the nearest named colour to the masked region "
                      f"(distance {s['expect_distance']:.3f} vs "
                      f"{s['expect_nearest_other']:.3f} to the next), so a "
                      "correct answer would be 'no change' and d_inside cannot "
                      "tell a correct no-op from a failure. Pick a target the "
                      "region does not already have, or use --expect tgt.")
    if s["content_out_frac"] < 0.05:
        print("  WARNING: less than 5% of the frame is non-white content "
              "outside the mask. The background verdict rests on very few "
              "pixels and the diluted number will look reassuring. Use a real "
              "photograph before concluding anything about protection.")
    return s


def _self_test() -> int:
    """CPU, no weights. Checks the metric arithmetic AND the dilution warning,
    on a constructed case where both answers are known by construction."""
    import tempfile

    rng = np.random.default_rng(0)
    h = w = 64
    src = np.ones((h, w, 3), np.float32)
    src[8:24, 8:24] = np.array([0.9, 0.1, 0.1], np.float32)     # in mask
    src[40:56, 40:56] = np.array([0.1, 0.7, 0.1], np.float32)   # other object
    mask = np.zeros((h, w), np.float32)
    mask[8:24, 8:24] = 1.0

    # (a) perfect: nothing changed anywhere
    s = score(src, src.copy(), mask)
    assert s["d_inside"] == 0.0 and s["d_out_content"] == 0.0
    assert not s["verdict"]["target_changed"]

    # (b) inside changes a lot, outside content untouched -> the good case, and
    #     d_out_all is diluted to ~0 because most of the frame is flat white
    out = src.copy()
    out[8:24, 8:24] = np.array([0.0, 0.0, 0.9], np.float32)
    s = score(src, out, mask)
    # per-channel |delta| = (0.9 + 0.1 + 0.8)/3
    assert abs(s["d_inside"] - 0.6) < 1e-6, s["d_inside"]
    assert s["d_out_content"] == 0.0
    assert s["verdict"]["target_changed"] and s["verdict"]["background_held"]

    # (c) the tier A failure: the OTHER object is destroyed. d_out_all stays
    #     small because ~93% of the frame is flat white and cannot change,
    #     while d_out_content reveals it. This is the 11.3x gap, constructed.
    out2 = src.copy()
    out2[8:24, 8:24] = np.array([0.0, 0.0, 0.9], np.float32)
    out2[40:56, 40:56] = np.array([0.95, 0.95, 0.95], np.float32)
    s = score(src, out2, mask)
    assert s["d_out_content"] > 0.5, s["d_out_content"]
    assert s["d_out_all"] < 0.1, s["d_out_all"]
    assert s["d_out_dilution"] > 5.0, s["d_out_dilution"]
    assert not s["verdict"]["background_held"], "must FAIL background_held"
    # THE POINT OF THE WHOLE TOOL: Gate C's own `valid` criterion PASSES here,
    # on a frame whose only other object was destroyed. It is a diluted ratio
    # and must never be read as "the background was preserved". (On the real
    # tier A output it happened to fail as well, 0.409 vs 0.513 -- which is
    # luck, not a property of the criterion.)
    assert s["verdict"]["valid_wholeframe"], (
        "this construction is supposed to show valid_wholeframe PASSING while "
        "the content outside is destroyed; if it now fails, the criterion "
        "changed and the docstring above is wrong")

    # (d) an all-white frame has NO content outside, which must not be read as
    #     "background held" by accident -- it is reported as 0.0 with a warning
    empty = src.copy()
    empty[40:56, 40:56] = 1.0
    s = score(empty, empty.copy(), mask)
    assert s["content_out_frac"] == 0.0 and s["d_out_content"] == 0.0

    # (e) saturation/shrinkage direction
    out3 = src.copy()
    out3[8:24, 8:24] = np.array([0.5, 0.45, 0.45], np.float32)
    s = score(src, out3, mask)
    assert s["sat_in"] < 1.0, s["sat_in"]
    assert s["struct_in"] < 1.0, s["struct_in"]

    # (g) --expect: the metric that tells "changed" from "changed correctly".
    # The source region is red; ask for blue.
    blue = parse_expect("blue")
    assert np.allclose(blue, [0, 0, 1])
    assert np.allclose(parse_expect("0,0,255"), [0, 0, 1])
    # the do-nothing baseline: exp_gain is exactly 0 and d_inside exactly 0,
    # so a negative run is provably worse than not running the model
    noop = score(src, src.copy(), mask, expect=blue)
    assert noop["expect_gain"] == 0.0 and noop["d_inside"] == 0.0
    assert noop["verdict"]["background_held"]
    # a correct recolour toward blue gains; the bleached wash LOSES
    correct = src.copy()
    correct[8:24, 8:24] = np.array([0.0, 0.0, 0.9], np.float32)
    s_ok = score(src, correct, mask, expect=blue)
    assert s_ok["expect_gain"] > 0.5, s_ok["expect_gain"]
    assert not s_ok["expect_vacuous"]
    wash = src.copy()
    wash[8:24, 8:24] = np.array([1.0, 0.69, 0.64], np.float32)   # the real one
    s_wash = score(src, wash, mask, expect=blue)
    assert s_wash["expect_gain"] < 0.0, s_wash["expect_gain"]
    # and the vacuity guard, on a region that really is red: asking for red is
    # a vacuous test, asking for blue is not
    red_disc = np.ones((h, w, 3), np.float32)
    red_disc[8:24, 8:24] = np.array([0.86, 0.12, 0.12], np.float32)
    s_vac = score(red_disc, red_disc.copy(), mask,
                  expect=parse_expect("red"))
    assert s_vac["expect_vacuous"], s_vac
    s_not = score(red_disc, red_disc.copy(), mask, expect=blue)
    assert not s_not["expect_vacuous"], s_not

    # (h) --expect tgt: the general form, for edits with no colour in them.
    # The "target" is an image, the scale is anchored at no-op = 0 and
    # exact-target = 1, and reproducing the target scores exactly 1.
    assert parse_expect("tgt") == "tgt"
    assert parse_expect("TGT") == "tgt"
    tgt_img = src.copy()
    tgt_img[8:24, 8:24] = np.array([0.1, 0.2, 0.95], np.float32)  # "add a thing"
    s_t = score(src, src.copy(), mask, expect=tgt_img)           # no-op
    assert abs(s_t["expect_gain"]) < 1e-9, s_t["expect_gain"]
    assert s_t["expect_kind"] == "tgt-image"
    s_1 = score(src, tgt_img.copy(), mask, expect=tgt_img)       # exact
    assert abs(s_1["expect_gain"] - 1.0) < 1e-9, s_1["expect_gain"]
    assert s_1["verdict"]["toward_expect"] and not s_1["expect_vacuous"]
    # a run that moves AWAY from the target is negative, which is the case a
    # bare d_inside cannot distinguish from success. The colour is chosen so the
    # arithmetic is not a coincidence: with src=(0.9,0.1,0.1) and
    # tgt=(0.1,0.2,0.95), mean|src-tgt| = 1.75/3, and mean|yellow-tgt| =
    # (0.9+0.8+0.95)/3 = 2.65/3 > 1.75/3. (White is NOT a valid choice here --
    # it lands at exactly 1.75/3 and scores a spurious 0.000.)
    away = src.copy()
    away[8:24, 8:24] = np.array([1.0, 1.0, 0.0], np.float32)
    s_away = score(src, away, mask, expect=tgt_img)
    assert s_away["expect_gain"] < 0.0, s_away["expect_gain"]
    # vacuous when the dataset's target is itself a no-op
    s_v = score(src, src.copy(), mask, expect=src.copy())
    assert s_v["expect_vacuous"], s_v
    # a shape mismatch is refused rather than broadcast
    try:
        score(src, src.copy(), mask, expect=np.zeros((h, w, 4), np.float32))
    except SystemExit as exc:
        assert "channels" in str(exc)
    else:
        raise AssertionError("a 4-channel expect must be refused")

    # (f) glob + sidecar plumbing on a real directory layout
    assert glob.glob(str(Path(__file__).resolve().parent / "*.py")), \
        "glob mode is broken"
    missing = sidecar(Path(__file__).resolve().parent / "does_not_exist.png")
    assert missing == {}, missing

    # (i) the sidecar reader against BOTH shapes edit.py writes. `terms` is a
    # dict for one guidance term and a LIST for --guide both; the reader died
    # with "'list' object has no attribute 'get'" the first time it met the
    # list, so both shapes are pinned here.
    with tempfile.TemporaryDirectory() as tmp:
        one = Path(tmp) / "one.json"
        one.write_text(json.dumps({
            "beta": 5.0, "kappa": 0.6, "terms": {"guide": "source",
                                                 "weight": "1-mask"},
            "mode": "tier D (guided)", "guided_steps": 49,
            "scheduler_dev": 1e-7,
        }))
        p = sidecar(Path(tmp) / "one.png")
        assert p["guide"] == "source" and p["terms_n"] == 1, p
        assert p["kappa"] == 0.6 and p["guided_steps"] == 49, p

        both = Path(tmp) / "both.json"
        both.write_text(json.dumps({
            "beta": 5.0, "kappa": 0.6, "kappa_out_used": 1.0,
            "terms": [{"guide": "both", "term": "outside", "weight": "1-mask"},
                      {"guide": "both", "term": "inside", "weight": "mask"}],
            "mode": "tier D (guided)",
        }))
        q = sidecar(Path(tmp) / "both.png")
        assert q["guide"] == "both" and q["terms_n"] == 2, q
        assert q["term_names"] == ["outside", "inside"], q
        assert q["term_weights"] == ["1-mask", "mask"], q
        # both strengths must survive into the table, or a reader cannot tell
        # which kappa produced the row
        assert q["kappa"] == "0.6/1", q["kappa"]
        assert one.with_suffix(".json").is_file()      # sanity: files written

        # a record with no `terms` at all (an older tier A run) still parses
        old = Path(tmp) / "old.json"
        old.write_text(json.dumps({"beta": 5.0, "kappa": 0.0}))
        o = sidecar(Path(tmp) / "old.png")
        assert o["guide"] is None and o["terms_n"] == 0, o
        # and a malformed one must not raise
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"terms": "not a dict or a list"}))
        assert sidecar(Path(tmp) / "bad.png")["terms_n"] == 0

    # (j) criterion calibration, pinned to the REAL MagicBrush balloon numbers.
    # These come from an actual run's tgt row: the dataset's own edit scores
    # d_in 0.070, d_out_content 0.0069, contrast 10.15. Two of the four criteria
    # have essentially no margin, which is why the `valid` column read FAIL for
    # a run that matched the truth on every absolute number.
    balloon = {"d_inside": 0.070, "d_out_content": 0.0069, "d_out_all": 0.0069,
               "contrast": 10.15}
    margins = dict(criterion_margins(balloon))
    assert abs(margins["target_changed"] - 1.1667) < 1e-3, margins
    assert abs(margins["valid_wholeframe"] - 1.0145) < 1e-3, margins
    assert margins["background_held"] > 2.0 and margins["contrast_ok"] > 2.0
    weak = [n for n, _m in calibration_warnings(balloon)]
    assert weak == ["target_changed", "valid_wholeframe"], weak
    # a case with real margin flags nothing
    healthy = {"d_inside": 0.35, "d_out_content": 0.004, "d_out_all": 0.004,
               "contrast": 12.0}
    assert calibration_warnings(healthy) == [], calibration_warnings(healthy)
    # and the "against the truth" multiples are what the both-terms run should
    # have been read on: d_in 0.075 against the truth's 0.070 is 1.07x
    assert abs(0.075 / 0.070 - 1.0714) < 1e-3

    print("score self-test OK")
    print("  (b) good case: d_out_content 0.0 and d_out_all ~0 together")
    print("  (c) the real failure: an object destroyed outside the mask shows")
    print("      up in d_out_content and is HIDDEN in d_out_all by 5x+, AND")
    print("      Gate C's own diluted `valid` ratio still PASSES there")
    print("  (e) saturation/structure ratios go the right way")
    print("  (g) expect_gain is POSITIVE for a correct recolour and NEGATIVE")
    print("      for the pale wash the first tier A run actually produced")
    print("  (h) --expect tgt: the general form. Anchored at no-op = 0.000 and")
    print("      exact-target = 1.000, negative when a run moves AWAY, and it")
    print("      refuses a shape it cannot interpret instead of reshaping it")
    print("  (i) the .json sidecar parses in all three shapes edit.py writes")
    print("      (one term / two terms / none) and never raises")
    print("  (j) criterion calibration on the real balloon tgt numbers: the")
    print("      truth clears valid_wholeframe by 1.01x and target_changed by")
    print("      1.17x, so those two columns must not be read as verdicts")
    return 0


def _fmt(x, width: int, prec: int) -> str:
    """Right-aligned, but never overflowing the column: a non-finite value
    becomes 'inf'/'nan' rather than a 10-digit ratio."""
    if x is None or not np.isfinite(x):
        return f"{('inf' if x == float('inf') else 'nan'):>{width}}"
    return f"{x:>{width}.{prec}f}"


def _row(name: str, s: dict, expect) -> str:
    """One table line. Shared by the runs, the no-op row and the tgt row so
    they are guaranteed comparable."""
    row = (f"{name:<34}{str(s.get('guide', '-')):>7}"
           f"{str(s.get('beta', '-')):>6}{str(s.get('kappa', '-')):>7}"
           f"{_fmt(s['d_inside'], 8, 3)}{_fmt(s['d_out_content'], 9, 4)}"
           f"{_fmt(s['d_out_dilution'], 7, 1)}{_fmt(s['contrast'], 8, 2)}"
           f"{_fmt(s['sat_in'], 8, 3)}")
    if expect is not None:
        row += _fmt(s.get("expect_gain", float("nan")), 9, 3)
    row += (f"{'PASS' if s['verdict']['background_held'] else 'FAIL':>6}"
            f"{'PASS' if s['verdict']['valid_wholeframe'] else 'FAIL':>7}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="the UNEDITED source image the run started from")
    ap.add_argument("--mask", default="white", help="the edit mask")
    ap.add_argument("--out", nargs="+", required=True,
                    help="output image(s); globs are expanded")
    ap.add_argument("--expect", default=None,
                    help="what the instruction asked for: a colour name "
                         "(red/green/blue/...), 'R,G,B', or 'tgt' for the "
                         "dataset's own target (needs --tgt). Adds expect_gain: "
                         "did the masked region move TOWARD the expected "
                         "content, or merely change. 'tgt' is the only form "
                         "that works for edits with no colour in them, e.g. "
                         "'a hot air balloon should also be floating'.")
    ap.add_argument("--tgt", default=None,
                    help="the dataset's own edited target (from "
                         "cache_item.py's tgt.png). Prints it as a reference "
                         "row on the same axes as the model, which is the "
                         "calibration STAGE1_REPORT SS3 requires: if the real "
                         "edit does not score near the top of your metric, the "
                         "metric is wrong.")
    ap.add_argument("--rank", action="store_true",
                    help="print a ranked one-line-per-run table")
    ap.add_argument("--size", type=int, default=None)
    args = ap.parse_args(argv)

    expect = parse_expect(args.expect)
    tgt_img = None
    if args.tgt is not None:
        tgt_img = load_image(args.tgt, args.size)
    if isinstance(expect, str) and expect == "tgt":
        if tgt_img is None:
            raise SystemExit("--expect tgt requires --tgt")
        expect = tgt_img
    src = load_image(args.src, args.size)
    mask = load_mask(args.mask, src.shape[0])
    if mask.shape != src.shape[:2]:
        print(f"mask {mask.shape} vs source {src.shape[:2]}: disagreeing "
              "grids make every number meaningless; stopping.")
        return 2

    paths: list[Path] = []
    for pat in args.out:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        paths.extend(Path(h) for h in hits)
    if not paths:
        print(f"no output matched {args.out}")
        return 2

    scored = [report_one(p, src, mask, expect=expect, verbose=not args.rank)
              for p in paths]
    scored = [s for s in scored if s]

    if args.rank:
        head = (f"{'output':<34}{'guide':>7}{'beta':>6}{'kappa':>7}"
                f"{'d_in':>8}{'d_out_c':>9}{'dilut':>7}{'contr':>7}"
                f"{'sat_in':>8}")
        if expect is not None:
            head += f"{'exp_gain':>9}"
        head += f"{'held':>6}{'valid':>7}"
        print(head)
        rows = []
        for s in sorted(scored, key=lambda x: (x["d_out_content"],
                                               -x["d_inside"])):
            rows.append(_row(Path(s["_path"]).name, s, expect))
        for row in rows:
            print(row)
        # THE TWO REFERENCE ROWS. A single-number "did something change" cannot
        # tell a correct edit from a wash; these two rows are what make the
        # columns above readable.
        #
        #   no-op  the source scored against itself. expect_gain is exactly 0,
        #          so any run BELOW it on exp_gain is worse on the requested
        #          attribute than not running the model at all.
        #   tgt    the dataset's own edited target. This is the calibration
        #          STAGE1_REPORT SS3 demands -- "score the ground truth on your
        #          metric before trusting it; if it doesn't score near the top,
        #          the metric is wrong". If the real edit does not clear
        #          d_out_content < 0.02, our background threshold is stricter
        #          than reality and must be reported as such, not used as a
        #          verdict.
        noop = score(src, src.copy(), mask, expect=expect)
        print(_row("(no-op: the source itself)", noop, expect))
        st = None
        if tgt_img is not None:
            if tgt_img.shape != src.shape:
                print(f"  !! --tgt shape {tgt_img.shape} != source "
                      f"{src.shape}; skipping the ground-truth row rather "
                      "than resizing it.")
            else:
                st = score(src, tgt_img, mask, expect=expect)
                print(_row("(tgt: the dataset's real edit)", st, expect))
                if expect is not None and st.get("expect_gain", 0) <= 0:
                    print("  !! the DATASET'S OWN EDIT does not move toward "
                          "the expectation. Either the expectation and the "
                          "instruction disagree, or the metric is wrong. Fix "
                          "that before reading any model row.")
                if not st["verdict"]["background_held"]:
                    print(f"  !! the DATASET'S OWN EDIT scores d_out_content "
                          f"{st['d_out_content']:.4f}, above the "
                          f"{BACKGROUND_HELD} background threshold. The "
                          "threshold is stricter than the data; report it as a "
                          "scale, not a verdict.")

        # CRITERION CALIBRATION. STAGE1_REPORT SS3 says to score the ground
        # truth on your metric before trusting it. This goes one step further:
        # a criterion the ground truth clears by only a hair cannot discriminate
        # anything, and should not be reported as a verdict. Measured on the
        # MagicBrush balloon case, the truth clears `valid_wholeframe` by 1.4%
        # (ratio 10.15 against a threshold of 10) and `target_changed` by 1.2x
        # (d_inside 0.070 against 0.06). A run within 7% of the truth on the
        # inside change therefore reads FAIL on a criterion calibrated to
        # nothing -- which is exactly how the both-terms result would have been
        # misread.
        if st is not None:
            weak = calibration_warnings(st)
            if weak:
                print("\n  !! THE GROUND TRUTH BARELY CLEARS "
                      + ", ".join(f"{n} ({m:.2f}x)" for n, m in weak)
                      + ".\n     A criterion the data itself clears by less "
                      "than 1.5x cannot discriminate a good run from a bad "
                      "one. Do NOT read those columns as verdicts on this "
                      "case -- compare against the tgt row instead "
                      "(d_in, d_out_c, sat_in, exp_gain as absolute numbers).")

            # The comparison that IS calibrated: our numbers against the
            # truth's, as multiples. A ratio criterion flatters a run whose
            # inside change is large and punishes a small-object edit; a
            # multiple against the truth does neither.
            print(f"\n  against the ground truth (multiples; 1.0 = as good as "
                  f"the dataset's own edit):")
            print(f"    {'run':<34}{'d_in/tgt':>10}{'d_out_c/tgt':>13}"
                  f"{'sat_in/tgt':>12}")
            for s in sorted(scored, key=lambda x: (x["d_out_content"],
                                                   -x["d_inside"])):
                din = s["d_inside"] / max(st["d_inside"], 1e-9)
                # The truth's own background drift can be exactly 0, in which
                # case "times the truth" is infinite, not 1e9. Say so.
                if st["d_out_content"] > 1e-6:
                    dout = s["d_out_content"] / st["d_out_content"]
                else:
                    dout = (0.0 if s["d_out_content"] <= 1e-6
                            else float("inf"))
                sat = s["sat_in"] / max(st["sat_in"], 1e-9)
                print(f"    {Path(s['_path']).name:<34}{din:>10.2f}"
                      f"{_fmt(dout, 13, 2)}{sat:>12.2f}")
            print("    (d_in ~1.0 = the right amount of change; d_out_c ~1.0 = "
                  "the truth's own background drift; these two are the ones to "
                  "steer by.)")

        print("\nsorted by d_out_content ascending: the best-protecting run "
              "first. A run 'wins' on protection only if it also keeps "
              "d_inside above 0.06 -- read both columns.")
        if expect is not None:
            what = ("the dataset's target" if noop.get("expect_kind")
                    == "tgt-image" else "the requested colour")
            print(f"exp_gain > 0 means the masked region moved TOWARD {what}; "
                  "~0 is a no-op; < 0 moved AWAY. Read the model rows against "
                  "the two reference rows below them, not against zero."
                  "\nCAVEAT: pixel distance to the target is a DIRECTION, not "
                  "a verdict. Two different-looking objects can both be close "
                  "to the target region, so a positive exp_gain is necessary "
                  "and not sufficient -- 'is it actually the thing that was "
                  "asked for' needs eyes or Gate C's VLM judge.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
