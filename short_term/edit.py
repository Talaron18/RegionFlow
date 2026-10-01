#!/usr/bin/env python3
"""
Tier A and tier D, one CLI.

    # TIER A -- run an editor that already exists. No training, no guidance.
    python short_term/edit.py --image src.png --mask m.png \
        --instruction "change the thing in the mask to blue" --out runs/st/a.png

    # TIER D -- same checkpoint, guidance toward a reference region.
    python short_term/edit.py --image src.png --mask m.png --guide-image ref.png \
        --instruction "change the thing in the mask to blue" --guide target \
        --beta 5 --out runs/st/d.png

    # TIER D -- no reference: protect the background outside the mask (soft
    # pin), which is the train-free version of Stage 0's pin_background.
    python short_term/edit.py --image src.png --mask m.png \
        --instruction "change the thing in the mask to blue" \
        --guide source --invert-weight --sign +1 --kappa 0.3 \
        --out runs/st/protect.png

    # validate the plumbing without loading a model
    python short_term/edit.py --smoke

WHICH INSTRUCTION IS NOT ARBITRARY. The first version of this docstring said
"make the shirt red", copied from the repo's own canonical attribute edit
(region/alpha_field.py:8; bench/cases_refedit_final.json:5631 "Let the shirt of
the little girl be red."). Pointed at the synthetic two-disc test image it is
nonsense, and it cost the edit-quality half of the first experiment: the masked
region was ALREADY red, so a correct answer was "do nothing" and d_inside could
not tell a correct no-op from a failure. Two rules for a test instruction:

  1. it must name something that EXISTS in the image, and
  2. the masked region must NOT already have the requested attribute.

`score.py --expect` checks rule 2 and marks a vacuous target rather than
reporting a meaningless PASS.

WHAT IS AND IS NOT VERIFIED HERE. `guidance.py`, `sampler.py`, `backends.py` and
`score.py` are verified on CPU (`--self-test` on each). The diffusers side is
NOT: the box this was written on has stock diffusers 0.36.0, which has no
`StableDiffusion3InstructPix2PixPipeline` at all. So `--dry-run` validates
everything that can be validated without the fork -- arguments, mask polarity,
mask->latent pooling, term construction -- and `backends.guided_scheduler`
carries the runtime check that matters (whether the solver's step really is
`sample + ds*v`, which the guidance algebra assumes).

WHY THERE IS NO TOY BACKEND. A fake CPU backend would exercise the argument
plumbing and none of the real risk (scheduler semantics, sigma conventions, VAE
conventions), which is exactly the adjacent-path test STAGE1_REPORT SS7 records
as having cost this project time. `--dry-run` plus the toy demo cover the parts
that are genuinely shared; the rest is labelled unverified rather than papered
over with a mock.

SEEDS. The seed is fixed across a sweep, so every setting sees the SAME initial
noise and the comparison isolates the guidance. This is the same discipline the
Stage 0 strength sweep needed (one generator advanced across the sweep would
have compared different modes at different strengths).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: What THIS entry point imports, and nothing more. A bare
#: `ModuleNotFoundError: No module named 'short_term.sampler'` is what a PARTIAL
#: file sync looks like from Python -- and because a directory without
#: __init__.py is still importable as a namespace package, the missing file's
#: siblings import fine and the message points at the wrong thing.
#:
#: ONLY the four edit.py itself imports. This list used to name every sibling
#: file "for completeness", which made edit.py REFUSE TO RUN on a box that was
#: missing lab.py or cache_item.py -- files edit.py never touches. That is a
#: false failure, and it shows up as `exit 1` with no other trace, which is much
#: worse than the ModuleNotFoundError it was trying to prevent. The other files
#: are listed by --doctor, which is where "is my copy complete" belongs.
_REQUIRED = ("__init__.py", "guidance.py", "sampler.py", "backends.py")

#: Everything the package ships. --doctor lists these, so a partial copy is
#: still visible; they are NOT a precondition for running an edit.
_SHIPPED = _REQUIRED + ("edit.py", "demo_toy.py", "propose_y.py", "app.py",
                        "lab.py", "score.py", "cache_item.py", "suite.py",
                        "subset_cache.py", "page_harness.js", "README.md")


def _check_sync() -> None:
    here = Path(__file__).resolve().parent
    missing = [n for n in _REQUIRED if not (here / n).is_file()]
    if not missing:
        return
    have = sorted(p.name for p in here.glob("*.py"))
    raise SystemExit(
        "short_term/ is incompletely synced, so this failure is a copy "
        "problem, not a bug.\n"
        f"  directory : {here}\n"
        f"  missing   : {', '.join(missing)}  (edit.py imports these)\n"
        f"  present   : {', '.join(have) if have else '(no .py files)'}\n"
        "Copy the WHOLE short_term/ directory, including __init__.py -- "
        "without it the directory is still importable as a namespace package, "
        "which is why the siblings import and only the missing one fails.\n"
        "Verify with:  python short_term/edit.py --doctor")


def _doctor() -> int:
    """List every shipped file. Also the place to check a partial copy.

    Every file is listed even though only _REQUIRED blocks a run: knowing that
    lab.py is missing is useful, and it is NOT a reason to refuse to edit an
    image (that mistake made this script exit 1 on a box where the model, the
    weights and the inputs were all fine).
    """
    here = Path(__file__).resolve().parent
    print(f"short_term/ at {here}")
    print(f"{'file':<16}{'bytes':>8}  sha256[:12]")
    worst = 0
    for name in _SHIPPED:
        p = here / name
        if not p.is_file():
            # missing but not required -> a warning, not a failure
            note = "MISSING" if name in _REQUIRED else "missing (optional)"
            print(f"{name:<16}{note:>8}")
            worst = worst or int(name in _REQUIRED)
            continue
        blob = p.read_bytes()
        print(f"{name:<16}{len(blob):>8}  "
              f"{hashlib.sha256(blob).hexdigest()[:12]}")
    return worst


# `--doctor` first: it must work even when files are missing, so it cannot come
# after the package imports. Then the sync check, then the imports.
if "--doctor" in sys.argv:
    raise SystemExit(_doctor())
_check_sync()

from short_term.backends import (  # noqa: E402
    SD3IP2PEditor,
    load_image,
    load_mask,
    mask_to_latent,
    parse_size,
    save_image,
)
from short_term.guidance import GuidanceTerm  # noqa: E402
from short_term.sampler import kappa_schedule  # noqa: E402


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


#: Printed when the required arguments are missing. Uses files that are IN THIS
#: REPO at the root, so it can be pasted and run as-is -- the previous version
#: of this doc used `...` as a placeholder, which is not shell syntax and
#: produced a bare argparse error instead of an example.
USAGE_HINT = """\
this CLI needs --image and --instruction. To check the plumbing with no files
and no model at all, run:

    python short_term/edit.py --smoke

A complete tier-D dry run, using the synthetic files already in this repo:

    python short_term/edit.py --dry-run \\
        --image synthetic_src.png --mask synthetic_mask.png \\
        --guide-image synthetic_tgt.png \\
        --instruction "change the red circle to blue" --guide target \\
        --beta "1 3 10"

The same without --dry-run needs the UltraEdit diffusers fork and a GPU, and
downloads the checkpoint. Order matters -- see short_term/README.md section 4:
tier A baseline first, then beta, then kappa, then score on Gate C.
"""


def _smoke_inputs(size) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic source image and a centred square mask, in memory.

    Exists so that "is the plumbing working on this box" is one command with no
    file dependencies: `python short_term/edit.py --smoke`. `size` is WxH, so
    `--smoke --size 512x384` exercises the non-square path with no files at all.
    """
    w, h = parse_size(size)
    gx = np.linspace(0.0, 1.0, w, dtype=np.float32)
    gy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    img = np.stack([np.tile(gx, (h, 1)),
                    np.tile(gy[:, None], (1, w)),
                    np.full((h, w), 0.4, dtype=np.float32)], axis=-1)
    m = np.zeros((h, w), dtype=np.float32)
    qy, qx = h // 4, w // 4
    m[qy:3 * qy, qx:3 * qx] = 1.0
    return img, m


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tier A / tier D local editor.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=None, help="source image")
    ap.add_argument("--instruction", default=None,
                    help='e.g. "change the red circle to blue" -- must name '
                         'something that exists in the image, and the masked '
                         'region must NOT already be the requested attribute '
                         '(see short_term/README.md section 4.1)')
    ap.add_argument("--smoke", action="store_true",
                    help="generate a synthetic image and mask in memory and run "
                         "the structural checks. No files, no model, no GPU -- "
                         "this is the 'is the plumbing working on this box' "
                         "command.")
    ap.add_argument("--mask", default="white",
                    help="edit mask, white = editable (the project convention). "
                         "The literal 'white' means editable everywhere.")
    ap.add_argument("--out", default="runs/st/edit.png",
                    help="output STEM. The guide/beta/kappa tag is appended, so "
                         "a sweep cannot overwrite itself "
                         "(e.g. edit_both_b5_kin0.6_kout1_flat.png). Use "
                         "--out-exact for the literal path instead.")
    ap.add_argument("--out-exact", action="store_true",
                    help="treat --out as the literal file path, with no "
                         "parameter tag. Required by any caller that needs to "
                         "find the one file it asked for; only safe when one "
                         "beta/kappa point is being run.")
    ap.add_argument("--size", default="512",
                    help="output frame, WxH ('512x384' for 4:3) or a single "
                         "number for square. Width first, like patch_rect.")
    ap.add_argument("--latent-size", default=None,
                    help="latent grid WxH, overriding the default size/8. SD3 "
                         "at 512x384 is 64x48. Only shapes the guidance terms, "
                         "so it matters for --dry-run and for the mask warp.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="BleachNick/SD3_UltraEdit_w_mask")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--image-guidance", type=float, default=1.5,
                    help="the pipeline's own image_guidance_scale. Note this is "
                         "ALREADY a primitive guidance-toward-the-source, which "
                         "is why tier D is an extension of the house style "
                         "rather than a new idea.")
    ap.add_argument("--guidance", type=float, default=7.5)

    g = ap.add_argument_group("tier D -- guidance")
    g.add_argument("--guide", choices=("none", "target", "source", "both"),
                   default="none",
                   help="none = tier A. target = towards --guide-image, inside "
                        "the mask. source = towards the source image; with "
                        "--invert-weight that protects the background OUTSIDE "
                        "the mask. both = BOTH terms at once: inside -> "
                        "--guide-image, outside -> source. 'both' is the "
                        "configuration the measurements call for: the two terms "
                        "fix opposite halves (protect: background 0.447->0.020 "
                        "but edit still wrong; target: edit fixed, saturation "
                        "0.28->0.93, but background destroyed 0.444) and "
                        "neither alone fixes both.")
    g.add_argument("--guide-image", default=None,
                   help="reference image for --guide target")
    g.add_argument("--beta", default="5.0",
                   help="gain cap of equation (2). Accepts a list to sweep, "
                        "e.g. --beta '1 3 10'. Note the uncapped gain has a "
                        "MINIMUM of 2 (at t=0.5), so beta<=2 is constant on the "
                        "whole trajectory and the schedule has no content.")
    g.add_argument("--sign", type=float, default=1.0, choices=(1.0, -1.0),
                   help="+1 pulls the clean estimate toward the target, -1 "
                        "pushes away (e.g. to force change inside the mask).")
    g.add_argument("--kappa", default="0.0",
                   help="direct-blend form: blend the clean estimate toward Y "
                        "by kappa inside the mask. 0 = use the velocity form of "
                        "equation (2) instead. Accepts a list to sweep.")
    g.add_argument("--kappa-out", default=None,
                   help="with --guide both, a separate kappa for the OUTSIDE "
                        "(protection) term. Accepts a list parallel to --kappa, "
                        "or one value reused for every point. Worth separating "
                        "because the two saturate at different strengths.")
    g.add_argument("--kappa-t-lo", type=float, default=0.15,
                   help="no guidance below this t; the direct form's noise "
                        "recovery divides by t, so it must vanish near the "
                        "data end.")
    g.add_argument("--kappa-mode", choices=("flat", "ramp"), default="flat",
                   help="'flat' applies kappa constantly above t_lo. 'ramp' "
                        "scales it by (t-t_lo)/(1-t_lo), so a nominal kappa of "
                        "1.0 is really 0.41 at t=0.5. DEFAULT IS FLAT because "
                        "measured: on a real MagicBrush photo the ramp's "
                        "background change was 0.396-0.403 (contrast 1.09, i.e. "
                        "no localisation) against flat's 0.0196-0.0222 "
                        "(contrast 19.7-21.3) -- an 18x difference at the SAME "
                        "nominal kappa. The ramp looked harmless on the "
                        "synthetic two-disc fixture (0.0243 vs 0.0255) because "
                        "that fixture's flat white background cannot change "
                        "anyway; it hid the effect completely. Use 'ramp' only "
                        "to reproduce an older run.")
    g.add_argument("--invert-weight", action="store_true",
                   help="use 1 - mask as W (act outside the mask)")
    g.add_argument("--hard-mask", action="store_true",
                   help="threshold the mask at 0.5 instead of area-averaging. "
                        "Soft is the default because a hard boundary in a "
                        "guidance weight is the hard blend that produces seams.")
    g.add_argument("--jacobian", choices=("identity", "vjp"), default="identity",
                   help="'vjp' is exact but needs a differentiable velocity "
                        "function, which the callback path does not have; it is "
                        "rejected there with an explanation rather than "
                        "silently downgraded.")
    g.add_argument("--cfg-chunk", default=None,
                   help="OBSOLETE, accepted only so an old command line does "
                        "not break. Guidance now enters at scheduler.step, "
                        "whose model_output is already the CFG-COMBINED "
                        "velocity, so there is no chunk left to choose. See "
                        "short_term/README.md section 3.")

    ap.add_argument("--dry-run", action="store_true",
                    help="validate arguments, masks and terms; no model, no GPU")
    return ap


def _to_float01(img) -> np.ndarray:
    """Pipeline output (PIL image or array) -> (H, W, 3) float in [0, 1]."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def build_terms(args, editor, image, mask_px, h: int, w: int):
    """Assemble the guidance terms. Returns a list of specs, each
    {name, target, weight, desc}.

    Separation of concerns worth keeping: this function decides WHAT to steer
    toward and WHERE; `guidance.py` decides the algebra. Every guidance failure
    in this project's history so far has been the former.

    WHY IT RETURNS A LIST. `--guide both` applies TWO terms in one run, and the
    measurements say that is the only configuration that covers both halves:

        term        W        acts on   d_in    d_out_c  sat_in  exp_gain
        none        -        -         0.440   0.4474   0.281   -5.413
        source      1-mask   outside   0.424   0.0199   0.290   -5.182
        target      mask     inside    0.097   0.4444   0.926   +0.322
        (dataset truth)                0.070   0.0069   1.126   +1.000

    Protect fixes the background and leaves the edit wrong; target-guided fixes
    the edit (saturation 0.281 -> 0.926, change magnitude 0.440 -> 0.097 against
    the truth's 0.070) and leaves the background destroyed. They are orthogonal,
    so composing them is the obvious next measurement and the CLI has to be able
    to express it.
    """
    weight = mask_to_latent(mask_px, h, w, hard=args.hard_mask)
    if args.invert_weight:
        weight = 1.0 - weight

    def encode_array(arr, label):
        """For the SOURCE term: `image` is already a loaded (H,W,3) array.

        Calling load_image() on it is the bug this replaced -- the first
        `--guide both` run died with "'numpy.ndarray' object has no attribute
        'seek'" because PIL was handed an array instead of a path. Keeping the
        two helpers separate makes the distinction a signature instead of a
        comment.
        """
        if editor is None or arr is None:
            return None, label
        return editor.encode_image(arr), label

    def encode_path(path, label):
        """For the TARGET term: `--guide-image` is a path on disk."""
        if editor is None or path is None:
            return None, label
        return editor.encode_image(load_image(path, args.size)), label

    specs: list[dict] = []

    def add(name, target, w, source_label, weight_label):
        specs.append({
            "name": name, "target": target, "weight": w,
            "desc": {
                "guide": args.guide, "term": name,
                "target_from": source_label, "weight": weight_label,
                "hard_mask": bool(args.hard_mask), "sign": args.sign,
                "weight_mean": float(w.mean()),
                "weight_frac_gt_half": float((w > 0.5).float().mean()),
            },
        })

    if args.guide in ("source", "both"):
        tgt, label = encode_array(image, "source image")
        if args.guide == "both":
            # `both` sets its own weights: inside = mask, outside = 1 - mask.
            # That IS the flag, so --invert-weight would be nonsense here.
            add("outside", tgt, 1.0 - weight, label, "1-mask")
        else:
            add(args.guide, tgt, weight, label,
                "1-mask" if args.invert_weight else "mask")

    if args.guide in ("target", "both"):
        if not args.guide_image:
            raise SystemExit(
                f"--guide {args.guide} requires --guide-image (it is the "
                "content to steer the masked region toward)")
        tgt, label = encode_path(args.guide_image, args.guide_image)
        if args.guide == "both":
            add("inside", tgt, weight, label, "mask")
        else:
            add(args.guide, tgt, weight, label,
                "1-mask" if args.invert_weight else "mask")

    if args.guide == "both" and args.invert_weight:
        print("[edit] --guide both sets its own weights (inside = mask, "
              "outside = 1-mask); --invert-weight is ignored for it.")
    return specs


def _reject_vjp(args) -> None:
    """`--jacobian vjp` is exact but unavailable on the callback path. Say so
    rather than quietly using the identity form, and do NOT offer a
    finite-difference stand-in: the paper's expression is a VJP, so a single
    finite difference computes a JVP, which is a different vector."""
    if args.jacobian == "vjp" and args.guide != "none":
        raise SystemExit(
            "--jacobian vjp is not available through the callback path: the "
            "velocity is captured from the pipeline's own denoiser, so it is "
            "not a differentiable function of the state. Use the standalone "
            "loop in sampler.py (which does support it), or leave the default "
            "identity.")


def _self_test() -> int:
    """CPU, no model, no cache. Tests `build_terms`, which is the plumbing that
    has broken twice: once by hooking the wrong layer, once by handing PIL an
    array instead of a path.

    The decisive check: `encode_image` must be given a NUMPY ARRAY in both
    terms. If anything downstream calls load_image() on the already-loaded
    source, `Image.open(array)` raises "'numpy.ndarray' object has no attribute
    'seek'" -- which is the exact failure this test exists to prevent, and it
    only shows up on a GPU box minutes into a run otherwise.
    """
    import torch

    seen: list = []

    class _FakeEditor:
        def encode_image(self, arr):
            assert isinstance(arr, np.ndarray), (
                f"encode_image was handed {type(arr).__name__}, not an array; "
                "a path here means someone called load_image on the "
                "already-loaded source")
            seen.append(arr.shape)
            return torch.zeros(1, 4, 8, 8)

    editor = _FakeEditor()
    h = w = 8
    image = np.zeros((64, 64, 3), np.float32)
    mask_px = np.zeros((64, 64), np.float32)
    mask_px[16:48, 16:48] = 1.0
    guide = Path("synthetic_tgt.png")   # a real image in this repo

    def ns(**kw):
        base = dict(guide="none", guide_image=None, size=64, hard_mask=False,
                    invert_weight=False, sign=1.0)
        base.update(kw)
        return argparse.Namespace(**base)

    # guide=none -> no terms at all
    assert build_terms(ns(), editor, image, mask_px, h, w) == []
    seen.clear()

    # source: one term, weight = the mask (1 in the masked block, 0 outside),
    # target from the in-memory array
    specs = build_terms(ns(guide="source"), editor, image, mask_px, h, w)
    assert len(specs) == 1 and specs[0]["name"] == "source"
    wgt = specs[0]["weight"]
    assert float(wgt[0, 0, 4, 4]) > 0.5 and float(wgt[0, 0, 0, 0]) < 0.5, \
        "W = mask must be ~1 at the centre of the block and ~0 at the corner"
    assert specs[0]["desc"]["weight"] == "mask" and len(seen) == 1

    # source + invert: weight = 1 - mask, and the mask is a square in the middle
    specs = build_terms(ns(guide="source", invert_weight=True), editor, image,
                        mask_px, h, w)
    wgt = specs[0]["weight"]
    assert specs[0]["desc"]["weight"] == "1-mask"
    assert float(wgt[0, 0, 4, 4]) < 0.5 and float(wgt[0, 0, 0, 0]) > 0.5, \
        "1-mask must be ~0 at the centre and ~1 at the corner"

    # both: TWO terms with opposite weights, inside=target outside=source
    specs = build_terms(ns(guide="both", guide_image=str(guide)), editor,
                        image, mask_px, h, w)
    names = [s["name"] for s in specs]
    assert names == ["outside", "inside"], names
    by = {s["name"]: s for s in specs}
    assert by["outside"]["desc"]["weight"] == "1-mask"
    assert by["inside"]["desc"]["weight"] == "mask"
    # and they partition the frame: W_out + W_in == 1 everywhere
    total = by["outside"]["weight"] + by["inside"]["weight"]
    assert torch.allclose(total, torch.ones_like(total)), \
        "the two weights must sum to 1, or some pixels get double-guided"
    assert by["inside"]["desc"]["target_from"] == str(guide)
    assert by["outside"]["desc"]["target_from"] == "source image"

    # target without --guide-image is refused, not silently built
    try:
        build_terms(ns(guide="target"), editor, image, mask_px, h, w)
    except SystemExit as exc:
        assert "guide-image" in str(exc)
    else:
        raise AssertionError("--guide target without --guide-image must raise")

    # dry-run path: editor=None must not touch the disk, and must not build a
    # GuidanceTerm (a term with target=None is rejected by the dataclass)
    specs = build_terms(ns(guide="both", guide_image=str(guide)), None, image,
                        mask_px, h, w)
    assert len(specs) == 2 and all(s["target"] is None for s in specs)

    # THE SYNC CHECK, BOTH WAYS. This is the regression for a real failure: the
    # required list was widened "for completeness" to every sibling file, so
    # edit.py refused to run on a box that was missing lab.py or cache_item.py
    # -- files it never imports -- and the UI reported only "exited 1 and no
    # result PNG was found", with the reason buried in a log panel.
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "short_term"
        root.mkdir()
        for name in _REQUIRED:
            shutil.copy(Path(__file__).parent / name, root / name)
        shutil.copy(Path(__file__), root / "edit.py")
        assert not (root / "lab.py").exists() and not (root / "app.py").exists()

        # (a) optional files missing -> the edit still RUNS
        r = subprocess.run([sys.executable, str(root / "edit.py"), "--smoke",
                            "--size", "96x72"], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout[-900:], r.stderr[-900:])
        assert "latent grid 12x9" in r.stdout, r.stdout[-500:]
        d = subprocess.run([sys.executable, str(root / "edit.py"), "--doctor"],
                           capture_output=True, text=True)
        assert d.returncode == 0, d.stdout[-500:]
        assert "missing (optional)" in d.stdout, d.stdout
        assert "lab.py" in d.stdout and "app.py" in d.stdout, d.stdout

        # (b) a file edit.py DOES import missing -> refuse, and say which
        (root / "sampler.py").unlink()
        r = subprocess.run([sys.executable, str(root / "edit.py"), "--smoke"],
                           capture_output=True, text=True)
        assert r.returncode != 0, r.stdout
        msg = r.stdout + r.stderr
        assert "incompletely synced" in msg and "sampler.py" in msg, msg
        assert "edit.py imports these" in msg, msg

    print("edit self-test OK")
    print("  build_terms gives numpy ARRAYS to encode_image in every mode")
    print("  (the 'numpy.ndarray has no attribute seek' bug)")
    print("  --guide both: 2 terms, W_out + W_in == 1 exactly")
    print("  --guide target without --guide-image is refused")
    print("  a partial sync: missing lab.py/app.py does NOT block an edit, but a")
    print("  missing sampler.py refuses and names the file")
    return 0


def _validate(args) -> None:
    """Argument-level conflicts, checked BEFORE --dry-run/--smoke return.

    `--dry-run` exists to validate arguments, so a check that lives after its
    early return is a check that never runs -- which is how `--out-exact` plus a
    sweep, a combination that silently overwrites files, passed a dry run.
    """
    if args.out_exact:
        n = len(_floats(args.beta)) * len(_floats(args.kappa))
        if n > 1:
            raise SystemExit(
                f"--out-exact with {n} beta/kappa points would have all {n} "
                "runs write to the same file. Drop --out-exact for a sweep, or "
                "sweep one point at a time.")


def main(argv=None, editor=None) -> int:
    """`editor` is an optional pre-built SD3IP2PEditor, so a multi-item runner
    (suite.py) loads the 2B checkpoint ONCE instead of once per run. Reuse is
    safe: every call makes a fresh generator, and guided_scheduler restores
    scheduler.step in a finally."""
    args = build_parser().parse_args(argv)
    _reject_vjp(args)
    _validate(args)
    if args.cfg_chunk is not None:
        print("[edit] --cfg-chunk is obsolete and ignored: guidance now enters "
              "at scheduler.step, so model_output is already the CFG-combined "
              "velocity and there is no chunk to select.")
    if args.smoke:
        args.dry_run = True
        args.guide = "source"   # exercise the guidance path, not just tier A
        image, mask_px = _smoke_inputs(args.size)
        args.image = args.image or "<synthetic smoke image>"
        args.instruction = args.instruction or "<smoke>"
        print(f"[smoke] synthetic {args.size} image + centred mask; "
              "--guide forced to 'source' so the guidance term is built too. "
              "No model is loaded.")
    else:
        if not args.image or not args.instruction:
            print(USAGE_HINT)
            return 2
        image = load_image(args.image, args.size)
        mask_px = load_mask(args.mask, args.size)
    # The latent grid, PER AXIS. 512x384 is 64x48, so a single number cannot
    # describe a non-square frame -- and the guidance weights are shaped (h, w),
    # so getting this wrong is a broadcast error deep inside the transformer.
    # `h, w` below are the LATENT (rows, cols) in numpy order; `fw, fh` are the
    # frame in WxH order, which is the opposite. That is the whole reason
    # parse_size() exists.
    fw, fh = parse_size(args.size)
    if args.latent_size:
        lw, lh = parse_size(args.latent_size)
    else:
        if fw % 8 or fh % 8:
            raise SystemExit(
                f"--size {fw}x{fh} is not a multiple of 8, so the VAE's latent "
                f"grid would be fractional. SD3 works on /8: use e.g. 512x384. "
                f"Pass --latent-size to override if you know better.")
        lw, lh = fw // 8, fh // 8
    h, w = lh, lw
    # `editor` may arrive pre-built from a batch runner; only build it here when
    # nobody gave us one.
    if not args.dry_run and editor is None:
        editor = SD3IP2PEditor(args.model, device=args.device, steps=args.steps,
                               image_guidance=args.image_guidance,
                               guidance=args.guidance)

    if args.guide == "none":
        terms, descs = [], [{"guide": "none"}]
    elif args.dry_run:
        # No model: validate everything structural, report what needs the VAE.
        specs = build_terms(args, None, image, mask_px, h, w)
        terms = []
        descs = [s["desc"] for s in specs]
        for d in descs:
            d["note"] = "dry run: Y not built (needs the model's VAE)"
    else:
        specs = build_terms(args, editor, image, mask_px, h, w)
        terms = [GuidanceTerm(beta=1.0, weight=s["weight"], target=s["target"],
                              sign=args.sign, jacobian=args.jacobian,
                              name=s["name"]) for s in specs]
        descs = [s["desc"] for s in specs]

    kappa_out = _floats(args.kappa_out) if args.kappa_out else None
    betas = _floats(args.beta)
    kappas = _floats(args.kappa)
    plan = {
        "model": args.model, "image": args.image, "mask": args.mask,
        "instruction": args.instruction, "size": list(parse_size(args.size)),
        "size_WxH": f"{fw}x{fh}", "seed": args.seed,
        "steps": args.steps, "image_guidance": args.image_guidance,
        "guidance": args.guidance, "latent_grid": [h, w],
        "mask_px_mean": float(mask_px.mean()),
        "mask_latent_mean": float(mask_to_latent(mask_px, h, w).mean()),
        "betas": betas, "kappas": kappas, "kappa_out": kappa_out,
        "terms": descs if len(descs) > 1 else descs[0],
    }
    print(json.dumps(plan, indent=1))

    if args.dry_run:
        print("\n[dry-run] no model loaded. Structural checks:")
        print(f"  frame {image.shape[1]}x{image.shape[0]} (WxH), "
              f"image array {image.shape}, mask {mask_px.shape}")
        print(f"  latent grid {w}x{h} (WxH) = {image.shape[1]//8}x"
              f"{image.shape[0]//8} at /8")
        if args.guide != "none" and not (0.0 < plan["mask_latent_mean"] < 1.0):
            print("  WARNING: the mask is all-zero or all-one at the latent "
                  "grid. That is either the 'white' control arm or a mistake; "
                  "check --mask and --size.")
        if args.guide == "none":
            print("  tier A: guidance off. Run this FIRST -- without it there "
                  "is no same-convention baseline to compare tier D against.")
        else:
            print(f"  tier D: {len(betas) * len(kappas)} run(s) planned; "
                  f"seed fixed at {args.seed} across them so every setting "
                  "sees the same initial noise.")
        return 0

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out).stem

    for beta in betas:
      for ki, kappa in enumerate(kappas):
        for term in terms:
            term.beta = beta
        # Index the parallel --kappa-out list by POSITION, not by searching for
        # the kappa value: duplicate entries in --kappa would then both match
        # the first index and silently share one outside strength.
        k_out = (kappa_out[min(ki, len(kappa_out) - 1)]
                 if kappa_out else kappa)
        if kappa <= 0:
            kfn = None
        elif not terms or kappa_out is None:
            kfn = kappa_schedule(args.kappa_mode, kappa, t_lo=args.kappa_t_lo)
        else:
            # Per-term kappa: `renoise` already calls direct_kappa(t, term), so
            # the inside and outside pulls can differ. That matters because they
            # saturate at different strengths (protection is flat from 0.6, the
            # inside pull is still moving at 1.0).
            kf_in = kappa_schedule(args.kappa_mode, kappa,
                                   t_lo=args.kappa_t_lo)
            kf_out = kappa_schedule(args.kappa_mode, k_out,
                                    t_lo=args.kappa_t_lo)

            def kfn(t, term, _a=kf_in, _b=kf_out):
                fn = _a if term.name.startswith("inside") else _b
                return fn(t, term)
        # The mode is ALWAYS in the filename, and with two terms so is the
        # per-term strength. They used to be distinguishable only by their
        # numbers, and the forms differ by up to 20x -- exactly the field a
        # reader needs in the name.
        ktag = (f"kin{kappa:g}_kout{k_out:g}" if kappa_out
                else f"k{kappa:g}")
        tag = f"b{beta:g}_{ktag}_{args.kappa_mode}"
        # `--out` has always been a STEM, not a path: the parameter tag is
        # appended so a sweep cannot overwrite itself. That is right for a
        # sweep and wrong for a caller that wants to find the one file it just
        # asked for -- which is how short_term/app.py looked for `out.png`,
        # found nothing, and reported "no output was written" for a run that
        # had exited 0 and written an image.
        path = (Path(args.out) if args.out_exact
                else out_dir / f"{stem}_{args.guide}_{tag}.png")
        rec = dict(plan)
        rec.update({"beta": beta, "kappa": kappa, "kappa_out_used": k_out,
                    "out": str(path)})
        if not terms:
            img = editor.edit(image, args.instruction, mask_px, seed=args.seed)
            rec["mode"] = "tier A (no guidance)"
        else:
            img, state = editor.edit_guided(
                image, args.instruction, mask_px, terms, kappa_fn=kfn,
                seed=args.seed)
            rec["mode"] = "tier D (guided)"
            rec["guided_steps"] = state["steps"]
            rec["scheduler_dev"] = state["max_dev"]
            rec["scheduler_last_dev"] = state["last_dev"]
            rec["latent_ch"] = state["lat_ch"]
        save_image(_to_float01(img), path)
        Path(path).with_suffix(".json").write_text(json.dumps(rec, indent=1))
        print(f"wrote {path}")

    print("\nNEXT, and it is NOT Gate C yet:"
          "\n  1. score locally, on the honest background number and the "
          "requested-attribute number:"
          "\n       python short_term/score.py --src ... --mask ... \\"
          "\n           --out 'runs/st/*.png' --expect <the colour you asked "
          "for> --rank"
          "\n     read d_out_content (not d_out_all) and expect_gain (not "
          "d_inside)."
          "\n  2. re-run the winning kappa under --kappa-mode flat, because "
          "'ramp' scales it down over most of the trajectory and a plateau "
          "under ramp may just be the ramp."
          "\n  3. Gate C (bench/run_gate_c.py) is a 151-case scoreboard over "
          "REAL cases with per-case masks and instructions. It measures "
          "instance routing (mis_targeted) and edit quality (blind-audit "
          "clean). The synthetic fixtures are not in it, and a two-disc image "
          "with no instances cannot answer either question -- running it now "
          "would spend GPU hours on a question the local scorer already "
          "answers. Go there when the case is a real one with a real "
          "sibling instance to be mis-targeted.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
