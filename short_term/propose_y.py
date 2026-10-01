#!/usr/bin/env python3
"""
Line 2: propose Y from a text instruction, so an instruction-only edit can use
the guidance that is already measured to work.

THE TWO PRODUCT LINES, AND WHY THIS IS THE SECOND ONE.

    line 1  the user supplies a reference image            PROVEN
            edit.py --guide both --guide-image ref.png
            measured: d_in 0.075 against the dataset truth's 0.070 (1.07x),
            sat_in 1.053 against 1.126, background 0.0166, exp_gain +0.788

    line 2  the user supplies only text                    UNVALIDATED
            propose_y.py -> y.png, then the SAME edit.py command
            The mechanism is sufficient (line 1); what is missing is a Y. This
            script manufactures one with a text-to-image model.

WHAT "PROPOSE Y" HAS TO GET RIGHT, AND WHAT IT CANNOT. Y only matters inside
the mask, because the guidance weight is the mask -- so this script ALWAYS ends
by blending with the mask, making `Y == source` outside it exactly. That is not
a detail: if Y differed from the source outside the mask, the protect term and
the inside term would be pulling against each other, and the result would be
neither.

What it cannot get right is alignment. A balloon generated at 512x512 is not
already placed where this photograph needs it, so the generated image has to be
placed -- see --strategy. Every strategy here is a guess, which is why
`--from-image` and `--variant` exist: they let you measure how good Y has to be
BEFORE trusting any of the strategies. If a half-size misalignment already
destroys exp_gain, then no placement heuristic will save line 2 and the answer
is to train a Y-proposer (tier B).

    # line 2, the full path
    python short_term/propose_y.py \
        --image runs/st/real/src.png --mask runs/st/real/mask.png \
        --instruction "A hot air balloon should also be floating." \
        --y-prompt "a colourful hot air balloon floating in a blue sky" \
        --out runs/st/y.png
    # then, the same command line 1 uses:
    python short_term/edit.py \
        --image runs/st/real/src.png --mask runs/st/real/mask.png \
        --guide-image runs/st/y.png \
        --instruction "A hot air balloon should also be floating." \
        --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat \
        --out runs/st/y_edit.png

    # the calibration run: how good must Y be? (uses the dataset truth as the
    # content, so the ONLY variable is the degradation)
    python short_term/propose_y.py --image .../src.png --mask .../mask.png \
        --from-image .../tgt.png --variant blur:4  --out runs/st/y_blur4.png
    python short_term/propose_y.py ... --variant shift:24 --out runs/st/y_shift24.png
    python short_term/propose_y.py ... --variant solid    --out runs/st/y_solid.png

    python short_term/propose_y.py --self-test      # CPU, no model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term.backends import load_image, load_mask, save_image  # noqa: E402

#: How the generated content is placed. Every one of these is a guess about
#: alignment; none has been measured against a real generated image yet.
STRATEGIES = ("bbox", "center", "region", "full")

#: Degradations, for measuring how good Y has to be. `shift` and `scale` model
#: the alignment errors a generated Y actually has; `blur`/`noise` model
#: quality loss; `solid` is the "Y says nothing" control and should reproduce
#: the no-guidance result.
VARIANTS = ("none", "blur", "noise", "shift", "scale", "solid")


def mask_bbox(mask_px: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    """(H,W) mask -> (y0, x0, y1, x1) of its bounding box, padded and clipped.

    Clipped, not wrapped: a padded box that ran past the edge would need
    padding logic at the composite site, and silently wrapping it into the
    other side of the frame is the kind of bug that looks like a model failure.
    """
    m = np.asarray(mask_px) > 0.5
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        raise SystemExit("the mask is empty; there is no region to propose Y for")
    h, w = m.shape
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    return y0, x0, y1, x1


def _resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    h, w = hw
    a = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    out = Image.fromarray((a * 255 + 0.5).astype(np.uint8)).resize(
        (w, h), Image.LANCZOS)
    return np.asarray(out, np.float32) / 255.0


def _center_crop(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Centre-crop (or pad with edge pixels) to exactly (h, w)."""
    h, w = hw
    H, W = img.shape[:2]
    out = img
    if H > h:
        t = (H - h) // 2
        out = out[t:t + h]
    elif H < h:
        out = np.pad(out, ((0, h - H), (0, 0), (0, 0)), mode="edge")
    if out.shape[1] > w:
        l = (out.shape[1] - w) // 2
        out = out[:, l:l + w]
    elif out.shape[1] < w:
        out = np.pad(out, ((0, 0), (0, w - out.shape[1]), (0, 0)),
                     mode="edge")
    return out


def degrade(img: np.ndarray, variant: str, seed: int = 0) -> np.ndarray:
    """Apply a named degradation to the CONTENT before it is placed."""
    if not variant or variant == "none":
        return img
    name, _, arg = variant.partition(":")
    val = float(arg) if arg else 0.0
    if name == "solid":
        return np.full_like(img, 0.5)
    if name == "blur":
        from PIL import Image, ImageFilter
        a = np.clip(img, 0, 1)
        im = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
        im = im.filter(ImageFilter.GaussianBlur(max(val, 0.1)))
        return np.asarray(im, np.float32) / 255.0
    if name == "noise":
        rng = np.random.default_rng(seed)
        return np.clip(img + rng.normal(0.0, max(val, 1e-3), img.shape),
                       0.0, 1.0)
    if name == "shift":
        return np.roll(img, int(round(val)), axis=1)
    if name == "scale":
        f = max(val, 1e-3)
        H, W = img.shape[:2]
        return _center_crop(_resize(img, (max(1, int(H * f)),
                                          max(1, int(W * f)))), (H, W))
    raise SystemExit(f"unknown --variant {variant!r}; expected one of "
                     + ", ".join(f"{v}[:value]" for v in VARIANTS if v != "none"))


def content_canvas(content: np.ndarray, shape: tuple[int, int],
                   bbox: tuple[int, int, int, int],
                   strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Place `content` on a zero canvas of `shape`. -> (canvas, footprint).

    `footprint` is 1 wherever the content was actually pasted. It exists because
    "where the patch IS" and "where the mask is" are different questions once
    the user places the patch themselves, and the caller needs to be able to
    tell them apart (see `compose_y`'s `clip_to_mask`).
    """
    H, W = shape
    y0, x0, y1, x1 = bbox
    bh, bw = y1 - y0, x1 - x0
    canvas = np.zeros((H, W, 3), np.float32)
    foot = np.zeros((H, W), np.float32)
    if strategy == "full":
        canvas = _resize(content, (H, W))
        foot[:] = 1.0
        return canvas, foot
    if strategy == "bbox":
        patch = _resize(content, (bh, bw))
    elif strategy == "center":
        # centre-crop to the box's aspect FIRST, so the object is not squashed
        ar = bw / max(bh, 1)
        Hc, Wc = content.shape[:2]
        if Wc / max(Hc, 1) > ar:
            patch = _center_crop(content, (Hc, max(1, int(round(Hc * ar)))))
        else:
            patch = _center_crop(content, (max(1, int(round(Wc / ar))), Wc))
        patch = _resize(patch, (bh, bw))
    elif strategy == "region":
        patch = _center_crop(content, (bh, bw))
    else:
        raise SystemExit(f"unknown --strategy {strategy!r}; expected one of "
                         + ", ".join(STRATEGIES))
    canvas[y0:y1, x0:x1] = patch
    foot[y0:y1, x0:x1] = 1.0
    return canvas, foot


def place_rect(shape: tuple[int, int], content_shape: tuple[int, ...],
               rect: tuple[int, int, int, int], fit: str = "contain"
               ) -> tuple[int, int, int, int]:
    """Where the patch actually lands, inside the rect the user asked for.

    `fit="contain"` scales the patch to fit INSIDE the rect with its aspect
    preserved and centres it; `fit="stretch"` fills the rect exactly. Contain is
    the default because a dragged rect and a typed `(x, y, w, h)` do not have to
    agree on aspect, and silently stretching a face into a wide box is not a
    thing anyone asks for.

    This is returned rather than recomputed so that the CONTENT and the MASK can
    be placed by the same numbers. Placing them by two separate calculations is
    exactly how a mask and its content drift apart.
    """
    H, W = shape
    x, y, w, h = (int(round(v)) for v in rect)
    x0 = max(0, min(W - 1, x))
    y0 = max(0, min(H - 1, y))
    x1 = max(x0 + 1, min(W, x + w))
    y1 = max(y0 + 1, min(H, y + h))
    bw, bh = x1 - x0, y1 - y0
    if fit == "stretch":
        return x0, y0, bw, bh
    if fit != "contain":
        raise SystemExit(f"--fit must be 'contain' or 'stretch', got {fit!r}")
    ch, cw = content_shape[0], content_shape[1]
    s = min(bw / max(cw, 1), bh / max(ch, 1))
    dw = max(1, min(bw, int(round(cw * s))))
    dh = max(1, min(bh, int(round(ch * s))))
    return x0 + (bw - dw) // 2, y0 + (bh - dh) // 2, dw, dh


def _to_pil(a: np.ndarray):
    from PIL import Image
    arr = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), mode="L")
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8))


def _from_pil_G(im) -> np.ndarray:
    return np.asarray(im.convert("L"), np.float32) / 255.0


def grow_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Dilate with a max filter, in pixels. radius<=0 is a no-op."""
    if radius <= 0:
        return np.asarray(mask, np.float32)
    from PIL import ImageFilter
    k = int(round(2 * radius)) | 1          # MaxFilter needs an odd size
    return _from_pil_G(_to_pil(mask).filter(ImageFilter.MaxFilter(k)))


def feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur the mask edge, in pixels. sigma<=0 is a no-op."""
    if sigma <= 0:
        return np.asarray(mask, np.float32)
    from PIL import ImageFilter
    return _from_pil_G(_to_pil(mask).filter(ImageFilter.GaussianBlur(sigma)))


def carry_mask_onto_source(patch_mask: np.ndarray, shape: tuple[int, int],
                           dst: tuple[int, int, int, int]) -> np.ndarray:
    """The patch's own mask, warped into the source frame by the placement.

    THIS IS THE UNIFICATION. The user marks the OBJECT inside the patch (a face
    on a busy crop), places and scales the patch, and this carries that marking
    to the source with the same scale and translation the content got. So the
    region that is edited and the content that is pulled into it are the same
    region by construction -- and the patch's own background never enters, which
    is what a hard pasted rectangle looks like when it does.
    """
    H, W = shape
    x, y, w, h = dst
    out = np.zeros((H, W), np.float32)
    pm = _resize(np.repeat(np.asarray(patch_mask, np.float32)[..., None], 3, 2)
                 if np.asarray(patch_mask).ndim == 2
                 else np.asarray(patch_mask, np.float32), (h, w))
    out[y:y + h, x:x + w] = pm[..., 0] if pm.ndim == 3 else pm
    return out


def compose_patch(source: np.ndarray, patch: np.ndarray,
                  patch_mask: np.ndarray | None,
                  rect: tuple[int, int, int, int],
                  fit: str = "contain", grow: float = 0.0,
                  feather: float = 0.0,
                  refine_mask: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """-> (Y, source_mask), built from ONE placement.

    Y is the source with the patch's masked content placed; source_mask is the
    patch's mask carried to the source frame. Both come from `place_rect`, so
    they cannot disagree.

    With `patch_mask=None` the mask falls back to the patch's whole footprint
    (fine for a tightly cropped patch, wrong for one with a background).

    `refine_mask` is a THREE-STATE image in source coordinates, applied AFTER
    the derived mask: ~1 forces a pixel ON (add a shadow), ~0 forces it OFF
    (the patch brought in more than you want), and the middle value abstains.
    It exists because the derived mask is right about WHERE the object goes but
    is only as good as the marking, and the marking is done on a small panel --
    touching it up on the full-size source is a different, easier judgement.
    The base stays derived, so mask and content cannot drift apart.
    """
    H, W = source.shape[:2]
    dst = place_rect((H, W), patch.shape, rect, fit=fit)
    x, y, w, h = dst
    content = _resize(patch, (h, w))
    canvas = np.zeros((H, W, 3), np.float32)
    canvas[y:y + h, x:x + w] = content
    foot = np.zeros((H, W), np.float32)
    foot[y:y + h, x:x + w] = 1.0

    if patch_mask is None:
        alpha = foot
    else:
        alpha = carry_mask_onto_source(patch_mask, (H, W), dst)

    if refine_mask is not None:
        r = np.asarray(refine_mask, np.float32)
        if r.shape != (H, W):
            raise SystemExit(f"refine_mask is {r.shape}, expected {(H, W)}")
        alpha = np.where(r > 0.75, 1.0, np.where(r < 0.25, 0.0, alpha))
    if grow > 0:
        # Dilate the MASK, not just the footprint: "grow" should widen whatever
        # region you actually decided on, including a touch-up stroke.
        alpha = grow_mask(alpha, grow)
    if feather > 0:
        alpha = feather_mask(alpha, feather)
    alpha = np.clip(alpha, 0.0, 1.0)
    y_img = source * (1.0 - alpha[..., None]) + canvas * alpha[..., None]
    return y_img, alpha


def compose_y(source: np.ndarray, content: np.ndarray, mask_px: np.ndarray,
              strategy: str = "bbox", pad: int = 0,
              soft: bool = True,
              rect: tuple[int, int, int, int] | None = None,
              clip_to_mask: bool = True) -> np.ndarray:
    """The proposed Y.

    `rect` = (x, y, w, h) in PIXELS, and it OVERRIDES the mask's bounding box:
    that is the "the user placed the patch themselves" path, which is the one a
    real user has -- they have a local crop of the thing they want and they put
    it where it goes. When the user places the patch, the placement is no longer
    a guess, so `--strategy` is not consulted.

    `clip_to_mask=True` (the default, for a GENERATED reference) blends with the
    mask so `Y == source` outside it exactly. `clip_to_mask=False` (for a
    USER-PLACED patch) keeps the whole paste, so the preview shows what was
    actually placed -- and it is safe, because the inside guidance term is
    weighted by the mask anyway, so anything outside contributes nothing. Use
    `patch_outside_mask()` to warn when the two disagree by a lot.
    """
    if rect is not None:
        x, y, w, h = (int(round(v)) for v in rect)
        H, W = source.shape[:2]
        x0 = max(0, min(W - 1, x))
        y0 = max(0, min(H - 1, y))
        x1 = max(x0 + 1, min(W, x + w))
        y1 = max(y0 + 1, min(H, y + h))
        bbox = (y0, x0, y1, x1)
        strat = "bbox"
    else:
        bbox = mask_bbox(mask_px, pad)
        strat = strategy
    canvas, foot = content_canvas(content, source.shape[:2], bbox, strat)
    m = np.asarray(mask_px, np.float32)
    if not soft:
        m = (m > 0.5).astype(np.float32)
    alpha = (m * foot) if clip_to_mask else foot
    return source * (1.0 - alpha[..., None]) + canvas * alpha[..., None]


def patch_outside_mask(mask_px: np.ndarray,
                       rect: tuple[int, int, int, int]) -> float:
    """Fraction of the placed patch rect that falls outside the mask.

    Not an error: a user may deliberately place a patch larger than the mask
    (e.g. to keep the guidance from touching a boundary). But if it is large,
    the mask is not doing what they think, and that should be visible rather
    than silent.
    """
    x, y, w, h = (int(round(v)) for v in rect)
    H, W = mask_px.shape[:2]
    inside = np.zeros((H, W), bool)
    x0 = max(0, min(W, x))
    y0 = max(0, min(H, y))
    x1 = max(x0, min(W, x + w))
    y1 = max(y0, min(H, y + h))
    inside[y0:y1, x0:x1] = True
    n = int(inside.sum())
    if n == 0:
        return 1.0
    return float((inside & ~(np.asarray(mask_px) > 0.5)).sum()) / n


def generate_t2i(prompt: str, model: str, device: str, size: int, seed: int,
                 steps: int, guidance: float, negative: str = ""):
    """Text -> image, with the stock SD3 text-to-image pipeline.

    STOCK, not the UltraEdit fork: this is generation, not editing, and the
    fork is only needed for the InstructPix2Pix class. `stabilityai/
    stable-diffusion-3-medium-diffusers` is the model SETUP.md already
    downloads.
    """
    import torch
    try:
        from diffusers import StableDiffusion3Pipeline
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "StableDiffusion3Pipeline not found -- it is in STOCK diffusers, "
            "so this means the install is broken rather than the fork missing."
        ) from exc
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(model,
                                                    torch_dtype=dtype).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(prompt=prompt, num_inference_steps=steps,
                  guidance_scale=guidance, height=size, width=size,
                  generator=gen)
    if negative:
        kwargs["negative_prompt"] = negative
    return pipe(**kwargs).images[0]


def _self_test() -> int:
    """CPU, no model, no GPU. The composite is the part with the invariant, so
    that is what gets pinned."""
    h = w = 64
    rng = np.random.default_rng(0)
    src = rng.random((h, w, 3)).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    mask[16:48, 24:40] = 1.0            # deliberately NOT centred, and not square
    content = rng.random((h, w, 3)).astype(np.float32)

    # bbox: exact, padded, clipped, and empty-mask is refused
    assert mask_bbox(mask) == (16, 24, 48, 40), mask_bbox(mask)
    assert mask_bbox(mask, pad=4) == (12, 20, 52, 44), mask_bbox(mask, 4)
    edge = np.zeros((h, w), np.float32)
    edge[:2, :2] = 1.0
    assert mask_bbox(edge, pad=8) == (0, 0, 10, 10), mask_bbox(edge, 8)
    try:
        mask_bbox(np.zeros((h, w), np.float32))
    except SystemExit as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("an empty mask must be refused")

    # THE INVARIANT: outside the mask Y is the source EXACTLY, for every
    # strategy. If this breaks, the inside term and the protect term fight.
    for strat in STRATEGIES:
        y = compose_y(src, content, mask, strategy=strat, pad=4)
        assert y.shape == src.shape, (strat, y.shape)
        outside = mask < 0.5
        assert np.array_equal(y[outside], src[outside]), \
            f"{strat}: Y changed pixels outside the mask"
        inside = mask > 0.5
        assert not np.allclose(y[inside], src[inside]), \
            f"{strat}: Y equals the source inside the mask, so it proposes nothing"
        assert np.isfinite(y).all()

    # a hard mask must give a strictly binary composite: the canvas inside, the
    # source outside, nothing in between. (Comparing against the raw `content`
    # would be wrong -- `bbox` RESIZES the content into the box first.)
    canvas, foot = content_canvas(content, src.shape[:2], mask_bbox(mask),
                                  "bbox")
    assert foot.sum() > 0 and set(np.unique(foot)).issubset({0.0, 1.0})
    yh = compose_y(src, content, mask, soft=False)
    assert np.array_equal(yh[mask > 0.5], canvas[mask > 0.5]), \
        "with soft=False the masked region must be exactly the canvas"
    assert np.array_equal(yh[mask < 0.5], src[mask < 0.5]), \
        "with soft=False the outside must be exactly the source"

    # THE USER-PLACED-PATCH PATH, which is what a real user has: they drag a
    # local crop to where it goes, so the placement is given, not inferred.
    rect = (30, 8, 20, 24)                       # x, y, w, h in PIXELS
    yp = compose_y(src, content, mask, rect=rect, clip_to_mask=False)
    assert yp.shape == src.shape
    # the paste lands exactly on the rect, whatever the mask says
    assert not np.array_equal(yp[8:32, 30:50], src[8:32, 30:50])
    # ... and outside the rect nothing moved, because the footprint is the rect
    outside_rect = np.ones(src.shape[:2], bool)
    outside_rect[8:32, 30:50] = False
    assert np.array_equal(yp[outside_rect], src[outside_rect]), \
        "clip_to_mask=False must not touch anything outside the placed patch"
    # rect overrides the mask bbox: put it somewhere the mask is not
    assert not np.array_equal(yp[8:32, 30:50], src[8:32, 30:50])
    # clipping still available, and it is what makes Y == source outside the mask
    yc = compose_y(src, content, mask, rect=rect, clip_to_mask=True)
    assert np.array_equal(yc[mask < 0.5], src[mask < 0.5])
    # a rect partly off-frame is clipped, not wrapped
    yoff = compose_y(src, content, mask, rect=(-10, -10, 20, 20),
                     clip_to_mask=False)
    assert yoff.shape == src.shape and np.isfinite(yoff).all()

    # the overlap warning
    assert patch_outside_mask(mask, (24, 16, 16, 32)) == 0.0, "fully inside"
    assert patch_outside_mask(mask, (0, 0, 32, 32)) > 0.5, "mostly outside"
    assert patch_outside_mask(mask, (0, 0, 0, 0)) == 1.0, "degenerate"

    # THE UNIFICATION: one placement drives BOTH the content and the mask, so a
    # patch with its own background cannot paste that background in.
    H = W = 64
    src2 = np.full((H, W, 3), 0.4, np.float32)
    # a patch that is mostly BLACK with a bright object in one corner: exactly
    # the failure the user photographed (a black square pasted over a diagram)
    pat2 = np.zeros((40, 40, 3), np.float32)
    pat2[10:30, 10:30] = 1.0
    pmask2 = np.zeros((40, 40), np.float32)
    pmask2[10:30, 10:30] = 1.0          # the object, not the whole crop

    y2, m2 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30))
    assert y2.shape == src2.shape and m2.shape == (H, W)
    # the mask is strictly inside the patch footprint, because the corner of the
    # patch was NOT marked
    foot = np.zeros((H, W), bool)
    fx, fy, fw, fh = place_rect((H, W), pat2.shape, (10, 10, 30, 30))
    foot[fy:fy + fh, fx:fx + fw] = True
    assert m2[~foot].max() == 0.0, "the mask escaped the patch footprint"
    assert 0.0 < float(m2.mean()) < float(foot.mean()), \
        "the mask must cover less than the footprint (the crop has background)"
    # and inside the patch footprint, the BLACK part of the patch contributes
    # nothing, because the mask is 0 there
    assert np.allclose(y2[~foot], src2[~foot]), "outside the patch must be source"
    # a pixel that is inside the footprint but outside the marked object is
    # still source, not black -- this is the whole point
    inside_foot_outside_mask = foot & (m2 < 0.01)
    assert inside_foot_outside_mask.any(), "test is vacuous"
    assert np.allclose(y2[inside_foot_outside_mask],
                       src2[inside_foot_outside_mask]), \
        "the patch's unmarked (background) pixels must not be pasted"

    # the marked object IS pasted
    marked = m2 > 0.99
    assert marked.any() and not np.allclose(y2[marked], src2[marked])

    # patch_mask=None falls back to the whole footprint (tight-crop case)
    y3, m3 = compose_patch(src2, pat2, None, (10, 10, 30, 30))
    assert np.allclose(m3, foot.astype(np.float32))

    # THE DRIFT TEST: a NON-SQUARE patch, with the marking and the content
    # going through the same placement, must land on the SAME pixels.
    #
    # This is the invariant the browser broke. Its panel was a fixed 320x320
    # square with the patch letterboxed inside, while the mask was painted on
    # that square: the mask followed one coordinate system and the content
    # another, so a 4:3 patch drifted 30 px and a barely-non-square one (1.05)
    # drifted 5 px -- and a SQUARE patch drifted 0, which is why every test up
    # to here passed. The page now sizes the panel to the patch's aspect and
    # sends that panel, so panel and patch are the same shape and the server's
    # single place_rect() call covers both. The browser side of that contract
    # is asserted in page_harness.js; this is the server side.
    for (pw, ph) in ((400, 300), (400, 380), (512, 384), (500, 250), (300, 400)):
        # the page's panelSize(): the long edge is 320, aspect preserved
        k = min(320 / pw, 320 / ph)
        PW, PH = max(1, round(pw * k)), max(1, round(ph * k))
        assert abs(PW / PH - pw / ph) < 0.01, (pw, ph, PW, PH)
        # ...so the box computed from the PANEL equals the box from the PATCH
        assert place_rect((H, W), (PH, PW), (10, 10, 30, 30)) == \
            place_rect((H, W), (ph, pw), (10, 10, 30, 30)), (pw, ph)

        # a marker ON the object, in the panel's coordinates
        panel = np.zeros((PH, PW, 3), np.float32)
        panel[..., 2] = 0.1
        mx0, my0, mw, mh = PW // 4, PH // 5, PW // 5, PH // 4
        panel[my0:my0 + mh, mx0:mx0 + mw] = 1.0          # the marker (bright)
        mark = np.zeros((PH, PW), np.float32)
        mark[my0:my0 + mh, mx0:mx0 + mw] = 1.0           # and its marking
        y, m = compose_patch(src2, panel, mark, (10, 10, 30, 30), fit="contain")

        # where the MARKER landed in Y
        cb = (y > 0.9).all(axis=2)
        ys, xs = np.nonzero(cb)
        # where the MASK says it landed
        my, mx = np.nonzero(m > 0.5)
        assert cb.sum() > 0 and m.sum() > 0, (pw, ph, cb.sum(), m.sum())
        # the mask must cover the marker: same pixels, not merely similar area
        assert abs(int(xs.min()) - int(mx.min())) <= 1, (pw, ph, xs.min(), mx.min())
        assert abs(int(ys.min()) - int(my.min())) <= 1, (pw, ph, ys.min(), my.min())
        assert abs(int(xs.max()) - int(mx.max())) <= 1, (pw, ph, xs.max(), mx.max())
        assert abs(int(ys.max()) - int(my.max())) <= 1, (pw, ph, ys.max(), my.max())

    # contain, not stretch: a non-matching aspect must not distort
    wide = (0, 0, 60, 20)
    dx, dy, dw, dh = place_rect((H, W), pat2.shape, wide, fit="contain")
    assert abs(dw / dh - 40 / 40) < 0.08, (dw, dh)   # aspect kept
    assert dw <= 60 and dh <= 20 and dx >= 0 and dy >= 0
    sx, sy, sw, sh = place_rect((H, W), pat2.shape, wide, fit="stretch")
    assert (sw, sh) == (60, 20)
    try:
        place_rect((H, W), pat2.shape, wide, fit="bogus")
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown fit must be refused")

    # grow extends past the footprint on purpose; feather softens the edge
    _, m4 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30), grow=3)
    assert float(m4.mean()) > float(m2.mean())
    _, m5 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30), feather=4)
    edge = (m5 > 0.02) & (m5 < 0.98)
    assert edge.any(), "feathering must produce intermediate values"
    assert grow_mask(pmask2, 0).shape == pmask2.shape
    assert np.allclose(grow_mask(pmask2, 0), pmask2)
    assert np.allclose(feather_mask(pmask2, 0), pmask2)

    # source-side touch-up: 1 forces ON, 0 forces OFF, 0.5 abstains
    ref = np.full((H, W), 0.5, np.float32)
    y6, m6 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30),
                           refine_mask=ref)
    assert np.allclose(m6, m2), "an all-0.5 refine layer must change nothing"
    on = np.full((H, W), 0.5, np.float32)
    on[0:6, 0:6] = 1.0                      # add, in a corner the patch misses
    _, m7 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30), refine_mask=on)
    assert float(m7[0:6, 0:6].min()) == 1.0, "forced-ON must be exactly 1"
    assert float(m7.mean()) > float(m6.mean())
    off = np.full((H, W), 0.5, np.float32)
    off[12:28, 12:28] = 0.0                 # cut, inside the object
    _, m8 = compose_patch(src2, pat2, pmask2, (10, 10, 30, 30), refine_mask=off)
    assert float(m8[12:28, 12:28].max()) == 0.0, "forced-OFF must be exactly 0"
    assert float(m8.mean()) < float(m6.mean())
    # the refine layer is in SOURCE coordinates and must be the right size
    try:
        compose_patch(src2, pat2, pmask2, (10, 10, 30, 30),
                      refine_mask=np.zeros((4, 4), np.float32))
    except SystemExit as exc:
        assert "refine_mask" in str(exc)
    else:
        raise AssertionError("a wrong-sized refine mask must be refused")

    # degradation directions, each on a case where the sign is unambiguous
    flat = np.zeros((32, 32, 3), np.float32)
    flat[:, 16:] = 1.0                                    # left dark, right bright
    assert degrade(flat, "none").shape == flat.shape
    assert np.allclose(degrade(flat, "solid"), 0.5)
    assert np.allclose(degrade(flat, "blur:4"), degrade(flat, "blur:4"))
    n1 = degrade(flat, "noise:0.1", seed=1)
    n2 = degrade(flat, "noise:0.1", seed=1)
    assert np.array_equal(n1, n2), "the noise variant must be reproducible"
    assert not np.array_equal(n1, degrade(flat, "noise:0.1", seed=2))
    sh = degrade(flat, "shift:8")
    assert np.allclose(sh, np.roll(flat, 8, axis=1))
    # scale:2 = zoom in then centre-crop back, so a marker right of centre must
    # move FURTHER right. (In a 40x40 frame, a marker at column 30 becomes
    # column 40 after a 2x zoom-and-crop: 2*30 - 20 = 40.)
    marker = np.zeros((40, 40, 3), np.float32)
    marker[20, 30] = 1.0                                  # right of centre
    zoomed = degrade(marker, "scale:2")
    col = int(np.argmax(zoomed[20, :, 0]))
    assert col > 30, f"zoom must push the marker outward; got column {col}"
    for bad in ("bogus", "blur:x"):
        try:
            degrade(flat, bad)
        except (SystemExit, ValueError):
            pass
        else:
            raise AssertionError(f"--variant {bad!r} must be refused")

    # an unknown strategy is refused, not silently treated as the default
    try:
        compose_y(src, content, mask, strategy="nope")
    except SystemExit as exc:
        assert "strategy" in str(exc)
    else:
        raise AssertionError("an unknown --strategy must be refused")

    print("propose_y self-test OK")
    print("  bbox: exact / padded / edge-clipped, empty mask refused")
    print("  Y == source outside the mask EXACTLY, for all 4 strategies")
    print("  Y != source inside (it proposes something), soft and hard masks")
    print("  variants: solid, blur, reproducible noise, shift, zoom-in")
    print("  unknown strategy and unknown variant are refused")
    print("  compose_patch: ONE placement drives content AND mask, so a patch's")
    print("  own background is never pasted; contain does not distort; grow and")
    print("  feather behave")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--instruction", default=None,
                    help="the edit instruction; used as the T2I prompt only if "
                         "--y-prompt is not given")
    ap.add_argument("--y-prompt", default=None,
                    help="what to GENERATE. An edit instruction ('A hot air "
                         "balloon should also be floating.') is not a good T2I "
                         "prompt; give a describing one ('a colourful hot air "
                         "balloon floating in a blue sky').")
    ap.add_argument("--from-image", default=None,
                    help="skip generation and place this image instead. Use "
                         "with --variant to measure how good Y has to be.")
    ap.add_argument("--strategy", choices=STRATEGIES, default="bbox")
    ap.add_argument("--variant", default="none",
                    help="degrade the content: none|solid|blur:S|noise:S|"
                         "shift:PX|scale:F")
    ap.add_argument("--bbox-pad", type=int, default=0)
    ap.add_argument("--hard-mask", action="store_true",
                    help="blend with a binarised mask instead of the soft one")
    ap.add_argument("--out", default="runs/st/y.png")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers",
                    help="the T2I model; stock SD3, NOT the editing checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negative", default="")
    args = ap.parse_args(argv)

    source = load_image(args.image, args.size)
    mask_px = load_mask(args.mask, args.size)
    if mask_px.shape != source.shape[:2]:
        raise SystemExit(f"mask {mask_px.shape} vs image {source.shape[:2]}: "
                         "disagreeing grids make the composite meaningless")

    if args.y_prompt is None and args.instruction is None:
        raise SystemExit("give --y-prompt (what to generate) or --instruction")

    if args.from_image:
        content = load_image(args.from_image, args.size)
        prompt = None
    else:
        prompt = args.y_prompt or args.instruction
        if args.y_prompt is None:
            print("[propose_y] NOTE: no --y-prompt, so the EDIT INSTRUCTION is "
                  "being used as the text-to-image prompt. That is usually a "
                  "bad prompt ('A hot air balloon should also be floating.' "
                  "describes a change, not a picture). Give --y-prompt.")
        print(f"[propose_y] generating: {prompt!r}  ({args.model})")
        content = np.asarray(generate_t2i(
            prompt, args.model, args.device, args.size, args.seed,
            args.steps, args.guidance, args.negative), np.float32) / 255.0

    if args.variant and args.variant != "none":
        content = degrade(content, args.variant, seed=args.seed)

    y = compose_y(source, content, mask_px, strategy=args.strategy,
                  pad=args.bbox_pad, soft=not args.hard_mask)

    out = Path(args.out)
    save_image(y, out)
    bbox = mask_bbox(mask_px, args.bbox_pad)
    inside = mask_px > 0.5
    rec = {
        "image": args.image, "mask": args.mask, "out": str(out),
        "prompt": prompt, "from_image": args.from_image,
        "strategy": args.strategy, "variant": args.variant,
        "bbox_y0x0y1x1": list(bbox), "bbox_pad": args.bbox_pad,
        "soft_mask": not args.hard_mask, "model": args.model,
        "seed": args.seed, "size": args.size,
        "mask_frac": float(inside.mean()),
        "y_vs_source_inside": float(np.abs(y - source)[inside].mean()),
        "y_vs_source_outside": float(np.abs(y - source)[~inside].mean()),
    }
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    if rec["y_vs_source_outside"] != 0.0:
        print("  !! Y differs from the source OUTSIDE the mask. That should be "
              "impossible by construction; do not use this Y.")
    print(f"\nwrote {out}. NEXT, the same command line 1 uses:\n"
          f"  python short_term/edit.py --image {args.image} "
          f"--mask {args.mask} \\\n"
          f"      --guide-image {out} --instruction \"{args.instruction or prompt}\" \\\n"
          f"      --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat \\\n"
          f"      --out runs/st/y_edit.png\n"
          "and score it WITH --tgt so the proposed-Y run can be read against "
          "the dataset's own edit, not against a threshold.")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
