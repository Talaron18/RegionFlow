"""FLUX.2 [klein] as a Gate C model: instruction + source image -> edited image.

    python bench/run_gate_c.py --model flux2-klein-4b --mask-mode none --limit 4 \
        --out runs/gate_c/klein4b_smoke.jsonl \
        --save-images runs/gate_c/img_klein4b_smoke

WHY A SEPARATE FILE. It needs stock diffusers recent enough to ship
`Flux2KleinPipeline` (the repo's configs say 0.37.0.dev0; the model card says
install from git). The UltraEdit arms need UltraEdit's diffusers FORK at
0.30.0.dev0. The two cannot share an environment, so run this arm from its
own venv and leave the project image alone.

NO MASK. FLUX.2 klein takes a prompt and reference image(s); no mask or
region input is documented. Only --mask-mode none is meaningful, and the
wrapper refuses anything else rather than silently dropping the mask --
a dropped mask would be scored as an "oracle" arm that never saw one.

SETTINGS are the model card's for the distilled 4B checkpoint: 4 steps,
guidance_scale 1.0. run_gate_c.py's own --steps / --guidance defaults
(50 / 7.5) are UltraEdit's and are deliberately NOT forwarded.

COMPOSITE BASELINE (--klein-composite). The model still never sees the mask.
It edits the whole image exactly as in the no-mask run (same seed, same
call), and the result is then pasted into the SOURCE through the
conditioning mask run_gate_c builds (white = editable, already dilated by
--mask-dilate), with a Gaussian feather. Everything outside the mask is the
source pixel for pixel, so sibling leakage is removed by construction. This
is the trivial thing any mask-based method on this base has to beat: it
shows what a mask alone buys, and what it costs at the boundary.

    python bench/run_gate_c.py --model flux2-klein-4b --klein-composite \
        --mask-mode oracle --out runs/gate_c/klein4b_comp_oracle.jsonl \
        --save-images runs/gate_c/img_klein4b_comp_oracle

CROP BASELINE (--klein-crop). Tells the model WHERE without changing the
model: crop the source to the mask's bounding box plus a margin, so the
siblings are (mostly) out of view; edit that crop; paste it back. Run it
with --ablation so the benchmark's single-instance instruction
("Replace the cow with a cat") is scored too -- the full referring
expression ("the cow in back") describes context the crop no longer shows.

    python bench/run_gate_c.py --model flux2-klein-4b --klein-crop --ablation \
        --mask-mode oracle --out runs/gate_c/klein4b_crop_oracle.jsonl \
        --save-images runs/gate_c/img_klein4b_crop_oracle

--klein-crop-paste mask (default) pastes back through the mask, as the
composite arm does: nothing outside the mask changes. `box` pastes the
whole crop, which keeps additions placed beside the object but lets the
margin change too.

NOT RUN ON A GPU when written. `self_test` checks the call the wrapper
makes against a stub pipeline; whether the real pipeline accepts it is
what the 4-case smoke run above is for. Look at those four images first.
"""

from __future__ import annotations

REPOS = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
}

#: FLUX.2's VAE is 1/8 with 2x2 patches, so sides must be multiples of 16.
SIZE_MULTIPLE = 16


def composite(src, edited, mask, feather: float = 4.0):
    """Paste `edited` into `src` where `mask` is white. PIL images in, PIL out.

    mask is the runner's conditioning image (RGB or L, white = editable).
    feather is a Gaussian radius in pixels on the alpha; 0 gives a hard
    edge and leaves every pixel outside the mask exactly equal to `src`.
    With feather > 0 the blend reaches a few pixels past the mask edge.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415

    src = src.convert("RGB")
    if edited.size != src.size:
        edited = edited.resize(src.size, Image.BILINEAR)
    alpha = mask.convert("L")
    if alpha.size != src.size:
        alpha = alpha.resize(src.size, Image.NEAREST)
    alpha = alpha.point(lambda v: 255 if v >= 128 else 0)
    if feather and feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return Image.composite(edited.convert("RGB"), src, alpha)


def _alpha(mask, size):
    """The runner's conditioning image as a binary L image at `size`."""
    from PIL import Image  # noqa: PLC0415
    a = mask.convert("L")
    if a.size != size:
        a = a.resize(size, Image.NEAREST)
    return a.point(lambda v: 255 if v >= 128 else 0)


def crop_box(mask, size, margin: float = 0.25, min_margin: int = 16):
    """Bounding box of the white mask region, grown by `margin` of its own
    width/height on each side (at least `min_margin` px), clamped to `size`."""
    box = _alpha(mask, size).getbbox()
    if box is None:
        raise SystemExit("--klein-crop: the mask is empty, nothing to crop to")
    x0, y0, x1, y1 = box
    mx = max(min_margin, int(round(margin * (x1 - x0))))
    my = max(min_margin, int(round(margin * (y1 - y0))))
    W, H = size
    return (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))


def model_size(w: int, h: int, area: int = 512 * 512, max_side: int = 1024,
               multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """Size to run the model at for a w x h crop: same aspect, about `area`
    pixels, sides multiples of `multiple` and at most `max_side`. A small
    crop is upsampled so the object gets the model's full resolution."""
    s = (area / float(w * h)) ** 0.5
    s = min(s, max_side / float(w), max_side / float(h))
    return snap_size(w * s, h * s, multiple)


def snap_size(w: int, h: int, multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """Nearest multiple-of-`multiple` size, never below one multiple."""
    def _snap(x):
        return max(multiple, int(round(x / multiple)) * multiple)
    return _snap(w), _snap(h)


class Flux2KleinModel:
    """Same call contract as run_gate_c.SD3InstructPix2Pix."""

    def __init__(self, name: str = "flux2-klein-4b", steps: int = 4,
                 guidance: float = 1.0, device: str = "cuda",
                 offload: bool = False, pipe=None,
                 composite_mask: bool = False, feather: float = 4.0,
                 crop: bool = False, crop_margin: float = 0.25,
                 crop_paste: str = "mask"):
        if name not in REPOS:
            raise SystemExit(f"unknown klein model {name!r}; "
                             f"known: {sorted(REPOS)}")
        self.repo = REPOS[name]
        self.composite_mask, self.feather = bool(composite_mask), float(feather)
        # distinct name so composite results never pool with the plain arm
        self.crop, self.crop_margin = bool(crop), float(crop_margin)
        if crop_paste not in ("mask", "box"):
            raise SystemExit(f"crop_paste must be 'mask' or 'box', got {crop_paste!r}")
        self.crop_paste = crop_paste
        if self.crop and self.composite_mask:
            raise SystemExit("--klein-crop already pastes back; do not combine "
                             "it with --klein-composite")
        self.name = name + ("+composite" if self.composite_mask else
                            f"+crop-{crop_paste}" if self.crop else "")
        self.steps, self.g = int(steps), float(guidance)
        self.device, self.offload = device, bool(offload)
        if pipe is not None:            # injected (self_test)
            self.pipe = pipe
            return
        import torch  # noqa: PLC0415
        try:
            from diffusers import Flux2KleinPipeline  # noqa: PLC0415
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit(
                "Flux2KleinPipeline not found in this diffusers install. It "
                "needs a recent stock diffusers (`pip install "
                "git+https://github.com/huggingface/diffusers.git`), in a "
                "SEPARATE environment from the UltraEdit fork the other Gate "
                "C arms use.") from exc
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.repo, torch_dtype=torch.bfloat16)
        if self.offload:
            # the model card's own low-VRAM path; slower per image
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    def _generator(self, seed: int):
        import torch  # noqa: PLC0415
        # offload moves modules between devices per call; a CPU generator is
        # valid in both modes there, a CUDA one is not.
        dev = "cpu" if self.offload else self.device
        return torch.Generator(device=dev).manual_seed(int(seed))

    def __call__(self, img, prompt, seed, mask=None):
        if self.crop:
            if mask is None:
                raise SystemExit("--klein-crop needs a mask to crop to; use "
                                 "--mask-mode oracle (or sibling as a control).")
            return self._crop_edit(img, prompt, seed, mask)
        if mask is not None and not self.composite_mask:
            raise SystemExit(
                f"{self.name} has no mask input; run it with --mask-mode none, "
                "or add --klein-composite to paste the edit back through the "
                "mask. Refusing to drop the mask silently.")
        if mask is None and self.composite_mask:
            raise SystemExit(
                "--klein-composite needs a mask to paste through; use "
                "--mask-mode oracle (or white / sibling as controls).")
        src = img
        w, h = snap_size(*img.size)
        if (w, h) != img.size:
            img = img.resize((w, h))
        edited = self.pipe(prompt=prompt, image=img, height=h, width=w,
                           num_inference_steps=self.steps,
                           guidance_scale=self.g,
                           generator=self._generator(seed)).images[0]
        if self.composite_mask:
            return composite(src, edited, mask, self.feather)
        return edited


def _crop_edit_impl(self, img, prompt, seed, mask):
    from PIL import Image, ImageFilter  # noqa: PLC0415

    src = img.convert("RGB")
    box = crop_box(mask, src.size, self.crop_margin)
    src_crop = src.crop(box)
    mw, mh = model_size(*src_crop.size)
    edited = self.pipe(prompt=prompt,
                       image=src_crop.resize((mw, mh), Image.LANCZOS),
                       height=mh, width=mw,
                       num_inference_steps=self.steps, guidance_scale=self.g,
                       generator=self._generator(seed)).images[0]
    edited = edited.convert("RGB").resize(src_crop.size, Image.LANCZOS)
    if self.crop_paste == "mask":
        region = composite(src_crop, edited, _alpha(mask, src.size).crop(box),
                           self.feather)
    else:
        # whole crop, faded out toward the box edge so no hard rectangle shows
        a = Image.new("L", src_crop.size, 0)
        inset = int(round(2 * self.feather))
        if src_crop.size[0] > 2 * inset and src_crop.size[1] > 2 * inset:
            a.paste(255, (inset, inset, src_crop.size[0] - inset,
                          src_crop.size[1] - inset))
        else:
            a.paste(255, (0, 0) + src_crop.size)
        if self.feather > 0:
            a = a.filter(ImageFilter.GaussianBlur(radius=self.feather))
        region = Image.composite(edited, src_crop, a)
    out = src.copy()
    out.paste(region, box[:2])
    return out


Flux2KleinModel._crop_edit = _crop_edit_impl


def self_test() -> None:
    from PIL import Image  # noqa: PLC0415

    # torch is only needed for the real generator. Without it, stand in a
    # token so the rest of the wrapper is still exercised (the generator
    # stream check below is then skipped, and says so).
    try:
        import torch  # noqa: PLC0415,F401
        have_torch = True
    except ImportError:
        have_torch = False
        Flux2KleinModel._generator = lambda self, seed: ("seed", int(seed))

    assert snap_size(512, 512) == (512, 512)
    assert snap_size(500, 333) == (496, 336)
    assert snap_size(3, 3) == (16, 16)

    calls = []

    class _Out:
        def __init__(self, im):
            self.images = [im]

    class _Pipe:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out(Image.new("RGB", (kw["width"], kw["height"])))

    m = Flux2KleinModel(pipe=_Pipe(), device="cpu")
    out = m(Image.new("RGB", (500, 333)), "make the left cup blue", seed=7)
    kw = calls[-1]
    assert set(kw) == {"prompt", "image", "height", "width",
                       "num_inference_steps", "guidance_scale", "generator"}, kw
    assert kw["prompt"] == "make the left cup blue"
    assert (kw["width"], kw["height"]) == (496, 336) == kw["image"].size
    assert out.size == (496, 336)
    # the card's distilled settings, not run_gate_c's UltraEdit defaults
    assert kw["num_inference_steps"] == 4 and kw["guidance_scale"] == 1.0

    # same seed -> same generator stream; different seed -> different
    if have_torch:
        import torch  # noqa: PLC0415
        a = torch.rand(3, generator=m._generator(7))
        b = torch.rand(3, generator=m._generator(7))
        c = torch.rand(3, generator=m._generator(8))
        assert torch.equal(a, b) and not torch.equal(a, c)
    else:
        print("  (torch not installed: generator stream check skipped)")

    # a mask must be refused, never dropped
    try:
        m(Image.new("RGB", (512, 512)), "x", seed=0,
          mask=Image.new("RGB", (512, 512)))
    except SystemExit:
        pass
    else:
        raise AssertionError("mask was accepted")
    # composite: the pipeline call is IDENTICAL to the no-mask arm (same
    # kwargs, no mask anywhere), and only the paste differs.
    import numpy as np  # noqa: PLC0415

    class _Red:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out(Image.new("RGB", (kw["width"], kw["height"]), (255, 0, 0)))

    src = Image.new("RGB", (512, 512), (0, 0, 255))
    msk = Image.new("RGB", (512, 512), (0, 0, 0))
    msk.paste((255, 255, 255), (100, 100, 200, 200))
    hard = Flux2KleinModel(pipe=_Red(), device="cpu", composite_mask=True, feather=0)
    assert hard.name == "flux2-klein-4b+composite"
    o = np.asarray(hard(src, "x", seed=7, mask=msk))
    assert set(calls[-1]) == set(kw), "composite changed the pipeline call"
    inside = np.zeros((512, 512), bool); inside[100:200, 100:200] = True
    assert (o[inside] == (255, 0, 0)).all()
    assert (o[~inside] == (0, 0, 255)).all(), "pixels outside the mask changed"
    soft = Flux2KleinModel(pipe=_Red(), device="cpu", composite_mask=True, feather=4)
    o = np.asarray(soft(src, "x", seed=7, mask=msk))
    assert tuple(o[150, 150]) == (255, 0, 0)            # centre: the edit
    far = np.ones((512, 512), bool); far[70:230, 70:230] = False
    assert (o[far] == (0, 0, 255)).all(), "feather reached far from the mask"
    assert 0 < o[100, 150, 0] < 255                     # edge: a blend
    # mask size differing from the source is resized, not an error
    o = np.asarray(hard(src, "x", seed=7, mask=msk.resize((256, 256))))
    assert tuple(o[150, 150]) == (255, 0, 0) and tuple(o[20, 20]) == (0, 0, 255)
    # white mask == the plain model output (the "edit anywhere" control)
    o = np.asarray(hard(src, "x", seed=7, mask=Image.new("RGB", (512, 512), "white")))
    assert (o == (255, 0, 0)).all()
    try:
        hard(src, "x", seed=0)
    except SystemExit:
        pass
    else:
        raise AssertionError("composite without a mask was accepted")

    # ---- crop baseline -------------------------------------------------
    assert crop_box(msk, (512, 512)) == (75, 75, 225, 225)       # 100px box, 25% margin
    assert crop_box(msk, (512, 512), margin=0.0) == (84, 84, 216, 216)  # min margin 16
    edge = Image.new("L", (512, 512), 0); edge.paste(255, (0, 0, 40, 40))
    assert crop_box(edge, (512, 512)) == (0, 0, 56, 56)          # clamped at the border
    assert model_size(150, 150) == (512, 512)
    mw_, mh_ = model_size(300, 100)
    assert mw_ % 16 == 0 and mh_ % 16 == 0 and abs(mw_ * mh_ - 512 * 512) < 0.1 * 512 * 512
    assert max(model_size(4000, 40)) <= 1024
    try:
        crop_box(Image.new("L", (512, 512), 0), (512, 512))
    except SystemExit:
        pass
    else:
        raise AssertionError("empty mask was accepted")

    cm = Flux2KleinModel(pipe=_Red(), device="cpu", crop=True, feather=0)
    assert cm.name == "flux2-klein-4b+crop-mask"
    o = np.asarray(cm(src, "x", seed=7, mask=msk))
    # the model saw the CROP (upsampled to its working size), not the frame
    assert calls[-1]["image"].size == (512, 512) == (calls[-1]["width"], calls[-1]["height"])
    assert (o[inside] == (255, 0, 0)).all()
    assert (o[~inside] == (0, 0, 255)).all(), "crop+mask paste changed pixels outside the mask"
    cb = Flux2KleinModel(pipe=_Red(), device="cpu", crop=True, crop_paste="box", feather=0)
    assert cb.name == "flux2-klein-4b+crop-box"
    o = np.asarray(cb(src, "x", seed=7, mask=msk))
    inbox = np.zeros((512, 512), bool); inbox[75:225, 75:225] = True
    assert (o[inbox] == (255, 0, 0)).all()
    assert (o[~inbox] == (0, 0, 255)).all(), "box paste changed pixels outside the crop"
    cbf = Flux2KleinModel(pipe=_Red(), device="cpu", crop=True, crop_paste="box", feather=4)
    o = np.asarray(cbf(src, "x", seed=7, mask=msk))
    assert (o[~inbox] == (0, 0, 255)).all(), "feathered box paste leaked outside the crop"
    assert tuple(o[150, 150]) == (255, 0, 0)
    for bad in (lambda: cm(src, "x", seed=0),
                lambda: Flux2KleinModel(pipe=_Red(), crop=True, composite_mask=True),
                lambda: Flux2KleinModel(pipe=_Red(), crop=True, crop_paste="oval")):
        try:
            bad()
        except SystemExit:
            pass
        else:
            raise AssertionError("bad crop configuration was accepted")

    try:
        Flux2KleinModel(name="flux2-klein-99b", pipe=_Pipe())
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown name was accepted")
    print("flux2_klein_model self-test OK")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(__doc__)
