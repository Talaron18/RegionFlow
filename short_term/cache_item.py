#!/usr/bin/env python3
"""
Export a real MagicBrush triplet from the precomputed cache as a local fixture.

WHY THIS REPLACES THE SYNTHETIC FIXTURE. The two-disc image is out of
distribution for a model trained on photographs, and -- worse for diagnosis --
it has no ground truth: "make the shirt red" on a disc that is already red
cannot distinguish a correct no-op from a failure, and it cost the edit-quality
half of the first experiment (short_term/README.md SS4.1). A cache item is a
real photograph with a real instruction, a real mask, AND the dataset's own
edited target. That last one is the whole point: it is the control
STAGE1_REPORT SS3 says would have prevented a wrong claim --

    "Score the ground truth on your metric before trusting it."

so `score.py --tgt` can put the dataset's real edit on the same axes as the
model's output.

WHAT IT WRITES

    src.png     the source, decoded from the cached latent
    tgt.png     the dataset's edited target, decoded the same way
    mask.png    the edit mask, white = editable (the project convention)
    meta.json   id, instruction, mask_frac, moved, cache path, latent shapes
    thumb_*.png only with --no-decode: the cache's 128px thumbnails

WHY DECODE THE LATENTS RATHER THAN USE THE THUMBS. `precompute_cache.py` writes
thumbs as `case.image[::4, ::4]`, i.e. 128x128 at a 512px cache -- too small to
judge a photo edit on. The cached latents are the 64x64 grid the model actually
consumes, and `precompute_cache.encode_image` stores them as
`(vae_latent - shift) * scaling`, which is exactly the convention
`SD3IP2PEditor.encode_image/decode_latent` uses. So decoding them reproduces
both the image the model sees and the target it was trained toward, at 512px,
with no resize mismatch.

This needs only `AutoencoderKL`, which is in STOCK diffusers -- unlike the
editor path, fixture export does not require the UltraEdit fork.

    # what is in the cache, best candidates first
    python short_term/cache_item.py --cache /data/cache/<hash> --list 20

    # export one, by position or by its dataset id
    python short_term/cache_item.py --cache /data/cache/<hash> --index 137 \\
        --out runs/st/real
    python short_term/cache_item.py --cache /data/cache/<hash> --id 523394_t1 \\
        --out runs/st/real

    # no GPU: 128px thumbs plus the mask, for inspection only
    python short_term/cache_item.py --cache /data/cache/<hash> --index 137 \\
        --out runs/st/real --no-decode

    python short_term/cache_item.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from short_term.backends import save_image  # noqa: E402

#: A usable local case: the mask has to be big enough for a visible edit and
#: small enough that "did the rest survive" is a real question. Stage 0 swept
#: 0.93%-15.1% and the coupling held across all of it, so this is a display
#: filter, not a claim about difficulty.
MASK_MIN, MASK_MAX = 0.02, 0.30


def load_index(cache: str | Path) -> tuple[dict, list[dict]]:
    """-> (meta, items). Reads meta.json only; no npz, no torch."""
    root = Path(cache)
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(
            f"{meta_path} not found. --cache takes the HASH directory itself "
            "(e.g. /data/cache/13061d51aee4), not its parent. If the hash is "
            "unknown, look under the parent for a directory with meta.json.")
    meta = json.loads(meta_path.read_text())
    return meta, meta["items"]


def rank(items: list[dict], mask_min: float = MASK_MIN,
         mask_max: float = MASK_MAX) -> list[dict]:
    """Candidates a local fixture should be drawn from, most-changed first.

    `moved` is the cache's own mean |z_tgt - z_src|; a case near 0 has almost
    no edit to show, and the cache writer already refuses the exactly-zero ones.
    """
    keep = [r for r in items
            if mask_min <= float(r.get("mask_frac", 0.0)) <= mask_max]
    return sorted(keep, key=lambda r: -float(r.get("moved", 0.0)))


def _decode(z: np.ndarray, vae, torch_mod, device: str) -> np.ndarray:
    """(C,h,w) cached latent -> (H,W,3) float [0,1]."""
    dtype = next(vae.parameters()).dtype
    scale = float(vae.config.scaling_factor)
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    t = torch_mod.from_numpy(np.asarray(z, np.float32))[None].to(device, dtype)
    with torch_mod.no_grad():
        img = vae.decode((t / scale + shift).to(dtype)).sample
    img = img.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img + 1.0) / 2.0


def export(args) -> int:
    meta, items = load_index(args.cache)
    if args.index is not None:
        if not 0 <= args.index < len(items):
            raise SystemExit(f"--index {args.index} out of range (cache has "
                             f"{len(items)} items)")
        rec = items[args.index]
    elif args.id is not None:
        hit = [r for r in items if r["id"] == args.id]
        if not hit:
            raise SystemExit(f"--id {args.id!r} not in this cache")
        rec = hit[0]
    else:
        raise SystemExit("give --index or --id (see --list)")

    root = Path(args.cache)
    d = np.load(root / f"{rec['stem']}.npz")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = d["mask"].shape[-1] if d["mask"].ndim >= 2 else 0
    factor = int(round(meta["config"]["size"] / max(n, 1))) if n else 8

    # mask: latent grid -> pixel grid. NEAREST, because it is a binary grid and
    # any interpolating resize would invent a soft ring that the pipeline never
    # saw. (The guidance weight is deliberately soft elsewhere; this is the
    # CONDITIONING mask, which is thresholded by construction.)
    m_lat = (np.asarray(d["mask"])[0] > 0.5).astype(np.float32)
    m_px = np.repeat(np.repeat(m_lat, factor, axis=0), factor, axis=1)
    save_image(m_px, out / "mask.png")

    info = {
        "cache": str(root), "stem": rec["stem"], "id": rec["id"],
        "instruction": rec["instruction"], "mask_frac": rec.get("mask_frac"),
        "moved": rec.get("moved"), "latent_grid": list(d["z_src"].shape[-2:]),
        "pixel_size": int(m_px.shape[0]), "upsample_factor": factor,
        "emb": rec.get("emb"),
    }

    if args.no_decode:
        for key, name in (("thumb_src", "thumb_src.png"),
                          ("thumb_tgt", "thumb_tgt.png")):
            if key in d.files:
                save_image(np.asarray(d[key], np.float32) / 255.0, out / name)
                info[name] = list(np.asarray(d[key]).shape)
            else:
                info[name] = None
        info["decoded"] = False
        (out / "meta.json").write_text(json.dumps(info, indent=1))
        if not getattr(args, "quiet", False):
            print(json.dumps(info, indent=1))
            print("\n[--no-decode] wrote 128px thumbs (or nothing, if the "
                  "cache has none) plus the mask. These are for INSPECTION "
                  "ONLY -- do not run the editor on a 128px upsample. Re-run "
                  "without --no-decode.")
        return 0

    import torch
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "diffusers is required to decode the cached latents. It is stock "
            "diffusers (AutoencoderKL), not the UltraEdit fork. Use "
            "--no-decode for the 128px thumbs without it.") from exc

    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae",
        torch_dtype=torch.float16 if args.device.startswith("cuda")
        else torch.float32).to(args.device)
    for key, name in (("z_src", "src.png"), ("z_tgt", "tgt.png")):
        save_image(_decode(d[key], vae, torch, args.device), out / name)
    info["decoded"] = True
    info["model"] = args.model
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    if getattr(args, "quiet", False):
        return 0
    print(json.dumps(info, indent=1))
    print(f"\nwrote {out}/src.png, {out}/tgt.png, {out}/mask.png, "
          f"{out}/meta.json")
    print("\nNEXT: run the editor on this fixture, then score the model against "
          "the dataset's own target --\n"
          f"  python short_term/edit.py --image {out}/src.png "
          f"--mask {out}/mask.png \\\n"
          f"      --instruction \"{rec['instruction']}\" --out "
          f"runs/st/real_a.png\n"
          f"  python short_term/score.py --src {out}/src.png --mask "
          f"{out}/mask.png \\\n"
          f"      --out runs/st/real_*.png --tgt {out}/tgt.png "
          "--expect <the colour in the instruction> --rank")
    return 0


def _self_test() -> int:
    """CPU, no model, no cache. Checks the fixture plumbing against a cache
    this test builds, so the reader is exercised on the real format."""
    import tempfile
    import torch  # noqa: F401  (only to prove the import path is clean)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "deadbeefcafe"
        root.mkdir(parents=True)
        lat = 8
        z_src = np.random.default_rng(0).random((16, lat, lat), np.float32)
        z_tgt = z_src + 0.3
        mask = np.zeros((1, lat, lat), np.float32)
        mask[0, 2:6, 2:6] = 1.0
        np.savez_compressed(root / "00000.npz", z_src=z_src, z_tgt=z_tgt,
                            mask=mask,
                            thumb_src=np.zeros((32, 32, 3), np.uint8),
                            thumb_tgt=np.zeros((32, 32, 3), np.uint8))
        (root / "meta.json").write_text(json.dumps({
            "config": {"size": 64, "latent_size": lat, "dtype": "float32",
                       "max_items": None, "context_length": 333},
            "n": 1,
            "items": [{"id": "case_1", "stem": "00000",
                       "instruction": "change the red circle to blue",
                       "emb": None, "mask_frac": 0.25, "moved": 0.3}],
        }))

        meta, items = load_index(root)
        assert meta["n"] == 1 and len(items) == 1
        assert rank(items)[0]["id"] == "case_1"
        assert rank(items, mask_min=0.5, mask_max=0.9) == [], \
            "the mask filter must reject a case outside its band"

        out = Path(tmp) / "fixture"
        rc = export(argparse.Namespace(
            cache=str(root), index=0, id=None, out=str(out), no_decode=True,
            model="unused", device="cpu", quiet=True))
        assert rc == 0
        info = json.loads((out / "meta.json").read_text())
        assert info["decoded"] is False and info["id"] == "case_1"
        assert info["upsample_factor"] == 8, info["upsample_factor"]
        from PIL import Image
        m = np.asarray(Image.open(out / "mask.png"))
        assert m.shape == (64, 64, 3), m.shape
        inside = (m[..., 0] > 127)
        assert inside.sum() == 16 * 64, int(inside.sum())   # 4x4 latent -> 32px
        # the instruction survives the round trip verbatim -- it is the field
        # the whole experiment is keyed on
        assert (out / "meta.json").read_text().count(
            "change the red circle to blue") == 1

        # and the failure modes are loud, not silent
        try:
            load_index(Path(tmp) / "nope")
        except SystemExit as exc:
            assert "meta.json" in str(exc)

    print("cache_item self-test OK")
    print("  reads the real meta.json/.npz layout, filters by mask_frac")
    print("  latent mask -> pixel mask with NEAREST x8, area preserved")
    print("  --no-decode writes thumbs and the mask with no model at all")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True,
                    help="the cache HASH directory (contains meta.json)")
    ap.add_argument("--list", type=int, default=0, metavar="N",
                    help="print the N best candidates and exit")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--out", default="runs/st/real")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-decode", action="store_true",
                    help="write the cache's 128px thumbs instead of decoding "
                         "the latents; needs no model, for inspection only")
    args = ap.parse_args(argv)

    if args.list:
        _meta, items = load_index(args.cache)
        cand = rank(items)
        print(f"{len(items)} items in the cache, {len(cand)} inside the "
              f"mask_frac band [{MASK_MIN}, {MASK_MAX}], most-changed first")
        print(f"\n{'index':>6}  {'id':<16}{'mask_frac':>10}{'moved':>8}  "
              "instruction")
        for r in cand[:args.list]:
            i = items.index(r)
            print(f"{i:>6}  {str(r['id'])[:16]:<16}"
                  f"{float(r.get('mask_frac', 0)):>10.4f}"
                  f"{float(r.get('moved', 0)):>8.3f}  "
                  f"{str(r['instruction'])[:60]}")
        print("\npick one whose instruction names something you can see, and "
              "whose mask covers it. Then --index I --out DIR.")
        return 0

    return export(args)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
