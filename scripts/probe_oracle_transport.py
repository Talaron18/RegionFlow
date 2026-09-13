"""
Is the coupling a property of SD3's latent space, or of APPROXIMATE transport?

Gate A established that a learned conditional transport reproduces Stage 0's
coupling: struct@dE30 ~ 0.36-0.39 across three checkpoints, flat as cos moved
0.506 -> 0.656. The tempting conclusion is that attribute change and structure
destruction are entangled in the SD3 latent.

That conclusion has a problem. MagicBrush's targets are REAL edits -- the booth
turns red with its panels intact. So the displacement

    v* = z_tgt - z_src

reaches large dE with structure preserved, by construction, and it lives in the
same latent space. If the latent coupled colour to structure, v* could not
exist. It does.

So the alternative is that the target direction is fine and APPROXIMATE
directions are not: at cos ~ 0.6 the model emits 0.6 * (right direction) plus
0.8 * (wrong direction), and a large step in a partly-wrong direction leaves
the manifold, which the decoder renders as smooth mush. Structure would then
collapse from off-manifold decoding, not from entanglement.

TWO EXPERIMENTS, NO TRAINING.

  1. ORACLE SWEEP. Apply a * v* for a in [0, 1.25], pinned to the mask, and
     measure the same dE / struct pair. At a=1 this IS the real edited image,
     so struct should be high at large dE.

       oracle struct@dE30 HIGH  -> a decoupling direction exists and the
                                   coupling is NOT a property of the latent
                                   space. The learner's problem is finding the
                                   direction, and Gate A's result is about
                                   approximation, not about SD3.
       oracle struct@dE30 LOW   -> even the ground-truth displacement destroys
                                   structure by this measure. Then either the
                                   latent really is entangled, or `struct` does
                                   not mean what we think on real edits -- and
                                   the metric must be fixed before any claim.

  2. MISALIGNMENT SWEEP. The decisive one. Build displacements with KNOWN
     alignment to v*:

         v(rho) = rho * v_hat* + sqrt(1 - rho^2) * n_hat,   n orthogonal to v*

     rescaled to the MSE-optimal magnitude for that alignment (rho * ||v*||,
     the amplitude Gate A showed the model adopts). Sweep rho from 1.0 down
     through the model's measured 0.585-0.656 to the constant-direction
     baseline 0.453, and measure struct at matched dE.

     This yields the tolerance curve: how much structure survives as a function
     of directional accuracy. It answers the question Gate A could not, because
     Gate A only ever sampled three nearby values of cos and confounded them
     with everything else that differs between checkpoints.

       struct falls off steeply between rho=1 and rho=0.65  -> the project
           needs a formulation that does not take a large step along an
           averaged direction. That is precisely the noise-conditioned,
           on-manifold generative path -- Gate B -- and Gate A becomes its
           motivation rather than a refutation.
       struct stays low even at rho=0.95  -> the coupling is real and
           intrinsic, and instance-level editing via latent displacement in
           SD3 is not viable. Report that; it is a strong negative result.

    python scripts/probe_oracle_transport.py --data-root /data/datasets/MagicBrush
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.decoupling import evaluate_point, struct_at_dE  # noqa: E402

STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
RHOS = (1.0, 0.95, 0.85, 0.75, 0.656, 0.585, 0.5, 0.453)
STAGE0_IDS = {"17320_t1": "ball", "158548_t1": "sheep",
              "429011_t1": "phone_booth"}


def orthogonal_noise(v, rng):
    """Unit vector orthogonal to v, same shape."""
    n = rng.standard_normal(v.shape).astype(np.float32)
    vf, nf = v.ravel(), n.ravel()
    nf -= (nf @ vf) / (vf @ vf) * vf
    n = nf.reshape(v.shape)
    return n / (np.linalg.norm(n) + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/datasets/MagicBrush")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--captions", nargs="+",
                    default=["captions/captions.json",
                             "captions/captions_booth.json"])
    ap.add_argument("--out", default="runs/oracle")
    ap.add_argument("--model",
                    default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusion3Pipeline

    from bench.inversion import SD3EditPipeline
    from precompute_cache import iter_magicbrush
    from transport.conditioning import mask_to_latent_grid

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    wanted = set()
    for cp in args.captions:
        p = Path(cp)
        if p.exists():
            wanted |= set(json.loads(p.read_text()))

    trips = {}
    for t in iter_magicbrush(args.data_root, args.split, args.size, limit=None):
        if t.case_id in wanted:
            trips[t.case_id] = t
        if len(trips) == len(wanted):
            break
    if not trips:
        raise RuntimeError(f"no case ids found in the '{args.split}' split")
    print(f"[oracle] {len(trips)} case(s): {sorted(trips)}")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    results = {"oracle": {}, "misalignment": {}}

    for cid, t in sorted(trips.items()):
        name = STAGE0_IDS.get(cid, cid)
        z_src = helper.encode_image(t.image).float()
        z_tgt = helper.encode_image(t.target_image).float()
        h, w = z_src.shape[-2:]
        m = mask_to_latent_grid(t.mask, h, w).to(z_src.device)
        recon = helper.decode_latent(z_src)

        v = (z_tgt - z_src).cpu().numpy()[0]          # (16, h, w)
        mnp = m.cpu().numpy()[0]                      # (1, h, w)

        # ---- 1. oracle sweep --------------------------------------------
        pts, panels, labels = [], [t.image, recon], ["source", "VAE recon"]
        for a in STRENGTHS:
            z_e = z_src + a * torch.from_numpy(v).to(z_src)[None] * m
            img = helper.decode_latent(z_e)
            p = evaluate_point(name, a, recon, img, t.mask)
            pts.append(p)
            panels.append(img)
            labels.append(f"oracle a={a} dE={p.dE:.1f} st={p.struct:.2f}")
        s30 = struct_at_dE(pts, 30.0)
        s25 = struct_at_dE(pts, 25.0)
        results["oracle"][name] = {
            "struct_at_dE25": s25, "struct_at_dE30": s30,
            "points": [{"a": p.strength, "dE": p.dE, "struct": p.struct}
                       for p in pts],
        }
        print(f"\n  {name}: ORACLE struct@dE25 {s25}  struct@dE30 {s30}")
        print(f"    dE 0 -> {pts[-1].dE:.1f}, struct 1.00 -> {pts[-1].struct:.2f}")
        _sheet(panels, labels, out / f"oracle_{name}.png")

        # ---- 2. misalignment sweep --------------------------------------
        # Fixed target magnitude ||v*||; each rho uses the MSE-optimal
        # amplitude rho*||v*||, which is what Gate A showed the model adopts.
        n_hat = orthogonal_noise(v, rng)
        v_norm = float(np.linalg.norm(v))
        v_hat = v / (v_norm + 1e-8)
        rows, mpanels, mlabels = [], [t.image, recon], ["source", "VAE recon"]
        for rho in RHOS:
            d = rho * v_hat + np.sqrt(max(1 - rho * rho, 0.0)) * n_hat
            d = d / (np.linalg.norm(d) + 1e-8) * (rho * v_norm)
            sub = []
            for a in (0.5, 1.0, 1.5, 2.0):
                z_e = z_src + a * torch.from_numpy(d.astype(np.float32)).to(
                    z_src)[None] * m
                img = helper.decode_latent(z_e)
                p = evaluate_point(name, a, recon, img, t.mask)
                sub.append(p)
                if a == 1.0:
                    mpanels.append(img)
                    mlabels.append(f"rho={rho} dE={p.dE:.1f} st={p.struct:.2f}")
            rows.append({
                "rho": rho,
                "struct_at_dE25": struct_at_dE(sub, 25.0),
                "struct_at_dE30": struct_at_dE(sub, 30.0),
                "points": [{"a": p.strength, "dE": p.dE, "struct": p.struct}
                           for p in sub],
            })
        results["misalignment"][name] = rows
        _sheet(mpanels, mlabels, out / f"misalign_{name}.png")

        print(f"    {'rho':>7}{'st@dE25':>10}{'st@dE30':>10}")
        for r in rows:
            f25 = "--" if r["struct_at_dE25"] is None else f"{r['struct_at_dE25']:.3f}"
            f30 = "--" if r["struct_at_dE30"] is None else f"{r['struct_at_dE30']:.3f}"
            print(f"    {r['rho']:>7.3f}{f25:>10}{f30:>10}")

    (out / "oracle.json").write_text(json.dumps(results, indent=1, default=float))
    _verdict(results)
    return 0


def _verdict(res):
    import numpy as np
    orc = [v["struct_at_dE25"] for v in res["oracle"].values()
           if v["struct_at_dE25"] is not None]
    print("\n" + "=" * 62)
    if orc:
        mo = float(np.mean(orc))
        print(f"ORACLE struct@dE25 = {mo:.3f}   (learned model: ~0.41)")
        if mo >= 0.70:
            print("  A structure-preserving displacement EXISTS in this latent\n"
                  "  space. The Gate A coupling is a property of APPROXIMATE\n"
                  "  transport, not of SD3. Do not write the latent-space claim.")
        elif mo <= 0.45:
            print("  Even the GROUND-TRUTH displacement destroys structure by\n"
                  "  this measure. Either the latent is genuinely entangled, or\n"
                  "  `struct` is wrong on real edits. Check the oracle contact\n"
                  "  sheet by eye before believing either -- at a=1.0 that\n"
                  "  panel IS the real edited image from the dataset, so if it\n"
                  "  looks correct and scores 0.3, the METRIC is the problem.")
        else:
            print("  Intermediate. Read the contact sheets; the a=1.0 oracle\n"
                  "  panel is the real target image and its score calibrates\n"
                  "  everything else.")
    for name, rows in res["misalignment"].items():
        good = [r for r in rows if r["struct_at_dE25"] is not None]
        if len(good) >= 2:
            hi, lo = good[0], good[-1]
            print(f"\n{name}: struct@dE25 {hi['struct_at_dE25']:.3f} at "
                  f"rho={hi['rho']:.2f}  ->  {lo['struct_at_dE25']:.3f} at "
                  f"rho={lo['rho']:.2f}")
    print("=" * 62)


def _sheet(panels, labels, path, size=300):
    from PIL import Image, ImageDraw

    ims = [Image.fromarray(p).resize((size, size), Image.LANCZOS) for p in panels]
    sheet = Image.new("RGB", (size * len(ims), size + 24), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        sheet.paste(im, (i * size, 24))
        d.text((i * size + 4, 6), lab, fill=(0, 0, 0))
    sheet.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
