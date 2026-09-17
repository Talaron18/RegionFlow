"""
Gate A -- does a learned conditional transport decouple attribute change from
source destruction?

    # train (3090Ti, hours not days)
    python scripts/gate_a_transport.py train --cache /data/cache/<hash> \
        --out runs/gate_a --steps 8000

    # evaluate against the pre-registered Stage 0 threshold
    python scripts/gate_a_transport.py eval --ckpt runs/gate_a/ckpt_final.pt \
        --cases captions/captions.json --out runs/gate_a/eval

The eval sweeps displacement magnitude a in [0, 1.5] and computes
corr(dE, struct) with the Stage 0 metric code, on the Stage 0 cases. The
verdict is decided by bench/decoupling.py against a threshold fixed before the
run.

READ THE CONTACT SHEET BEFORE THE VERDICT. Stage 0's rule, earned three times:
automated verdicts were wrong every time they fired, and a refutation here
would be the most consequential number the project has produced. `eval` writes
a contact sheet per case and refuses to print a verdict unless
--i-looked-at-the-images is passed. That flag is not a joke; it is the cheapest
control the project has.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.decoupling import (                      # noqa: E402
    decoupling_report, evaluate_point, format_report, oracle_relative_report,
)
from transport.adapter import AdapterConfig, TransportAdapter, freeze_and_report  # noqa: E402
from transport.data import TripletCache, torch_dataset  # noqa: E402
from transport.pairflow import PairFlowConfig, apply_transport, pairflow_loss  # noqa: E402
from transport.train import TrainConfig, Trainer, enable_memory_savers  # noqa: E402

#: Displacement magnitudes. Extended past 1.0 because the 2000-step run
#: measured cos = 0.49 with move_ratio/cos = 0.95: under MSE the optimal
#: amplitude for a direction with alignment rho is exactly rho, so the model
#: is correctly hedged rather than timid, and target amplitude is reached
#: around a = 1/cos ~ 2.0. The sweep must cover that or it measures a
#: systematically undersized edit and calls it a weak attribute direction.
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)

#: Stage 0 case ids -> the names its correlations were reported under.
#: bench/decoupling.STAGE0_CORR is keyed by these names.
STAGE0_IDS = {"17320_t1": "ball", "158548_t1": "sheep",
              "429011_t1": "phone_booth"}


def build_adapter(model: str, device: str, rank: int, max_blocks=None):
    import torch
    from diffusers import SD3Transformer2DModel

    tr = SD3Transformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    adapter = TransportAdapter(
        tr, AdapterConfig(mode="pair", rank=rank, max_blocks=max_blocks)
    ).to(device)
    # Adapter parameters stay fp32 even though the base is bf16. See train.py.
    for p in adapter.trainable_parameters():
        p.data = p.data.float()
    print(freeze_and_report(adapter))
    return adapter


def cmd_train(args) -> int:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    cache = TripletCache(args.cache)
    ds = torch_dataset(cache)
    print(f"cache {args.cache}: {len(ds)} triplets")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)

    adapter = build_adapter(args.model, args.device, args.rank, args.max_blocks)
    enable_memory_savers(adapter.transformer)

    # Structural guarantee, checked before a single gradient step.
    b = next(iter(loader))
    chk = adapter.check_identity(
        b["z_src"][:1].to(args.device),
        torch.full((1,), 0.5, device=args.device),
        b["prompt_embeds"][:1].to(args.device, adapter.model_dtype),
        b["pooled"][:1].to(args.device, adapter.model_dtype),
        mask=b["mask"][:1].to(args.device),
    )
    print("identity check:", json.dumps(chk, indent=1))
    if not chk["adapter_is_live"]:
        raise RuntimeError("adapter is inert; training would be a no-op")

    pf = PairFlowConfig(mask_dropout=args.mask_dropout,
                        cond_dropout=args.cond_dropout,
                        sigma0=args.sigma0)

    def loss_fn(model, batch, step, cfg):
        dev, mdt = args.device, model.model_dtype
        return pairflow_loss(
            model,
            batch["z_src"].to(dev).float(),
            batch["z_tgt"].to(dev).float(),
            batch["mask"].to(dev).float(),
            batch["prompt_embeds"].to(dev, mdt),
            batch["pooled"].to(dev, mdt),
            pf,
        )

    tcfg = TrainConfig(gate="a", out_dir=args.out, steps=args.steps,
                       batch_size=args.batch, grad_accum=args.accum,
                       lr=args.lr, max_hours=args.max_hours, log_every=args.log_every,
                       ckpt_every=args.ckpt_every, seed=args.seed)
    trainer = Trainer(adapter, loader, tcfg, loss_fn)
    resume = Path(args.out) / "ckpt_latest.pt"
    if resume.exists():
        trainer.load(resume)
    path = trainer.run()
    print(f"done: {path}")
    return 0


def load_eval_cases(captions_paths, data_root, split, size):
    """Stage 0's caption files select the cases; the dataset supplies the rest.

    captions.json is {case_id: [source_caption, target_caption_with_brackets]}
    and Stage 0 looked the images and masks up in MagicBrush by that id. This
    eval does the same, so the Gate A numbers are computed on exactly the
    cases Stage 0 reported -0.958 / -0.726 / -0.955 for.

    The CONDITIONING is the dataset's own instruction string, not the caption.
    Training used `instruction` from the parquet; evaluating on captions would
    change the conditioning distribution and confound the comparison with a
    prompt shift. Stage 0 measured a wrong caption at 3.65 dB -- larger than
    the solver effect it was trying to isolate -- so this is not a small
    detail.
    """
    import json as _json

    wanted = {}
    for cp in captions_paths:
        p = Path(cp)
        if not p.exists():
            print(f"[eval] skipping missing {p}")
            continue
        wanted.update(_json.loads(p.read_text()))
    if not wanted:
        raise FileNotFoundError(f"no case ids loaded from {captions_paths}")

    from precompute_cache import iter_magicbrush

    found = {}
    for trip in iter_magicbrush(data_root, split, size, limit=None):
        if trip.case_id in wanted:
            found[trip.case_id] = trip
        if len(found) == len(wanted):
            break

    missing = set(wanted) - set(found)
    if missing:
        print(f"[eval] {len(missing)} case id(s) not in the '{split}' split: "
              f"{sorted(missing)}")
        print("       Stage 0 used the DEV split "
              "(bench/data.py::load_magicbrush defaults to split='dev'). "
              "Pass --split dev.")
    if not found:
        raise RuntimeError(
            "none of the caption case ids were found in the dataset. Nothing "
            "to evaluate."
        )
    return found


def cmd_eval(args) -> int:
    import torch
    from diffusers import StableDiffusion3Pipeline

    from bench.inversion import SD3EditPipeline
    from transport.conditioning import mask_to_latent_grid

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = load_eval_cases(args.captions, args.data_root, args.split, args.size)
    print(f"[eval] {len(cases)} case(s): {sorted(cases)}")

    # Leakage check. Stage 0's cases are in dev and training ran on train, so
    # they should be disjoint -- but "should be" is how the last three days
    # went. A decoupling verdict measured on training data would be worthless
    # and would look identical to a real one.
    if args.cache:
        from transport.data import TripletCache
        trained_ids = {rec["id"] for rec in TripletCache(args.cache).items}
        overlap = set(cases) & trained_ids
        if overlap:
            raise RuntimeError(
                f"LEAKAGE: {len(overlap)} eval case(s) are in the training "
                f"cache: {sorted(overlap)}. The verdict would be measured on "
                "training data. Rebuild the cache from a split that excludes "
                "them, or evaluate different cases."
            )
        print(f"[eval] leakage check clean against {len(trained_ids)} "
              "trained ids")
    else:
        print("[eval] no --cache given; leakage NOT checked")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)

    adapter = TransportAdapter(
        pipe.transformer, AdapterConfig(mode="pair", rank=args.rank)
    ).to(args.device)
    for p_ in adapter.trainable_parameters():
        p_.data = p_.data.float()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has unexpected keys: {unexpected[:5]}")
    trained = [k for k in sd["adapter"] if "lora_b" in k]
    live = sum(float(sd["adapter"][k].abs().sum()) > 0 for k in trained)
    print(f"[eval] checkpoint step {sd['step']}, {live}/{len(trained)} "
          f"lora_b blocks are non-zero")
    if live == 0:
        raise RuntimeError(
            "every lora_b is still zero -- this checkpoint is the pretrained "
            "model. Evaluating it would produce a confident null.")
    adapter.eval()

    points_by_case, oracle_by_case = {}, {}
    for cid, trip in sorted(cases.items()):
        name = STAGE0_IDS.get(cid, cid)
        img, mask = trip.image, trip.mask

        z = helper.encode_image(img)
        h, w = z.shape[-2:]
        m = mask_to_latent_grid(mask, h, w).to(args.device)
        pe, _, pooled, _ = helper.encode_prompt(trip.instruction)

        # THE REFERENCE IS THE VAE ROUND TRIP, NOT THE SOURCE.
        # decode(encode(x)) already costs dE 1.9-4.4 and struct 0.73-0.87 on
        # these three images, before any transport is applied. Measuring
        # against the source folds that into every point: on the ball, struct
        # starts at 0.73 while Stage 0's weakest strength reported 0.645, so
        # most of that gap is the autoencoder rather than the edit. Comparing
        # edited against the a=0 reconstruction isolates what the transport
        # actually did.
        # ORACLE PASS. The dataset's target latent gives the displacement that
        # is known to work for THIS edit; scoring the model against it removes
        # any argument about which structure metric is right.
        z_tgt = helper.encode_image(trip.target_image).float()
        v_star = (z_tgt - z.float())

        recon = helper.decode_latent(z.float())
        vae_pt = evaluate_point(name, -1.0, img, recon, mask)
        print(f"  {name} ({cid}) instr={trip.instruction!r}")
        print(f"    VAE ceiling (a=0 vs source): dE {vae_pt.dE:.1f}  "
              f"struct {vae_pt.struct:.2f}  -- metrics below are vs this")

        panels = [img, recon]
        labels = ["source", f"VAE recon  dE={vae_pt.dE:.1f} st={vae_pt.struct:.2f}"]
        pts = []
        for a in STRENGTHS:
            # Re-seeded FRESH for every `a`, not one generator advancing
            # across the sweep -- the sweep is only meaningful if every
            # strength queries the SAME sampled mode (same eps) and only
            # `a` itself varies. An advancing generator would silently
            # compare different modes at different strengths.
            gen = (torch.Generator(device=args.device).manual_seed(args.eval_seed)
                  if args.sigma0 > 0 else None)
            z_e = apply_transport(adapter, z.float(), m, pe, pooled,
                                  strength=a, steps=args.steps_readout,
                                  pin_background=not args.no_pin,
                                  sigma0=args.sigma0, generator=gen)
            edited = helper.decode_latent(z_e)
            p = evaluate_point(name, a, recon, edited, mask)
            p.extra["vs_source_dE"] = evaluate_point(name, a, img, edited, mask).dE
            pts.append(p)
            panels.append(edited)
            labels.append(f"a={a}  dE={p.dE:.1f}  st={p.struct:.2f}")

        oracle_pts = []
        for a in STRENGTHS:
            z_o = z.float() + a * v_star * m
            p_o = evaluate_point(name, a, recon, helper.decode_latent(z_o), mask)
            oracle_pts.append(p_o)
        oracle_by_case[name] = oracle_pts

        points_by_case[name] = pts
        _sheet(panels, labels, out / f"sheet_{name}.png")
        print(f"    dE {pts[0].dE:.1f} -> {pts[-1].dE:.1f}   "
              f"struct {pts[0].struct:.2f} -> {pts[-1].struct:.2f}   "
              f"bg change {pts[-1].background_change:.4f}")

    rep = decoupling_report(points_by_case)
    rel = oracle_relative_report(points_by_case, oracle_by_case)
    # Parity at ONE dE is not parity everywhere. Sweep the probe so the
    # low-dE/high-dE picture is visible: the model may match the real edit at
    # moderate colour change and diverge as strength grows.
    rel_curve = [oracle_relative_report(points_by_case, oracle_by_case, probe=pr)
                 for pr in (10.0, 15.0, 20.0, 25.0, 30.0, 40.0)]
    (out / "decoupling.json").write_text(json.dumps(
        {"report": rep, "oracle_relative": rel, "ckpt": str(args.ckpt), "strengths": list(STRENGTHS),
         "reference": "vae_reconstruction" if True else "source",
         "background_pinned": not args.no_pin,
         "oracle_relative_curve": rel_curve,
         "points": {k: [vars(p) for p in v] for k, v in points_by_case.items()},
         "oracle_points": {k: [vars(p) for p in v]
                           for k, v in oracle_by_case.items()}},
        indent=1, default=float))

    if not args.i_looked_at_the_images:
        print(f"\nContact sheets written to {out}. Look at them, then rerun "
              "with --i-looked-at-the-images to print the verdict.\n"
              "Stage 0's automated verdicts were wrong every time they fired; "
              "this one decides whether the project continues.")
        return 0

    print("\n===== PRIMARY: oracle-relative structure =====")
    print(f"VERDICT {rel['verdict']}   mean oracle_rel = "
          f"{rel['mean_oracle_rel']}   ({rel['metric']} at dE="
          f"{rel['probe_dE']:.0f})")
    print(f"  rule: refute >= {rel['rule']['refute_above']}, "
          f"confirm <= {rel['rule']['confirm_below']}")
    print(f"\n{'case':<13}{'model':>9}{'oracle':>9}{'rel':>8}"
          f"{'struct0 m':>11}{'struct0 o':>11}")
    for c, v in rel["per_case"].items():
        f = lambda x: "--" if x is None else f"{x:.3f}"
        print(f"{c:<13}{f(v['model']):>9}{f(v['oracle']):>9}"
              f"{f(v['oracle_rel']):>8}{f(v['model_struct0']):>11}"
              f"{f(v['oracle_struct0']):>11}")
    print(f"\n  including excluded cases: mean = "
          f"{rel['mean_oracle_rel_all_cases']}  over {rel['n_cases_all']} "
          f"case(s), {rel['n_excluded']} excluded")
    print(f"\n{'probe dE':>9}{'mean rel':>10}{'n':>4}  verdict")
    for rc in rel_curve:
        m = "--" if rc["mean_oracle_rel"] is None else f"{rc['mean_oracle_rel']:.3f}"
        print(f"{rc['probe_dE']:>9.0f}{m:>10}{rc['n_cases']:>4}  {rc['verdict']}")
    print("\n===== SECONDARY: Stage 0 metric, kept for comparability =====")
    print(format_report(rep))
    return 0


def _sheet(panels, labels, path, size=320):
    from PIL import Image, ImageDraw

    ims = [Image.fromarray(p).resize((size, size), Image.LANCZOS) for p in panels]
    sheet = Image.new("RGB", (size * len(ims), size + 24), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        sheet.paste(im, (i * size, 24))
        d.text((i * size + 4, 6), lab, fill=(0, 0, 0))
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(model="stabilityai/stable-diffusion-3-medium-diffusers")

    t = sub.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", default="runs/gate_a")
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--max-blocks", type=int, default=None)
    t.add_argument("--mask-dropout", type=float, default=0.1)
    t.add_argument("--cond-dropout", type=float, default=0.1)
    t.add_argument("--sigma0", type=float, default=0.0,
                   help="Direction A (GATE_A_REOPENING_PLAN.md SS8): "
                        "mask-gated noise at the s=0 query point, "
                        "sigma(s)=sigma0*(1-s). 0.0 (default) is the "
                        "original deterministic pairflow, byte-identical. "
                        "Start small (e.g. 0.1-0.3 relative to latent "
                        "scale) and sweep -- not validated at any value yet.")
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--ckpt-every", type=int, default=500,
                   help="numbered snapshots land every 5x this")
    t.add_argument("--max-hours", type=float, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--model", default=common["model"])
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--captions", nargs="+",
                   default=["captions/captions.json",
                            "captions/captions_booth.json"],
                   help="Stage 0 caption files; their keys select the cases")
    e.add_argument("--data-root", default="/data/datasets/MagicBrush",
                   help="MagicBrush root -- images, masks and instructions "
                        "are read from here, by case id")
    e.add_argument("--split", default="dev",
                   help="Stage 0's cases come from the DEV split "
                        "(bench/data.py::load_magicbrush defaults to it). "
                        "Training runs on train, so the eval cases are held "
                        "out -- which is the point.")
    e.add_argument("--cache", default=None,
                   help="training cache; if given, cross-checks that no eval "
                        "case was trained on")
    e.add_argument("--size", type=int, default=512)
    e.add_argument("--out", default="runs/gate_a/eval")
    e.add_argument("--rank", type=int, default=16)
    e.add_argument("--steps-readout", type=int, default=1,
                   help=">1 integrates the path; if the result differs from "
                        "1 step, the path is curved -- report that")
    e.add_argument("--no-pin", action="store_true",
                   help="do NOT restrict the displacement to the mask. "
                        "Measures whether the model localises on its own, "
                        "which is a separate result from edit quality and is "
                        "NOT comparable with Stage 0's pinned numbers.")
    e.add_argument("--sigma0", type=float, default=0.0,
                   help="sample WITH Direction A's mode-selecting noise "
                        "(GATE_A_REOPENING_PLAN.md SS8) instead of querying "
                        "at z_src exactly. 0.0 (default) keeps this sweep "
                        "continuous with Stage 0's deterministic methodology "
                        "-- if the checkpoint was TRAINED with sigma0>0, "
                        "that means evaluating it slightly off its own "
                        "training distribution (see apply_transport's "
                        "docstring); use this flag to check that gap "
                        "directly, or to preview stochastic samples, but "
                        "the sweep's oracle-relative verdict is the "
                        "sigma0=0 run for comparability. Direction A's "
                        "PRIMARY evaluation is the Gate C scoreboard, "
                        "sampled with sigma0>0 -- not yet wired here, see "
                        "the reopening plan.")
    e.add_argument("--eval-seed", type=int, default=0,
                   help="generator seed for --sigma0 sampling; re-seeded "
                        "fresh for every strength in the sweep so all "
                        "points compare the same sampled mode")
    e.add_argument("--i-looked-at-the-images", action="store_true")
    e.add_argument("--device", default="cuda")
    e.add_argument("--model", default=common["model"])
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
