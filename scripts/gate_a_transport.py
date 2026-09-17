"""
Gate A -- does a learned conditional transport decouple attribute change from
source destruction?

    # train (3090Ti, hours not days)
    python scripts/gate_a_transport.py train --cache /data/cache/<hash> \
        --out runs/gate_a --steps 8000

    # evaluate against the pre-registered Stage 0 threshold
    python scripts/gate_a_transport.py eval --ckpt runs/gate_a/ckpt_final.pt \
        --cases captions/captions.json --out runs/gate_a/eval

    # watch a training run's log.jsonl and redraw a PNG every 100 steps --
    # run this in a second terminal alongside `train`, touches nothing in
    # the training loop itself
    python scripts/gate_a_transport.py watch-log --out runs/gate_a --every 100

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

    # persistent_workers=True: without it, PyTorch tears down and respawns
    # every worker process at EVERY epoch boundary (Trainer.run's own
    # `except StopIteration: it = iter(self.loader)` creates a fresh
    # DataLoader iterator there, and persistent_workers=False -- the
    # default -- means a fresh iterator means fresh worker processes).
    # Harmless for a single run; expensive when running several
    # concurrent sweep processes (sweep_sigma0.sh) that share roughly the
    # same cache size and so tend to hit their own epoch boundaries around
    # the same wall-clock time -- observed as a periodic multi-run stall
    # (GPU utilization sawtooth) rather than smooth throughput. No
    # downside here: same dataset for the whole run, nothing to reload.
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=True)

    adapter = build_adapter(args.model, args.device, args.rank, args.max_blocks)
    # Gradient checkpointing trades ~30-40% step time for activation memory,
    # same as gate_b_meanflow.py's own dial -- but Gate A never needs
    # forward-mode AD (pairflow_loss has no JVP estimator at all), so unlike
    # Gate B this is a pure memory/speed dial with no --estimator jvp
    # constraint to respect.
    #
    # It is also where this diffusers install breaks: diffusers==0.30.0.dev0
    # (vendored for StableDiffusion3InstructPix2PixPipeline) has a
    # checkpointed branch in transformer_sd3.py that assigns a
    # JointTransformerBlock's tuple return to hidden_states without
    # unpacking, so the NEXT block's LayerNorm gets a tuple and raises
    # "layer_norm(): argument 'input' ... must be Tensor, not tuple" -- on
    # the grad-ENABLED trainable forward itself, not just a no-grad probe
    # (see transport/adapter.py::no_checkpoint_without_grad's docstring,
    # which documents this exact error). Not caused by sigma0 or anything
    # else in Direction A -- this fires on ANY gradient-checkpointed
    # training step against this install. --no-grad-checkpointing is the
    # workaround; patching the vendored diffusers source to unpack the
    # tuple is the permanent fix.
    if args.grad_checkpointing:
        enable_memory_savers(adapter.transformer)
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[train] gradient checkpointing OFF -- if this OOMs, lower "
              "--batch and raise --accum by the same factor")

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


def summarize_diagnostics(raw: np.ndarray, pooled: np.ndarray,
                          centered: np.ndarray | None = None) -> dict:
    """Aggregate transport.pairflow.direction_position_diagnostics' per-
    example arrays (may contain NaN for degenerate examples, e.g. an
    empty mask) into mean/median/p10/p90 per treatment, NaNs dropped.
    Pulled out of cmd_diagnose so the aggregation/NaN-handling is
    self-testable without a GPU or a checkpoint.
    """
    def _stats(arr: np.ndarray) -> dict:
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"n": 0}
        return {"n": int(arr.size), "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90))}
    out = {"raw": _stats(raw), "pooled": _stats(pooled)}
    if centered is not None:
        out["centered"] = _stats(centered)
    return out


def print_diagnostics_summary(summary: dict) -> None:
    labels = {"raw": "raw (full frame)",
             "pooled": "pooled (masked, per-channel)",
             "centered": "centered (bias removed)"}
    for key, label in labels.items():
        if key not in summary:
            continue
        s = summary[key]
        if s["n"] == 0:
            print(f"  {label}: no valid examples")
            continue
        print(f"  {label}: n={s['n']}  mean {s['mean']:+.3f}  "
              f"median {s['median']:+.3f}  p10 {s['p10']:+.3f}  "
              f"p90 {s['p90']:+.3f}")


def _load_diagnose_checkpoint(adapter, ckpt_path: str) -> dict:
    """Load one checkpoint's adapter weights IN PLACE (reused across
    multiple --ckpt values so the expensive part -- loading the base
    transformer -- happens once). Returns the raw checkpoint dict (for its
    "step") after the same live-weight guard cmd_train/cmd_eval already
    use -- a checkpoint whose lora_b blocks are all still zero is the
    pretrained model, and diagnosing it would produce a confident null."""
    import torch

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _missing, unexpected = adapter.load_state_dict(sd["adapter"], strict=False)
    if unexpected:
        raise RuntimeError(f"{ckpt_path}: checkpoint has unexpected keys: "
                          f"{unexpected[:5]}")
    trained = [k for k in sd["adapter"] if "lora_b" in k]
    live = sum(float(sd["adapter"][k].abs().sum()) > 0 for k in trained)
    print(f"[diagnose] {ckpt_path}: step {sd.get('step', '?')}, "
          f"{live}/{len(trained)} lora_b blocks non-zero")
    if live == 0:
        raise RuntimeError(
            f"{ckpt_path}: every lora_b is still zero -- this checkpoint is "
            "the pretrained model, diagnosing it would produce a confident null")
    return sd


def _diagnose_checkpoint(adapter, dl, device, bias) -> dict:
    """Run one already-loaded checkpoint over the whole dataloader and
    return summarize_diagnostics()' summary dict. Pulled out of
    cmd_diagnose so multiple --ckpt values share one adapter/dataloader
    without repeating the accumulation loop."""
    import torch

    from transport.pairflow import direction_position_diagnostics

    raws, pooleds, centereds = [], [], []
    with torch.no_grad():
        for batch in dl:
            z_src = batch["z_src"].to(device).float()
            z_tgt = batch["z_tgt"].to(device).float()
            mask = batch["mask"].to(device).float()
            pe = batch["prompt_embeds"].to(device)
            pooled = batch["pooled"].to(device)
            b = z_src.shape[0]
            s0 = torch.zeros(b, device=device)
            # s=0, z=z_src exactly: the one-step readout's actual query
            # point (apply_transport's first grid point, at sigma0=0).
            u = adapter(z_src, s0, s0, pe, pooled, mask=mask)
            v_clean = z_tgt - z_src
            diag = direction_position_diagnostics(u.float(), v_clean, mask,
                                                  bias=bias)
            raws.append(diag["raw"].cpu().numpy())
            pooleds.append(diag["pooled"].cpu().numpy())
            if "centered" in diag:
                centereds.append(diag["centered"].cpu().numpy())

    raw_arr = np.concatenate(raws)
    pooled_arr = np.concatenate(pooleds)
    centered_arr = np.concatenate(centereds) if centereds else None
    return summarize_diagnostics(raw_arr, pooled_arr, centered_arr)


def cmd_diagnose(args) -> int:
    """WHERE/WHAT breakdown of one or more trained checkpoints' own
    predictions (GATE_A_REOPENING_PLAN.md SS2), following the ambiguity
    probe's mask-restrict work: the mask-pooled ceiling (0.621 on the real
    cache) sits well above the model's own full-field cos_clean
    (~0.50-0.60), which could mean the model is losing POSITION (getting
    the wrong pixels) or DIRECTION (getting the wrong per-channel shift)
    or both -- this tells you which. Runs entirely off the cached latents
    and prompt embeddings (transport/data.py's TripletCache); no VAE or
    text encoder needed, only the transformer + adapter, to compute
    u = model(z_src, 0, ...).

    --ckpt takes one or more paths (e.g. comparing sigma0=0.0/0.3/0.5
    checkpoints on the same cache) -- the base transformer is loaded ONCE
    and each checkpoint's adapter weights are swapped in, since loading
    the transformer is the expensive part.
    """
    from torch.utils.data import DataLoader

    import torch

    cache = TripletCache(args.cache)
    ds = torch_dataset(cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    print(f"[diagnose] {len(cache)} cached item(s) from {args.cache}")

    bias = None
    if args.bias_npy:
        bias = torch.from_numpy(np.load(args.bias_npy)).float()
        print(f"[diagnose] loaded bias vector shape {tuple(bias.shape)} "
              f"from {args.bias_npy}")
    else:
        print("[diagnose] no --bias-npy given -- skipping the centered "
              "treatment (generate one with: python scripts/"
              "ambiguity_probe.py --cache ... --save-bias PATH)")

    adapter = build_adapter(args.model, args.device, args.rank)
    adapter.eval()

    summaries = {}
    for ckpt_path in args.ckpt:
        sd = _load_diagnose_checkpoint(adapter, ckpt_path)
        summary = _diagnose_checkpoint(adapter, dl, args.device, bias)
        summaries[ckpt_path] = summary
        n = summary["raw"]["n"]
        print(f"\n[diagnose] {ckpt_path} (step {sd.get('step', '?')}) -- "
              f"model's own cos(u, v_clean) under three treatments "
              f"({n} examples):")
        print_diagnostics_summary(summary)

    if len(summaries) > 1:
        print("\n===== comparison (median cosine per treatment) =====")
        print(f"{'checkpoint':<45}{'raw':>8}{'pooled':>8}{'centered':>10}")
        for ckpt_path, summary in summaries.items():
            def _med(key):
                s = summary.get(key, {})
                return f"{s['median']:+.3f}" if s.get("n", 0) > 0 else "--"
            print(f"{ckpt_path:<45}{_med('raw'):>8}{_med('pooled'):>8}"
                  f"{_med('centered'):>10}")

    print("\ncompare against ambiguity_probe.py's cross-image ceilings on "
          "the same cache: raw 0.420, pooled 0.621, centered ~0.458 "
          "in-sample / see held-out distribution for the honest version. "
          "A model well below the pooled ceiling but close to the raw "
          "ceiling is losing DIRECTION more than position; the reverse "
          "means it already has the right direction but is imprecise "
          "about WHERE inside the mask to put it.")
    return 0


def find_dead_lora_b(adapter_sd: dict) -> list[str]:
    """lora_b keys whose tensor is exactly all-zero. The liveness count
    every train/eval/diagnose command already prints ("190/191 non-zero")
    collapses this to a number; this returns WHICH key(s), so a dead block
    that's the SAME one across multiple checkpoints (structural -- e.g. a
    block excluded by max_blocks, or one whose gradient path is genuinely
    never exercised) can be told apart from one that moves around
    (a training accident, e.g. an unlucky dead-ReLU-style zero-gradient
    step early in one specific run).
    """
    return sorted(k for k in adapter_sd
                 if "lora_b" in k and float(adapter_sd[k].abs().sum()) == 0.0)


def mask_enc_status(adapter_sd: dict) -> dict:
    """Per-key stats for every mask_enc.* parameter in a checkpoint's
    adapter state dict, flagging transport/conditioning.py::MaskEncoder's
    zero-init final projection (`out.weight`/`out.bias`) specifically.

    The lora_b liveness check doesn't cover mask_enc at all (it isn't a
    lora_b key) -- but delta_z = mask_enc.out(mask_enc.body(planes)) is
    IDENTICALLY zero for every input as long as out.weight/out.bias stay
    at their zero init, regardless of what mask_enc.body computed (a zero
    final linear map kills both the forward output AND the gradient
    flowing back to body, so body would never move either). In "pair"
    mode the gate multiplying this contribution is architecturally always
    1 (TransportAdapter._gate), never 0, so there's no structural reason
    for out to still be at zero after real training -- if it still is,
    that's a genuine bug in the conditioning path, not an expected
    artifact of the gating.
    """
    out = {}
    for k, t in adapter_sd.items():
        if "mask_enc" not in k:
            continue
        is_zero_init = k.endswith("out.weight") or k.endswith("out.bias")
        out[k] = {
            "shape": tuple(t.shape),
            "abs_sum": float(t.abs().sum()),
            "abs_max": float(t.abs().max()),
            "zero_init_layer": is_zero_init,
            "moved": (float(t.abs().sum()) > 0.0) if is_zero_init else None,
        }
    return out


def cmd_inspect(args) -> int:
    """CPU-only checkpoint inspection: which lora_b block(s) are still at
    zero-init (if any), and did mask_enc's zero-init output layer ever
    move? No GPU, no base transformer, no forward pass -- torch.load plus
    tensor stats on the adapter state dict only. Takes multiple --ckpt to
    check whether a dead block is the same one across runs (structural)
    or not (a training accident).
    """
    import torch

    for ckpt_path in args.ckpt:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        adapter_sd = sd["adapter"]
        print(f"\n{ckpt_path} (step {sd.get('step', '?')}):")

        lora_b_keys = [k for k in adapter_sd if "lora_b" in k]
        dead = find_dead_lora_b(adapter_sd)
        print(f"  lora_b: {len(lora_b_keys) - len(dead)}/{len(lora_b_keys)} "
              f"non-zero")
        for k in dead:
            print(f"    DEAD: {k}  shape={tuple(adapter_sd[k].shape)}")

        status = mask_enc_status(adapter_sd)
        if not status:
            print("  mask_enc: no keys in this checkpoint (use_mask=False?)")
            continue
        print("  mask_enc:")
        for k in sorted(status):
            s = status[k]
            tag = ""
            if s["zero_init_layer"]:
                tag = ("  <-- zero-init layer; "
                      + ("moved from zero-init" if s["moved"]
                         else "STILL AT ZERO -- mask contributes NOTHING "
                              "to z_in for every example"))
            print(f"    {k}  shape={s['shape']}  "
                  f"abs_sum={s['abs_sum']:.6g}  abs_max={s['abs_max']:.6g}{tag}")
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


# ---------------------------------------------------------- log watching --

def read_training_log(log_path: Path) -> list[dict]:
    """Parse log.jsonl into the step-indexed training records Trainer.run
    writes every cfg.log_every steps (transport/train.py). Drops val-only
    records (they carry "val"/"best_val" but no "loss" -- a separate,
    rarer event, not part of the per-step curve this plots).

    TOLERANT OF A PARTIAL LAST LINE ON PURPOSE: this is read while a
    training process may be mid-write to the same file (open("a"), one
    json.dumps + "\\n" per record -- appends are usually atomic at this
    size, but "usually" is not "always", and a watcher racing the
    trainer's own writes is exactly the situation where "usually" bites).
    A line that fails to parse is skipped, not raised -- it will parse
    cleanly next read once the write completes.
    """
    if not log_path.exists():
        return []
    recs = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in r and "loss" in r:
            recs.append(r)
    return recs


def plot_training_log(recs: list[dict], out_png: Path) -> None:
    """Render step-indexed training records (read_training_log's output)
    as a PNG. Four panels, not one combined axis -- loss (~1e-2), cos
    (in [-1,1]) and grad_norm (~1e1) live on scales that would flatten
    each other on a shared y-axis.

    cos / rel_in_floor are the two curves Direction A specifically needs:
    pairflow_loss's own stats dict already computes rel_in_floor = 1 -
    cos^2, the MSE-optimal floor for the CURRENT alignment (GATE_A_
    REOPENING_PLAN.md SS7 -- "1 - cos^2 is Gate B's own binding
    constraint" is the same quantity, same reason it binds). If rel_in
    is already tracking its own floor rather than falling further, more
    steps will not close the gap; that is the sigma0 sweep's question,
    not this run's, and the plot is what makes it visible without
    combing through log.jsonl by hand.

    Uses the Agg backend explicitly (headless -- this runs on a training
    box with no display) and imports matplotlib lazily, matching this
    project's convention for heavy/optional dependencies elsewhere
    (torch imported inside functions throughout transport/ and scripts/).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not recs:
        return
    steps = [r["step"] for r in recs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(steps, [r["loss"] for r in recs], label="loss")
    if "loss_in" in recs[0]:
        ax.plot(steps, [r["loss_in"] for r in recs], label="loss_in", alpha=0.7)
        ax.plot(steps, [r["loss_out"] for r in recs], label="loss_out", alpha=0.7)
    ax.set_title("loss")
    ax.set_xlabel("step")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if "cos" in recs[0]:
        ax.plot(steps, [r["cos"] for r in recs], label="cos (vs noised target)")
        if "cos_clean" in recs[0]:
            # cos vs cos_clean is itself the diagnostic for whether sigma0
            # is learning something real: cos alone can rise just because
            # the model partially echoes back sigma0*eps from its own
            # input (transport/pairflow.py's own note on this). cos_clean
            # can't be gamed that way -- if the two track together, the
            # improvement is real; if cos climbs while cos_clean doesn't,
            # that's the echo, not genuine mode-selection.
            ax.plot(steps, [r["cos_clean"] for r in recs],
                   label="cos_clean (vs true v)", linestyle="--")
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.set_ylim(-1.05, 1.05)
    ax.set_title("cos(u, v) -- directional alignment")
    ax.set_xlabel("step")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "rel_in" in recs[0]:
        ax.plot(steps, [r["rel_in"] for r in recs], label="rel_in")
        ax.plot(steps, [r["rel_in_floor"] for r in recs],
               label="rel_in_floor (1-cos²)", linestyle="--")
        ax.axhline(1.0, color="red", lw=0.5, alpha=0.5, label="predicting u=0")
    ax.set_title("rel_in vs its own MSE floor")
    ax.set_xlabel("step")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "grad_norm" in recs[0]:
        ax.plot(steps, [r["grad_norm"] for r in recs], label="grad_norm", alpha=0.6)
    if "move_ratio" in recs[0]:
        ax.plot(steps, [r["move_ratio"] for r in recs], label="move_ratio")
    ax.set_title("grad_norm / move_ratio")
    ax.set_xlabel("step")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{out_png.parent.name}  "
                f"({len(recs)} log points, last step={steps[-1]})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def cmd_watch_log(args) -> int:
    """Poll <out>/log.jsonl and redraw <out>/<png> once at least --every
    new steps have been logged since the last draw. Runs in a second
    terminal alongside `train` -- reads the log file Trainer.run already
    writes, never touches the training process or transport/train.py.
    """
    import time

    log_path = Path(args.out) / "log.jsonl"
    out_png = Path(args.out) / args.png
    last_drawn = -1
    print(f"[watch-log] watching {log_path} -> {out_png}, redraw every "
          f"{args.every} steps (polling every {args.poll_seconds:.0f}s"
          f"{', once' if args.once else ', Ctrl-C to stop'})")
    while True:
        recs = read_training_log(log_path)
        if recs and recs[-1]["step"] - last_drawn >= args.every:
            plot_training_log(recs, out_png)
            last_drawn = recs[-1]["step"]
            print(f"[watch-log] step {last_drawn} -> {out_png}  "
                  f"({len(recs)} points)")
        if args.once:
            break
        try:
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("\n[watch-log] stopped")
            break
    return 0


def plot_sweep_comparison(runs: dict[str, list[dict]], out_png: Path) -> None:
    """Overlay MULTIPLE training runs' logs (e.g. a sigma0 sweep, one run
    per value) on one set of axes -- one line per run per panel, so
    different sigma0 values compare directly instead of getting eyeballed
    across separate PNGs.

    Deliberately simpler than plot_training_log's own per-run view: just
    loss / cos_clean / rel_in / move_ratio, ONE line per run, not the
    in/out breakdown or the rel_in_floor overlay (which would need 2
    lines per run and stop being readable past 3-4 runs). Read a run's
    own training_curve.png (watch-log) for that detail -- this view
    answers the cross-run question, which sigma0 is winning, not the
    within-run one.

    cos_clean, NOT cos, is what a sigma0 sweep has to compare on. "cos"
    (pairflow_loss's stats dict) is cos(u, v) against sample_path's own
    NOISED target -- a DIFFERENT quantity at every sigma0, since that
    target is literally z_tgt - z_src - sigma0*eps. A model can partially
    learn to echo sigma0*eps straight back out of its own input (z_s
    contains it almost directly near s=0) rather than learning the true
    edit direction, and that echo gets EASIER to fit as sigma0 grows --
    so "cos" rising faster for larger sigma0 is exactly what you'd also
    see from pure metric-gaming, not only from real improvement.
    cos_clean = cos(u, z_tgt - z_src) can't be gamed that way: the true
    displacement has no eps in it regardless of which sigma0 or which eps
    produced z_s, so it's the one column comparable across the sweep.
    Falls back to "cos" for log files predating this field, with the
    panel title marking which one it actually drew.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {label: recs for label, recs in runs.items() if recs}
    if not runs:
        return

    cos_key = "cos_clean" if any("cos_clean" in r
                                 for recs in runs.values() for r in recs) \
             else "cos"
    cos_title = ("cos_clean(u, v) -- vs the TRUE displacement, comparable "
                "across sigma0" if cos_key == "cos_clean" else
                "cos(u, v) -- vs the noised target, NOT comparable across "
                "sigma0 (old log, predates cos_clean)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [(axes[0, 0], "loss", "loss"),
             (axes[0, 1], cos_key, cos_title),
             (axes[1, 0], "rel_in", "rel_in"),
             (axes[1, 1], "move_ratio", "move_ratio")]
    for ax, key, title in panels:
        for label, recs in runs.items():
            xs = [r["step"] for r in recs if key in r]
            ys = [r[key] for r in recs if key in r]
            if xs:
                ax.plot(xs, ys, label=label)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0, 1].axhline(0.0, color="grey", lw=0.5)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[1, 0].axhline(1.0, color="red", lw=0.5, alpha=0.5)

    last_steps = [recs[-1]["step"] for recs in runs.values()]
    fig.suptitle(f"sweep comparison -- {len(runs)} runs, "
                f"steps up to {max(last_steps)}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def cmd_watch_sweep(args) -> int:
    """Poll SEVERAL runs' log.jsonl (e.g. sweep_sigma0.sh's output
    directories) and redraw one combined comparison PNG once any run has
    advanced by --every steps since the last draw. Same non-interference
    principle as watch-log: reads log files from outside, touches none of
    the training processes.
    """
    import time

    outs = [Path(o) for o in args.outs]
    labels = args.labels if args.labels else [o.name for o in outs]
    if len(labels) != len(outs):
        raise SystemExit(f"--labels has {len(labels)} entries but --outs "
                         f"has {len(outs)}; must match 1:1")
    out_png = Path(args.png)
    last_drawn = -1
    print(f"[watch-sweep] watching {len(outs)} runs -> {out_png}, redraw "
          f"every {args.every} steps (polling every {args.poll_seconds:.0f}s"
          f"{', once' if args.once else ', Ctrl-C to stop'})")
    for label, o in zip(labels, outs):
        print(f"  {label}: {o}/log.jsonl")
    while True:
        runs = {label: read_training_log(o / "log.jsonl")
               for label, o in zip(labels, outs)}
        max_step = max((recs[-1]["step"] for recs in runs.values() if recs),
                       default=-1)
        if max_step >= 0 and max_step - last_drawn >= args.every:
            plot_sweep_comparison(runs, out_png)
            last_drawn = max_step
            counts = "  ".join(
                f"{l}={recs[-1]['step'] if recs else 'n/a'}"
                for l, recs in runs.items())
            print(f"[watch-sweep] -> {out_png}   {counts}")
        if args.once:
            break
        try:
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("\n[watch-sweep] stopped")
            break
    return 0


def self_test() -> None:
    """read_training_log / plot_training_log, CPU-only, no model weights,
    no GPU -- run before trusting `watch-log` on a live training run."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "log.jsonl"

        # Missing file -> empty, not an error (watch-log starts before
        # `train` has written its first record).
        assert read_training_log(log_path) == []

        recs_in = [
            {"step": 25, "lr": 1e-4, "grad_norm": 1.2, "hours": 0.01,
             "loss": 0.9, "loss_in": 0.95, "loss_out": 0.01,
             "rel_in": 0.98, "rel_in_floor": 0.9, "cos": 0.31,
             "move_ratio": 0.9, "amp_vs_optimal": float("nan")},
            {"step": 50, "lr": 1e-4, "grad_norm": 1.1, "hours": 0.02,
             "loss": 0.7, "loss_in": 0.75, "loss_out": 0.01,
             "rel_in": 0.85, "rel_in_floor": 0.8, "cos": 0.45,
             "move_ratio": 0.92, "amp_vs_optimal": 1.02},
        ]
        with log_path.open("w") as fh:
            for r in recs_in:
                fh.write(json.dumps(r) + "\n")
            # a val-only record (no "loss") must be read and skipped, not
            # raise -- Trainer.run() writes these interleaved with the
            # per-step records whenever --val-every fires.
            fh.write(json.dumps({"step": 50, "val": 0.5,
                                 "best_val": 0.5}) + "\n")
            # a truncated/corrupt line -- simulates reading mid-write,
            # racing the trainer's own append -- must be skipped too.
            fh.write('{"step": 75, "loss": 0.6, "in')

        recs = read_training_log(log_path)
        assert len(recs) == 2, recs
        assert [r["step"] for r in recs] == [25, 50]

        out_png = Path(td) / "curve.png"
        plot_training_log(recs, out_png)
        assert out_png.exists() and out_png.stat().st_size > 0

        # Missing optional keys (e.g. a future stats dict without "cos")
        # must not crash the render -- only the panels that have data.
        plot_training_log([{"step": 1, "loss": 0.5}], Path(td) / "bare.png")
        assert (Path(td) / "bare.png").exists()

        # plot_sweep_comparison: multiple runs overlaid. Different step
        # ranges on purpose (real concurrent runs don't stay in lockstep --
        # one process can be a few steps ahead of another) to check the
        # per-run x-axis isn't assumed shared.
        runs = {
            "sigma0=0.0": recs,
            "sigma0=0.3": [
                {"step": 25, "loss": 0.8, "cos": 0.5, "rel_in": 0.9,
                 "move_ratio": 0.95},
                {"step": 50, "loss": 0.6, "cos": 0.6, "rel_in": 0.7,
                 "move_ratio": 0.97},
                {"step": 75, "loss": 0.5, "cos": 0.65, "rel_in": 0.6,
                 "move_ratio": 0.98},
            ],
        }
        sweep_png = Path(td) / "sweep.png"
        plot_sweep_comparison(runs, sweep_png)
        assert sweep_png.exists() and sweep_png.stat().st_size > 0

        # An empty run (process just started, log.jsonl has nothing valid
        # yet) must be dropped, not crash the render.
        plot_sweep_comparison({**runs, "sigma0=0.6": []}, sweep_png)
        assert sweep_png.exists()

        # cos_clean preferred over cos when present -- the whole point of
        # the fallback logic (a sigma0 sweep must compare on cos_clean,
        # not the noised "cos", or it's comparing different quantities).
        runs_clean = {
            "sigma0=0.0": [{"step": 25, "loss": 0.5, "cos": 0.3,
                            "cos_clean": 0.3, "rel_in": 0.9,
                            "move_ratio": 0.8}],
            "sigma0=0.7": [{"step": 25, "loss": 0.6, "cos": 0.6,
                            "cos_clean": 0.35, "rel_in": 0.85,
                            "move_ratio": 0.82}],
        }
        clean_png = Path(td) / "clean.png"
        plot_sweep_comparison(runs_clean, clean_png)
        assert clean_png.exists() and clean_png.stat().st_size > 0

    # summarize_diagnostics / print_diagnostics_summary: the pure
    # aggregation half of `diagnose`, testable without a GPU or checkpoint.
    # NaNs (degenerate examples, e.g. an empty mask) must be dropped, not
    # propagate into the summary.
    raw = np.array([0.5, 0.6, float("nan"), 0.4])
    pooled = np.array([0.8, float("nan"), 0.7, 0.9])
    summary = summarize_diagnostics(raw, pooled)
    assert summary["raw"]["n"] == 3
    assert abs(summary["raw"]["mean"] - 0.5) < 1e-9
    assert summary["pooled"]["n"] == 3
    assert "centered" not in summary  # omitted entirely when not passed

    summary_c = summarize_diagnostics(raw, pooled, np.array([0.2, 0.3]))
    assert summary_c["centered"]["n"] == 2

    # all-NaN input (e.g. every mask in a batch was empty) must report
    # n=0, not raise or produce a NaN mean that would silently poison a
    # printed report.
    all_nan = np.array([float("nan"), float("nan")])
    summary_empty = summarize_diagnostics(all_nan, all_nan)
    assert summary_empty["raw"]["n"] == 0
    print_diagnostics_summary(summary_empty)  # must not raise
    print_diagnostics_summary(summary_c)       # must not raise

    # find_dead_lora_b / mask_enc_status: the checkpoint inspector's pure
    # logic, testable on a fake adapter state dict without a real
    # checkpoint or GPU.
    import torch as _torch
    fake_sd = {
        "block0.attn.lora_b.weight": _torch.zeros(4, 4),
        "block1.attn.lora_b.weight": _torch.randn(4, 4),
        "block2.mlp.lora_b.weight": _torch.zeros(4, 4),
        "mask_enc.body.0.weight": _torch.randn(8, 3, 3, 3),
        "mask_enc.body.0.bias": _torch.randn(8),
        "mask_enc.out.weight": _torch.zeros(4, 8, 3, 3),
        "mask_enc.out.bias": _torch.zeros(4),
    }
    dead = find_dead_lora_b(fake_sd)
    assert dead == ["block0.attn.lora_b.weight", "block2.mlp.lora_b.weight"], dead

    status = mask_enc_status(fake_sd)
    assert set(status) == {"mask_enc.body.0.weight", "mask_enc.body.0.bias",
                           "mask_enc.out.weight", "mask_enc.out.bias"}
    assert status["mask_enc.out.weight"]["zero_init_layer"] is True
    assert status["mask_enc.out.weight"]["moved"] is False
    assert status["mask_enc.body.0.weight"]["zero_init_layer"] is False
    assert status["mask_enc.body.0.weight"]["moved"] is None

    # a mask_enc whose out layer DID move must report moved=True.
    fake_sd_trained = dict(fake_sd)
    fake_sd_trained["mask_enc.out.weight"] = _torch.randn(4, 8, 3, 3)
    status_trained = mask_enc_status(fake_sd_trained)
    assert status_trained["mask_enc.out.weight"]["moved"] is True

    # no mask_enc keys at all (use_mask=False) must return {}, not raise.
    assert mask_enc_status({"block0.lora_b.weight": _torch.randn(2, 2)}) == {}

    print("gate_a_transport self-test OK")


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
    t.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes for THIS run. Was "
                        "hardcoded at 4; now tunable because sweep_sigma0.sh "
                        "runs several of these concurrently -- N runs x "
                        "this many workers each are all competing for CPU/IO "
                        "on the same box. Lower it (e.g. 2) when running a "
                        "wide sweep if you see contention.")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--max-blocks", type=int, default=None)
    t.add_argument("--mask-dropout", type=float, default=0.1)
    t.add_argument("--cond-dropout", type=float, default=0.1)
    t.add_argument("--grad-checkpointing", action="store_true", default=True)
    t.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false",
                   help="faster, more activation memory. Also the "
                        "workaround for this diffusers install's "
                        "checkpointed-branch tuple bug -- see cmd_train's "
                        "own comment for the exact error and why")
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

    w = sub.add_parser("watch-log")
    w.add_argument("--out", default="runs/gate_a",
                   help="same --out directory passed to `train`; reads "
                        "<out>/log.jsonl")
    w.add_argument("--every", type=int, default=100,
                   help="redraw after at least this many new steps are "
                        "logged since the last draw")
    w.add_argument("--poll-seconds", type=float, default=10.0,
                   help="how often to check the log file for new data")
    w.add_argument("--png", default="training_curve.png",
                   help="output filename, written inside --out")
    w.add_argument("--once", action="store_true",
                   help="draw once and exit instead of polling forever")
    w.set_defaults(fn=cmd_watch_log)

    ws = sub.add_parser("watch-sweep")
    ws.add_argument("--outs", nargs="+", required=True,
                    help="multiple --out directories to compare, e.g. the "
                         "ones sweep_sigma0.sh launched")
    ws.add_argument("--labels", nargs="+", default=None,
                    help="one label per --outs entry, same order; default "
                         "is each directory's own basename")
    ws.add_argument("--every", type=int, default=100,
                    help="redraw once ANY run has logged this many new "
                         "steps since the last draw")
    ws.add_argument("--poll-seconds", type=float, default=10.0)
    ws.add_argument("--png", default="runs/sigma0_sweep_comparison.png")
    ws.add_argument("--once", action="store_true",
                    help="draw once and exit instead of polling forever")
    ws.set_defaults(fn=cmd_watch_sweep)

    d = sub.add_parser("diagnose")
    d.add_argument("--ckpt", nargs="+", required=True,
                   help="one or more checkpoint paths; the base transformer "
                        "loads once and each checkpoint's adapter weights "
                        "are swapped in, with a comparison table printed "
                        "when more than one is given")
    d.add_argument("--cache", required=True,
                   help="TripletCache directory -- the same one used for "
                        "training/ambiguity_probe.py, no VAE or text "
                        "encoder needed since latents+embeddings are cached")
    d.add_argument("--bias-npy", default=None,
                   help="per-channel bias vector (.npy) from "
                        "ambiguity_probe.py --save-bias, for the centered "
                        "treatment; omit to report raw+pooled only")
    d.add_argument("--batch-size", type=int, default=8)
    d.add_argument("--num-workers", type=int, default=2)
    d.add_argument("--rank", type=int, default=16)
    d.add_argument("--device", default="cuda")
    d.add_argument("--model", default=common["model"])
    d.set_defaults(fn=cmd_diagnose)

    i = sub.add_parser("inspect")
    i.add_argument("--ckpt", nargs="+", required=True,
                   help="one or more checkpoint paths. CPU-only: no model, "
                        "no GPU needed, just torch.load + tensor stats")
    i.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("self-test")
    s.set_defaults(fn=lambda _args: self_test())

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
