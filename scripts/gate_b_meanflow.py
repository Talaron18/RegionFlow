"""
Gate B -- how far can a pretrained instantaneous field be stretched into an
interval field at LoRA rank?

GRADED, NOT PASS/FAIL. The question is not "does MeanFlow conversion work". It
is "what is the largest interval Delta at which the converted model still
agrees with the teacher's flow map". A pass/fail gate at Delta = 1 would
collapse LoRA capacity, learning rate, r-conditioning design and genuine
representational mismatch into a single uninterpretable bit. The curve is the
result, whether it ends at 0.1 or at 1.0.

    python scripts/probe_jvp_mmdit.py            # FIRST. decides --estimator
    python scripts/gate_b_meanflow.py train --cache /data/cache/<hash> \
        --estimator fd --delta-end 1.0 --out runs/gate_b --max-hours 5.5
    python scripts/gate_b_meanflow.py sweep --ckpt runs/gate_b/ckpt_final.pt

THE MEASUREMENT. Fix a set of noise seeds. Run the FROZEN teacher at heun-50
with CFG to get reference images -- the same operating point Stage 0 settled on,
so the reference is a known quantity. Run the student from the same noise at
k in {1, 2, 4, 8} steps, guidance 1.0. Report PSNR and LPIPS against the
teacher's output, and NFE alongside.

This measures flow-map AGREEMENT, not distributional quality. That is the right
question here (has the student learned the teacher's transport?) and it needs
no reference dataset and no FID sample budget, neither of which this project
has. FID belongs in the paper; agreement belongs in the loop.

BOUNDARY MONITOR. u(z, t, t, c) must equal the base velocity exactly, for all
parameters, forever -- it is enforced by the gate, not learned. The sweep
asserts it anyway on every checkpoint. If it ever fails, the gating has been
broken by an edit and every number in the run is void.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter, freeze_and_report  # noqa: E402
from transport.data import TripletCache, torch_dataset  # noqa: E402
from transport.meanflow import (                    # noqa: E402
    MeanFlowConfig, meanflow_loss, sample_interval,
)
from transport.meanflow_ref import (                # noqa: E402
    RefMeanFlowConfig, cond_drop, guided_velocity, ref_meanflow_loss,
)
from transport.sampling import sample, shifted_grid, uniform_grid  # noqa: E402
from transport.train import (                       # noqa: E402
    TrainConfig, Trainer, delta_schedule, enable_memory_savers,
)


def build_adapter(model, device, rank, max_blocks=None, dtype="float32",
                  mode="meanflow", interval_kind="concat"):
    """Load the frozen transformer.

    FLOAT32 IS REQUIRED FOR estimator='jvp'. Forward-mode AD promotes its
    tangents to float32, so with bfloat16 weights the two meet at the first
    attention projection and raise

        expected mat1 and mat2 to have the same dtype: float != BFloat16

    which is exactly what probe 1 and 2 hit before they were rerun with
    --dtype float32. The probe recorded "use jvp"; the dtype that made jvp
    work was not carried across to here, and this is that omission.

    Probe 2 measured 8.7 GB peak for the fp32 JVP under the math attention
    backend at 512px batch 1, inference only. Training adds adapter
    gradients, optimiser state (~245 MB for 20.4M fp32 Adam params) and
    checkpointed activations on top. If that overflows 24 GB, drop to
    batch 1 with more accumulation before reaching for anything else.
    """
    import torch
    from diffusers import SD3Transformer2DModel

    tr = SD3Transformer2DModel.from_pretrained(
        model, subfolder="transformer",
        torch_dtype=getattr(torch, dtype)).to(device)
    ad = TransportAdapter(
        tr, AdapterConfig(mode=mode, rank=rank, max_blocks=max_blocks,
                          interval_kind=interval_kind)
    ).to(device)
    for p in ad.trainable_parameters():
        p.data = p.data.float()
    print(freeze_and_report(ad))
    return ad


#: SD3's real context length: 77 (merged CLIP-L/CLIP-G) + 256 (T5, at the
#: default max_sequence_length) = 333. Matches transport.data.CacheConfig's
#: default, which precompute_cache.py's writer asserts against at cache-build
#: time -- so a mismatch here means encode_null and the cache disagree, not
#: that either one is wrong in isolation.
SD3_CONTEXT_LENGTH = 333


def encode_null(model_id, device):
    """The unconditional embedding for CFG, computed once.

    The reference uses a learned null CLASS embedding. SD3's unconditional is
    the EMPTY STRING through its three text encoders, which is not the same as
    a zero vector -- earlier code in this file used zeros, which is a silent
    deviation that changes the guidance target. Loads the encoders in
    bfloat16, encodes "", frees them; the transformer is loaded afterwards in
    float32, as in scripts/probe_field_stiffness.py.

    THE T5 BRANCH IS REPRODUCED BY HAND, not called through
    pipe.encode_prompt. Confirmed by reading the actual installed source (the
    UltraEdit-vendored diffusers==0.30.0.dev0 at
    /data/repos/UltraEdit/diffusers, which `from diffusers import ...`
    resolves to on this box and is liable to for any script, not just Gate
    C's): this snapshot predates `max_sequence_length` as a T5 parameter.
    `_get_t5_prompt_embeds` instead hardcodes
    `max_length=self.tokenizer_max_length` -- the SAME attribute
    `_get_clip_prompt_embeds` uses for CLIP's 77-token cap, visible in the T5
    helper's own warning string, copy-pasted from the CLIP one: "CLIP can
    only handle sequences up to {tokenizer_max_length} tokens", raised from
    inside the T5 branch. Setting `pipe.tokenizer_max_length = 256` to widen
    T5 would widen CLIP's cap along with it, past what CLIP's 77-position
    embedding table supports.

    So CLIP goes through the pipeline's own `_get_clip_prompt_embeds`,
    untouched, exactly as `encode_prompt` calls it (two calls, one per CLIP
    encoder, concatenated on the feature axis). T5 is tokenized and encoded
    here directly at max_length=256, via `pipe.tokenizer_3` /
    `pipe.text_encoder_3` -- copied verbatim from this installed version's
    `_get_t5_prompt_embeds` (padding, truncation and dtype-cast identical),
    with only `self.tokenizer_max_length` replaced by 256. The two are then
    padded/concatenated exactly as `encode_prompt`'s own "prompt_embeds is
    None" branch does. `do_classifier_free_guidance=False` in the original
    call meant only that branch ever ran -- the CFG/negative-prompt branch
    is not reproduced because encode_null never used it.

    If this diffusers install changes again, this breaks by construction
    (private methods, not the public API) -- which is the point: better an
    AttributeError naming which private method moved than a silent
    context-length mismatch two calls downstream in cond_drop.
    """
    import gc

    import torch
    import torch.nn.functional as tf
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, transformer=None, vae=None,
    ).to(device)
    with torch.no_grad():
        pe0, po0 = pipe._get_clip_prompt_embeds(
            prompt=[""], device=device, num_images_per_prompt=1,
            clip_skip=None, clip_model_index=0)
        pe1, po1 = pipe._get_clip_prompt_embeds(
            prompt=[""], device=device, num_images_per_prompt=1,
            clip_skip=None, clip_model_index=1)
        clip_pe = torch.cat([pe0, pe1], dim=-1)
        po = torch.cat([po0, po1], dim=-1)

        t5_in = pipe.tokenizer_3(
            [""], padding="max_length", max_length=256, truncation=True,
            add_special_tokens=True, return_tensors="pt")
        t5_pe = pipe.text_encoder_3(t5_in.input_ids.to(device))[0]
        t5_pe = t5_pe.to(dtype=pipe.text_encoder_3.dtype, device=device)

        clip_pe = tf.pad(clip_pe, (0, t5_pe.shape[-1] - clip_pe.shape[-1]))
        pe = torch.cat([clip_pe, t5_pe], dim=-2)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if pe.shape[-2] != SD3_CONTEXT_LENGTH:
        raise RuntimeError(
            f"encode_null produced context length {pe.shape[-2]}, expected "
            f"{SD3_CONTEXT_LENGTH} even after bypassing "
            "_get_t5_prompt_embeds. The installed diffusers must have "
            "changed shape again -- re-read "
            "StableDiffusion3Pipeline.encode_prompt and "
            "._get_t5_prompt_embeds's source on this box before touching "
            "this function further.")
    return pe.float(), po.float()


def run_bench(adapter, loader, tcfg, loss_fn, args) -> int:
    """Time the real training step. No checkpoints, no logs, no state kept.

    Runs the same Trainer inner loop -- micro-batches, accumulation, clip,
    Adam, EMA -- so the number is the step time training would actually get,
    not a forward-pass proxy.
    """
    import time

    import torch

    from transport.train import Trainer

    tcfg.steps = args.bench_steps + 3
    tcfg.val_every = 10 ** 9
    tcfg.ckpt_every = 10 ** 9
    # never touch the real run directory: Trainer creates out_dir on init
    tcfg.out_dir = tempfile.mkdtemp(prefix="gateb_bench_")
    trainer = Trainer(adapter, loader, tcfg, loss_fn)
    it = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(tcfg.steps):
        if i == 3:                       # warmup: allocator, workers, autotune
            torch.cuda.synchronize()
            t0 = time.time()
        trainer.opt.zero_grad(set_to_none=True)
        for _ in range(tcfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            loss, _ = loss_fn(adapter, batch, i, tcfg)
            (loss / tcfg.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(trainer.params, tcfg.grad_clip)
        trainer.opt.step()
        trainer.ema.update(trainer.params)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.bench_steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n[bench] {dt:.2f} s/step  ({args.batch}x{args.accum} = "
          f"{args.batch * args.accum}/step, {dt / (args.batch * args.accum):.3f} "
          f"s/sample)  peak VRAM {peak:.1f} GiB  "
          f"checkpointing={'on' if args.grad_checkpointing else 'off'}")
    print(f"[bench] 5000 steps at this rate: {5000 * dt / 3600:.1f} h")
    return 0


def cmd_train(args) -> int:
    import diffusers
    import torch
    from torch.utils.data import DataLoader

    # PROVENANCE, printed before anything else can fail. STAGE1C Sec.5 found
    # two boxes disagreeing on sse_base by 6.1% -- larger than effects being
    # measured -- and concluded that rows from different boxes must not share
    # a table. `import diffusers` is the part of this environment that has
    # moved most: which copy resolves first on sys.path decides the SD3
    # pipeline's text-encoder context length AND whether its transformer's
    # checkpointed branch unpacks, both of which have already broken runs
    # here. An arm-to-arm comparison is only valid if this line reads the
    # same for every arm, so it goes in every run's stdout rather than being
    # reconstructed afterwards from which warnings happened to fire.
    print(f"[env] torch {torch.__version__}  "
          f"diffusers {diffusers.__version__}  from {diffusers.__file__}")

    torch.manual_seed(args.seed)
    ds = torch_dataset(TripletCache(args.cache))
    # THROUGHPUT. batch x accum is the optimiser batch and is unchanged by
    # how it is split: the loss is a per-sample mean, so batch=8/accum=2 and
    # batch=2/accum=8 give the same gradient. Bigger micro-batches are
    # strictly better GPU utilisation IF they fit -- the 5k run peaked at
    # 9.5 of 24 GB with batch=2, so there is headroom. Measure, do not assume:
    # --bench-steps times a few steps and exits.
    #
    # persistent_workers keeps the 4 loader workers alive across epochs. The
    # trainer re-iterates the loader on StopIteration, which respawns them
    # every epoch otherwise; at batch=2 an epoch is short.
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    print(f"[data] {len(ds)} cached samples, batch {args.batch} x accum "
          f"{args.accum} = {args.batch * args.accum} per step, "
          f"{args.num_workers} workers")

    if args.estimator == "jvp" and args.dtype != "float32":
        raise SystemExit(
            f"--estimator jvp requires --dtype float32, got {args.dtype}. "
            "Forward-mode AD promotes tangents to fp32 and they cannot meet "
            "bf16 weights. Either pass --dtype float32 or switch estimator -- "
            "but note 'fd' is NOT validated on this model (see "
            "transport/meanflow.py)."
        )
    adapter = build_adapter(args.model, args.device, args.rank,
                            args.max_blocks, dtype=args.dtype,
                            mode="meanflow" if args.faithful else "interval",
                            interval_kind=args.interval_kind)
    # GUARD. Same lesson as the anchor check below: confirm the flag reached
    # the module that was actually built, not just the argparse namespace.
    from transport.conditioning import INTERVAL_KINDS, interval_kind_from_keys
    built = type(adapter.interval).__name__
    want = INTERVAL_KINDS[args.interval_kind].__name__
    if built != want:
        raise RuntimeError(f"--interval-kind {args.interval_kind} built {built}, "
                           f"expected {want}")
    print(f"[interval] kind = {args.interval_kind} ({built})")
    resume_ck = Path(args.out) / "ckpt_latest.pt"
    if resume_ck.exists():
        prev = interval_kind_from_keys(
            torch.load(resume_ck, map_location="cpu",
                       weights_only=False)["adapter"].keys())
        if prev is not None and prev != args.interval_kind:
            raise RuntimeError(
                f"{resume_ck} was trained with interval kind {prev!r} but "
                f"--interval-kind is {args.interval_kind!r}. Resuming would "
                "load with strict=False and silently skip the interval "
                "weights. Use a fresh --out.")
    # Gradient checkpointing trades ~30-40% step time for activation memory.
    # It is REQUIRED for --estimator jvp (forward-mode AD is taken under
    # no_grad by checkpoint_safe, but the backward still runs through the
    # checkpointed blocks). With --estimator fd there is no forward-mode AD
    # anywhere, so it is purely a memory/speed dial.
    if args.grad_checkpointing:
        enable_memory_savers(adapter.transformer)
    else:
        if args.estimator == "jvp":
            raise SystemExit("--no-grad-checkpointing with --estimator jvp is "
                             "not supported; use --estimator fd")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[train] gradient checkpointing OFF -- if this OOMs, lower "
              "--batch and raise --accum by the same factor")
    mf = MeanFlowConfig(estimator=args.estimator, fd_eps=args.fd_eps,
                        adaptive_p=args.adaptive_p)

    npe1, npo1 = (encode_null(args.model, args.device)
                  if args.faithful else (None, None))
    ref = RefMeanFlowConfig(estimator=args.estimator, fd_eps=args.fd_eps,
                            omega=args.omega, kappa=args.kappa,
                            cond_dropout_prob=args.cond_dropout,
                            data_proportion=args.data_proportion,
                            tangent=args.tangent, max_delta=args.max_delta,
                            max_t=args.max_t, first_term=args.first_term,
                            norm_p=args.norm_p)
    print(f"[meanflow] data_proportion = {ref.data_proportion} "
          f"(reference: 0.75)")

    from delta_curriculum import (DeltaCurriculumConfig, assert_flag_arrived,
                                  curriculum_stats, delta_cap,
                                  sample_tr_scheduled)

    cur = DeltaCurriculumConfig(
        enabled=args.curriculum,
        cap_start=args.cap_start,
        cap_end=args.cap_end,
        warmup_frac=args.warmup_frac,
        anchor_mode=args.anchor_mode,
        anchor_width=args.anchor_width,
        # ALWAYS, not only when enabled: curriculum-off has to be THIS run's
        # baseline (sample_tr with its caps), not the uncapped reference.
        hard_max=args.max_delta,
    )
    # Fail here, not at step 0 inside loss_fn_ref's per-batch guard. A cap
    # above --max-delta is silently a no-op curriculum once delta_cap clamps
    # it -- the schedule would flatten to a constant and the run would look
    # normal while testing nothing.
    if max(args.cap_start, args.cap_end) > args.max_delta + 1e-9:
        raise SystemExit(
            f"--cap-start {args.cap_start} / --cap-end {args.cap_end} exceed "
            f"--max-delta {args.max_delta}. The cap is bounded by max_delta, "
            "so the schedule would flatten against that ceiling and the "
            "curriculum would be a no-op at the wide end. Set the caps at or "
            "below --max-delta (for a 'no cap, anchors only' arm, pass "
            f"--cap-start {args.max_delta} --cap-end {args.max_delta}).")
    assert_flag_arrived(args, ref, [("data_proportion", "data_proportion")])
    assert_flag_arrived(args, cur, [("cap_start", "cap_start"),
                                    ("cap_end", "cap_end"),
                                    ("warmup_frac", "warmup_frac"),
                                    ("anchor_width", "anchor_width"),
                                    ("max_delta", "hard_max")])

    # Dedicated RNG stream for (t, r). Without this, "small_delta" draws one
    # extra torch.rand per step off the GLOBAL stream, so every subsequent
    # `eps = torch.randn_like(x0)` differs from the other arms and the three
    # runs stop being noise-matched from step 1. Keep noise on the global
    # stream, schedule on this one.
    tr_gen = torch.Generator(device=args.device).manual_seed(args.seed + 1)

    print(f"[curriculum] enabled={cur.enabled} cap {cur.cap_start}->{cur.cap_end} "
          f"warmup_frac={cur.warmup_frac} anchor_mode={cur.anchor_mode} "
          f"anchor_width={cur.anchor_width} hard_max={cur.hard_max} "
          f"(cap at step 0 = {delta_cap(0, args.steps, cur):.4f})")
    print(f"[args] {vars(args)}")
    # GUARD. Every target-shaping flag, read back from the config object the
    # loss will actually receive -- the P10 run is why this exists.
    import dataclasses
    for flag, field in (("estimator", "estimator"), ("fd_eps", "fd_eps"),
                        ("tangent", "tangent"), ("max_delta", "max_delta"),
                        ("max_t", "max_t"), ("first_term", "first_term"),
                        ("norm_p", "norm_p"),
                        ("data_proportion", "data_proportion")):
        if getattr(args, flag) != getattr(ref, field):
            raise RuntimeError(f"--{flag.replace('_', '-')} "
                               f"{getattr(args, flag)!r} did not reach the "
                               f"loss config ({getattr(ref, field)!r})")
    print(f"[meanflow] target: estimator={ref.estimator} fd_eps={ref.fd_eps} "
          f"tangent={ref.tangent} first_term={ref.first_term} "
          f"norm_p={ref.norm_p} max_delta={ref.max_delta} max_t={ref.max_t}")
    if ref.first_term == "frozen" and ref.norm_p != 0.0:
        # With e - x0 as first term every sample's sse sits near the
        # flow-matching floor (~2e4), so 1/(sse + eps) is nearly uniform and
        # norm_p=1 behaves like plain MSE. A frozen first term removes that
        # floor: anchor sse starts at 0, and 1/(sse + 0.01) would hand anchors
        # up to ~100x the weight of interval samples. norm_p=0 is the setting
        # that reproduces the weighting the fd run actually had.
        print("[meanflow] WARNING: first_term=frozen with norm_p="
              f"{ref.norm_p}: adaptive weights will concentrate on near-zero-"
              "error anchors. Watch adp_ess; --norm-p 0 is the matched choice.")
    if ref.estimator != "jvp" and ref.fd_eps < 0.01:
        print(f"[meanflow] note: fd_eps={ref.fd_eps} is below the smoothing "
              "range the target probe measured (0.01-0.05); a fine stencil "
              "sees the same fast components the JVP does")
    # Provenance. The checkpoint stores TrainConfig only; the target settings
    # live here, and a resume under different settings is refused.
    cfg_path = Path(args.out) / "meanflow_config.json"
    cfg_now = dataclasses.asdict(ref)
    if args.bench_steps:
        cfg_path = Path(tempfile.mkdtemp(prefix="gateb_bench_")) / "cfg.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        prev_cfg = json.loads(cfg_path.read_text())
        changed = {k: (prev_cfg.get(k), v) for k, v in cfg_now.items()
                   if prev_cfg.get(k) != v}
        if changed:
            raise RuntimeError(f"{cfg_path} records a different target "
                               f"config: {changed}. Use a fresh --out.")
    else:
        cfg_path.write_text(json.dumps(cfg_now, indent=1))

    def loss_fn_ref(model, batch, step, cfg):
        """Reference-faithful MeanFlow step. No curriculum, 75% anchors."""
        dev, mdt = args.device, model.model_dtype
        x0 = batch["z_tgt"].to(dev).float()
        b = x0.shape[0]
        pe = batch["prompt_embeds"].to(dev, mdt)
        po = batch["pooled"].to(dev, mdt)
        npe = npe1.to(dev, mdt).expand(b, -1, -1)
        npo = npo1.to(dev, mdt).expand(b, -1)

        t, r, anchor = sample_tr_scheduled(b, ref, dev, step, cfg.steps, cur,
                                           generator=tr_gen)
        eps = torch.randn_like(x0)
        tt = t.reshape(-1, 1, 1, 1)
        z_t = (1 - tt) * x0 + tt * eps
        v = eps - x0

        def base_fn(z_, t_, emb):
            e, p = emb
            return model.base_velocity(z_, t_, e, p)

        v_g = guided_velocity(v, z_t, t, base_fn, (pe, po), (npe, npo), ref)
        (pe_in, po_in), v_g = cond_drop(v, v_g, (pe, po), (npe, npo), ref)

        loss, stats = ref_meanflow_loss(
            lambda z_, t_, r_: model(z_, t_, r_, pe_in, po_in),
            z_t, t, r, v_g, ref,
            base_fn=lambda z_, t_: model.base_velocity(z_, t_, pe_in, po_in),
            anchor_mask=anchor)
        # ref_meanflow_loss infers anchors from (t - r) < 1e-8. Under
        # anchor_mode="small_delta" that is False for every anchor: frac_anchor
        # reads ~0, the guard below kills the run at batch 50, and sse_anchor --
        # the series this experiment exists to measure -- is never logged.
        # The sampler knows which samples are anchors; use its mask.
        stats["frac_anchor"] = float(anchor.float().mean())
        stats.update(curriculum_stats(t, r, anchor, step, cfg.steps, cur))
        # GUARD, per batch: a cap that did not reach the sampler.
        if stats["delta_max"] > ref.max_delta + 1e-6:
            raise RuntimeError(f"delta {stats['delta_max']:.4f} exceeds "
                               f"--max-delta {ref.max_delta}")
        loss_fn_ref.tangent_sum += stats["tangent_rel_vg"]
        loss_fn_ref.first_sum += stats["first_rel_vg"]
        # GUARD. A "P10" run that silently trained at 75% anchors produced a
        # full set of plausible logs, an eval, and a written interpretation
        # before frac_anchor was checked. The realised fraction is measured
        # per batch and noisy at small batch, so compare a running mean and
        # fail loudly rather than trusting the flag that was passed.
        loss_fn_ref.seen += 1
        loss_fn_ref.anchor_sum += stats["frac_anchor"]
        if loss_fn_ref.seen == 50:
            got = loss_fn_ref.anchor_sum / loss_fn_ref.seen
            if abs(got - ref.data_proportion) > 0.12:
                raise RuntimeError(
                    f"anchor fraction is {got:.3f} after 50 batches but "
                    f"--data-proportion asked for {ref.data_proportion}. The "
                    "config did not reach the loss. Stopping now rather than "
                    "producing a run that looks fine and answers a different "
                    "question."
                )
            print(f"[meanflow] anchor fraction check: {got:.3f} vs requested "
                  f"{ref.data_proportion} -- ok")
            # GUARD: the tangent. 'cond' must give exactly 0; any other mode
            # must not. A 'frozen' or 'boundary' tangent differs from e - x0
            # by the flow-matching noise, which is O(1) relative.
            tan = loss_fn_ref.tangent_sum / loss_fn_ref.seen
            if (ref.tangent == "cond") != (tan == 0.0):
                raise RuntimeError(
                    f"tangent={ref.tangent} but ||w - v_g||/||v_g|| averages "
                    f"{tan:.3g} over 50 batches; the tangent did not reach "
                    "the loss")
            print(f"[meanflow] tangent check: {ref.tangent}, "
                  f"||w - v_g||/||v_g|| = {tan:.3f} -- ok")
            fst = loss_fn_ref.first_sum / loss_fn_ref.seen
            if (ref.first_term == "cond") != (fst == 0.0):
                raise RuntimeError(
                    f"first_term={ref.first_term} but ||first - v_g||/||v_g|| "
                    f"averages {fst:.3g} over 50 batches; the first term did "
                    "not reach the loss")
            print(f"[meanflow] first-term check: {ref.first_term}, "
                  f"||first - v_g||/||v_g|| = {fst:.3f} -- ok")
        return loss, stats

    def loss_fn_custom(model, batch, step, cfg):
        dev, mdt = args.device, model.model_dtype
        x0 = batch["z_tgt"].to(dev).float()
        pe = batch["prompt_embeds"].to(dev, mdt)
        po = batch["pooled"].to(dev, mdt)
        max_delta = delta_schedule(step, cfg)
        t, r = sample_interval(x0.shape[0], dev, max_delta,
                               boundary_frac=mf.boundary_frac)
        eps = torch.randn_like(x0)
        t_ = t.reshape(-1, 1, 1, 1)
        z_t = (1 - t_) * x0 + t_ * eps
        v = (model.base_velocity(z_t, t, pe, po, guidance=args.teacher_cfg,
                                 negative_embeds=torch.zeros_like(pe),
                                 negative_pooled=torch.zeros_like(po))
             if args.v_source == "teacher" else eps - x0)
        loss, stats = meanflow_loss(
            lambda z_, r_, t_in: model(z_, t_in, r_, pe, po), z_t, r, t, v, mf)
        stats["max_delta"] = max_delta
        return loss, stats

    loss_fn_ref.seen = 0
    loss_fn_ref.anchor_sum = 0.0
    loss_fn_ref.tangent_sum = 0.0
    loss_fn_ref.first_sum = 0.0
    loss_fn = loss_fn_ref if args.faithful else loss_fn_custom

    # Held-out validation at a FIXED (t, r), independent of what the batch
    # happens to draw, so the number is comparable across the whole run.
    # Two of them: the anchors (r == t, plain flow matching) and a fixed
    # interval. The anchor number is the one that could not exist under the
    # gated formulation, and it should fall like an ordinary FM loss.
    val_batch = next(iter(DataLoader(ds, batch_size=args.batch, shuffle=False)))

    def val_fn(model, cfg):
        dev, mdt = args.device, model.model_dtype
        x0 = val_batch["z_tgt"].to(dev).float()
        b = x0.shape[0]
        pe = val_batch["prompt_embeds"].to(dev, mdt)
        po = val_batch["pooled"].to(dev, mdt)
        g = torch.Generator(device=dev).manual_seed(1234)
        eps = torch.randn(x0.shape, device=dev, generator=g)
        out = {}
        for name, delta in (("anchor", 0.0), ("interval", cfg.val_delta)):
            t = torch.full((b,), 0.5 + delta / 2, device=dev)
            r = t - delta
            tt = t.reshape(-1, 1, 1, 1)
            z_t = (1 - tt) * x0 + tt * eps
            # Same estimator, fd_eps and tangent as training. Previously this
            # built RefMeanFlowConfig(estimator=...) alone, so an fd run
            # validated at the default fd_eps. It is still a self-referential
            # number and still not fit for checkpoint selection.
            _, st = ref_meanflow_loss(
                lambda z_, t_, r_: model(z_, t_, r_, pe, po),
                z_t, t, r, eps - x0, ref,
                base_fn=lambda z_, t_: model.base_velocity(z_, t_, pe, po))
            out[name] = st["sse"]
        ratio = out["interval"] / max(out["anchor"], 1e-12)
        print(f"    val sse: anchor {out['anchor']:.4g}  "
              f"interval(d={cfg.val_delta}) {out['interval']:.4g}  "
              f"ratio {ratio:.1f}")
        # Stash for the trainer to write into log.jsonl. Only `val` was
        # logged before, so val_anchor -- the regulariser check, and the
        # denominator of the only number that answers Gate B -- existed
        # nowhere but stdout.
        val_fn.last = {"val_anchor": out["anchor"],
                       "val_interval": out["interval"],
                       "val_ratio": ratio}
        return out["interval"]

    val_fn.last = {}

    tcfg = TrainConfig(gate="b", out_dir=args.out, steps=args.steps,
                       batch_size=args.batch, grad_accum=args.accum,
                       lr=args.lr, max_hours=args.max_hours, log_every=args.log_every,
                       ckpt_every=args.ckpt_every, seed=args.seed,
                       delta_start=args.delta_start, delta_end=args.delta_end,
                       val_every=args.val_every, val_delta=args.val_delta,
                       keep_last=args.keep_last, grad_clip=args.grad_clip)
    # GUARD, same pattern as the target flags: read back what Trainer gets.
    if tcfg.grad_clip != args.grad_clip:
        raise RuntimeError(f"--grad-clip {args.grad_clip} did not reach "
                           f"TrainConfig ({tcfg.grad_clip})")
    print(f"[train] grad_clip={tcfg.grad_clip}"
          f"{' (disabled)' if tcfg.grad_clip <= 0 else ''}")
    if ref.norm_p == 0.0 and tcfg.grad_clip > 0:
        print("[train] WARNING: norm_p=0 gives gradients ~1e2-1e4; a clip of "
              f"{tcfg.grad_clip} will bind on every step. Watch "
              "grad_clip_frac; --grad-clip 0 matches the norm_p=1 runs, "
              "where the clip never binds.")
    if args.bench_steps:
        return run_bench(adapter, loader, tcfg, loss_fn, args)
    trainer = Trainer(adapter, loader, tcfg, loss_fn, val_fn=val_fn)
    resume = Path(args.out) / "ckpt_latest.pt"
    if resume.exists():
        trainer.load(resume)
    print(f"done: {trainer.run()}")
    return 0


def cmd_sweep(args) -> int:
    import torch
    from diffusers import StableDiffusion3Pipeline

    from bench.inversion import SD3EditPipeline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # bfloat16 is fine HERE: the sweep only samples, no forward-mode AD.
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device)
    helper = SD3EditPipeline(pipe, device=args.device)
    adapter = TransportAdapter(
        pipe.transformer, AdapterConfig(mode="interval", rank=args.rank)
    ).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    adapter.load_state_dict(sd["adapter"], strict=False)
    adapter.eval()

    prompts = json.loads(Path(args.prompts).read_text())
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    rows = []

    for pi, prompt in enumerate(prompts):
        pe, npe, po, npo = helper.encode_prompt(prompt)
        noise = torch.randn(1, 16, 64, 64, device=args.device,
                            generator=g).float()

        # --- boundary assertion, on every checkpoint --------------------
        t = torch.full((1,), 0.5, device=args.device)
        chk = adapter.check_identity(noise, t, pe, po)
        if not chk["boundary_ok"]:
            raise RuntimeError(
                f"BOUNDARY VIOLATED: u(z,t,t) != v_base, max abs err "
                f"{chk['boundary_max_abs_err']:.3e}. The gating in "
                "transport/gating.py has been broken; every number in this "
                "run is void until it is fixed."
            )

        # --- teacher reference, heun-50 + CFG (the Stage 0 operating point)
        ref, ref_nfe = _teacher_sample(helper, noise, pe, npe, po, npo,
                                       steps=args.teacher_steps,
                                       guidance=args.teacher_cfg)

        for k in args.steps_list:
            for grid_name, grid in (("uniform", uniform_grid(k)),
                                    ("shifted", shifted_grid(k))):
                z, nfe = sample(adapter, noise, pe, po, steps=k,
                                guidance=1.0, grid=grid)
                img = helper.decode_latent(z)
                rows.append({
                    "prompt_idx": pi, "steps": k, "grid": grid_name,
                    "nfe": nfe, "teacher_nfe": ref_nfe,
                    "psnr_vs_teacher": _psnr(ref, img),
                    "speedup": ref_nfe / max(nfe, 1),
                })
                from PIL import Image
                Image.fromarray(img).save(
                    out / f"p{pi}_k{k}_{grid_name}.png")
        from PIL import Image
        Image.fromarray(ref).save(out / f"p{pi}_teacher.png")

    (out / "sweep.json").write_text(json.dumps(rows, indent=1))
    _print_table(rows)
    return 0


def _teacher_sample(helper, noise, pe, npe, po, npo, steps, guidance):
    """Frozen SD3 at the Stage 0 operating point. Adapter fully disabled."""
    import torch

    sig = helper._sigmas(steps)
    x = noise.clone().float()
    nfe = 0
    for i in range(len(sig) - 1):
        s, s_next = float(sig[i]), float(sig[i + 1])
        x = helper._step(x, s, s_next, pe, po, solver="heun",
                         guidance=guidance, npe=npe, npool=npo)
        nfe += 4 if guidance > 1 else 2      # heun = 2 calls, CFG doubles
    return helper.decode_latent(x), nfe


def _psnr(a, b):
    a = np.asarray(a, np.float32) / 255.0
    b = np.asarray(b, np.float32) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


def _print_table(rows):
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["steps"], r["grid"])].append(r["psnr_vs_teacher"])
    print(f"\n{'steps':>6}{'grid':>10}{'nfe':>6}{'PSNR vs teacher':>18}")
    for (k, g), vals in sorted(agg.items()):
        print(f"{k:>6}{g:>10}{k:>6}{np.mean(vals):>18.2f}")
    print("\nRead this as the conversion curve, not as a quality claim. The "
          "largest k-step setting that stays close to the teacher is the "
          "Gate B result.")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    model = "stabilityai/stable-diffusion-3-medium-diffusers"

    t = sub.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", default="runs/gate_b")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=2,
                   help="micro-batch. batch x accum is what the optimiser "
                        "sees; raising batch and lowering accum by the same "
                        "factor leaves the gradient unchanged and uses the "
                        "GPU better, if it fits")
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--grad-checkpointing", action="store_true", default=True)
    t.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false",
                   help="faster, more activation memory; fd estimator only")
    t.add_argument("--bench-steps", type=int, default=0,
                   help="time this many steps (after 3 warmup steps), print "
                        "s/step and peak VRAM, then exit without training")
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--max-blocks", type=int, default=None)
    t.add_argument("--estimator", default="jvp", choices=["jvp", "fd", "fd1"])
    t.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="frozen transformer dtype. MUST be float32 for "
                        "estimator=jvp: forward-mode AD promotes tangents to "
                        "fp32 and bf16 weights raise a dtype error at the "
                        "first attention projection.")
    t.add_argument("--fd-eps", type=float, default=1e-3,
                   help="central-difference half-width for --estimator fd. "
                        "The target probe's usable range was 0.025-0.05; "
                        "1e-3 does not smooth")
    t.add_argument("--tangent", default="cond",
                   choices=["cond", "boundary", "frozen"],
                   help="z-tangent of du/dt. cond: e - x0 (reference). "
                        "boundary: u_theta(z,t,t), iMF. frozen: pretrained "
                        "v(z,t), what the probe's marg rows measured")
    t.add_argument("--max-delta", type=float, default=1.0,
                   help="cap on t - r for interval samples (1.0 = reference)")
    t.add_argument("--max-t", type=float, default=1.0,
                   help="cap on t (1.0 = reference)")
    t.add_argument("--first-term", default="cond", choices=["cond", "frozen"],
                   help="first term of the target. cond: e - x0 (reference, "
                        "iMF). frozen: pretrained v(z,t) -- anchors distil "
                        "the teacher. Pair frozen with --norm-p 0")
    t.add_argument("--norm-p", type=float, default=1.0,
                   help="adaptive loss weighting exponent (reference 1.0; "
                        "0 = plain sum-reduced MSE)")
    t.add_argument("--keep-last", type=int, default=3,
                   help="numbered snapshots (every 5x --ckpt-every) to keep")
    t.add_argument("--grad-clip", type=float, default=1.0,
                   help="global grad-norm clip (default 1.0); <= 0 disables. "
                        "Use 0 with --norm-p 0, whose gradients are ~2e4x "
                        "larger than the norm_p=1 runs'")
    t.add_argument("--adaptive-p", type=float, default=0.75)
    t.add_argument("--v-source", default="teacher",
                   choices=["teacher", "empirical"])
    t.add_argument("--teacher-cfg", type=float, default=5.0)
    t.add_argument("--delta-start", type=float, default=0.05)
    t.add_argument("--delta-end", type=float, default=1.0)
    t.add_argument("--curriculum", action="store_true",
                   help="enable the Delta-ORDER curriculum (delta_curriculum.py). "
                        "Off = the reference sampler, bit for bit.")
    t.add_argument("--cap-start", type=float, default=0.05)
    t.add_argument("--cap-end", type=float, default=0.5,
                   help="the operating point Delta = 1/NFE, not a tuning knob. "
                        "1.0 disables capping while leaving --anchor-mode live.")
    t.add_argument("--warmup-frac", type=float, default=0.5)
    t.add_argument("--anchor-mode", default="exact",
                   choices=["exact", "small_delta"])
    t.add_argument("--anchor-width", type=float, default=0.05)
    t.add_argument("--val-every", type=int, default=500,
                   help="steps between held-out validations; 0 disables")
    t.add_argument("--val-delta", type=float, default=0.5,
                   help="FIXED interval width for validation, independent of "
                        "the training curriculum, so the number is comparable "
                        "across the run")
    t.add_argument("--faithful", action="store_true", default=True,
                   help="reference-faithful MeanFlow (default). Uses "
                        "transport/meanflow_ref.py, adapter mode 'meanflow' "
                        "(ungated), 75%% r==t anchors, no curriculum.")
    t.add_argument("--custom", dest="faithful", action="store_false",
                   help="the ORIGINAL gated formulation, kept only as an "
                        "ablation against the faithful baseline. Its anchors "
                        "produce zero adapter gradient by construction.")
    t.add_argument("--interval-kind", default="concat",
                   choices=["concat", "centered"],
                   help="how r reaches pooled_projections. 'concat' is the "
                        "Stage 1 IntervalEmbedding. 'centered' is "
                        "g(t) + [f(t,Delta) - f(t,0)], so the Delta branch "
                        "cannot carry an r-independent offset.")
    t.add_argument("--omega", type=float, default=1.0)
    t.add_argument("--kappa", type=float, default=0.5)
    t.add_argument("--cond-dropout", type=float, default=0.1)
    t.add_argument("--data-proportion", type=float, default=0.75,
                   help="fraction of each batch forced to r == t (the "
                        "flow-matching anchors). Reference value 0.75. Lower "
                        "it to test whether anchors are outvoting the "
                        "interval branch -- and CHECK frac_anchor in the log, "
                        "because a run whose config never reached the loss "
                        "looks completely normal otherwise.")
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--ckpt-every", type=int, default=500,
                   help="numbered snapshots land every 5x this")
    t.add_argument("--max-hours", type=float, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--model", default=model)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("sweep")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--prompts", required=True, help="json list of strings")
    s.add_argument("--out", default="runs/gate_b/sweep")
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--steps-list", type=int, nargs="+", default=[1, 2, 4, 8])
    s.add_argument("--teacher-steps", type=int, default=50)
    s.add_argument("--teacher-cfg", type=float, default=5.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="cuda")
    s.add_argument("--model", default=model)
    s.set_defaults(fn=cmd_sweep)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
