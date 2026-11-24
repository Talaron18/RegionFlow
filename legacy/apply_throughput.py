#!/usr/bin/env python3
"""Apply the throughput edits to scripts/gate_b_meanflow.py by anchor text.

    python apply_throughput.py                      # from the repo root
    python apply_throughput.py --path some/other/gate_b_meanflow.py
    python apply_throughput.py --dry-run

Use this instead of `git apply throughput.patch` when the patch is rejected:
git matches surrounding LINES, which drift if the file differs anywhere near an
edit. This matches unique ANCHOR STRINGS instead, applies each edit
independently, skips edits already present, and writes nothing unless every
remaining edit is found.

Edits (all default-off; an unflagged run trains exactly as before):
  1. DataLoader: --num-workers, persistent_workers, prefetch_factor
  2. --no-grad-checkpointing (fd estimator only; refused with jvp)
  3. --batch help text, --num-workers, --bench-steps flags
  4. run_bench(): time the real training step, print s/step and peak VRAM
  5. bench mode routed before Trainer construction, into a temp directory
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

EDITS = [
    (
        "loader + workers",
        """    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
""",
        """    # THROUGHPUT. batch x accum is the optimiser batch and is unchanged by
    # how it is split: the loss is a per-sample mean, so batch=8/accum=2 and
    # batch=2/accum=8 give the same gradient. Bigger micro-batches are
    # strictly better GPU utilisation IF they fit -- the 5k run peaked at
    # 9.5 of 24 GB with batch=2, so there is headroom. Measure, do not assume:
    # --bench-steps times a few steps and exits.
    #
    # persistent_workers keeps the loader workers alive across epochs. The
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
""",
    ),
    (
        "optional gradient checkpointing",
        """    enable_memory_savers(adapter.transformer)
""",
        """    # Gradient checkpointing trades ~30-40% step time for activation memory.
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
""",
    ),
    (
        "CLI flags",
        """    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
""",
        """    t.add_argument("--batch", type=int, default=2,
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
""",
    ),
    (
        "run_bench()",
        """def cmd_train(args) -> int:""",
        '''def run_bench(adapter, loader, tcfg, loss_fn, args) -> int:
    """Time the real training step. No checkpoints, no logs, no state kept.

    Runs the same Trainer inner loop -- micro-batches, accumulation, clip,
    Adam, EMA -- so the number is the step time training would actually get,
    not a forward-pass proxy.
    """
    import tempfile
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
    print(f"\\n[bench] {dt:.2f} s/step  ({args.batch}x{args.accum} = "
          f"{args.batch * args.accum}/step, {dt / (args.batch * args.accum):.3f} "
          f"s/sample)  peak VRAM {peak:.1f} GiB  "
          f"checkpointing={'on' if args.grad_checkpointing else 'off'}")
    print(f"[bench] 5000 steps at this rate: {5000 * dt / 3600:.1f} h")
    return 0


def cmd_train(args) -> int:''',
    ),
    (
        "route bench before Trainer",
        """    trainer = Trainer(adapter, loader, tcfg, loss_fn, val_fn=val_fn)
""",
        """    if args.bench_steps:
        return run_bench(adapter, loader, tcfg, loss_fn, args)
    trainer = Trainer(adapter, loader, tcfg, loss_fn, val_fn=val_fn)
""",
    ),
]

# Optional: only present if distill_arm.patch was applied. Keeps the bench from
# writing meanflow_config.json into a real run directory.
OPTIONAL = [
    (
        "bench skips provenance write",
        """    cfg_now = dataclasses.asdict(ref)
""",
        """    cfg_now = dataclasses.asdict(ref)
    if args.bench_steps:
        import tempfile as _tf
        cfg_path = Path(_tf.mkdtemp(prefix="gateb_bench_")) / "cfg.json"
""",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="scripts/gate_b_meanflow.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}  (run from the repo root, or pass --path)")
        return 1
    src = path.read_text()
    print(f"{path}  sha256 {hashlib.sha256(src.encode()).hexdigest()[:12]}")

    out, applied, skipped, missing = src, [], [], []
    for name, old, new in EDITS:
        if new in out:
            skipped.append(name)
        elif out.count(old) == 1:
            out = out.replace(old, new)
            applied.append(name)
        else:
            missing.append((name, out.count(old), old))
    for name, old, new in OPTIONAL:
        if new in out:
            skipped.append(name)
        elif out.count(old) == 1:
            out = out.replace(old, new)
            applied.append(name)
        else:
            skipped.append(f"{name} (not applicable)")

    for name in applied:
        print(f"  applied  {name}")
    for name in skipped:
        print(f"  skipped  {name} (already present)")
    if missing:
        for name, n, old in missing:
            print(f"  FAILED   {name}: anchor found {n} times, expected 1")
            print("           anchor was:")
            for line in old.splitlines()[:4]:
                print(f"             {line}")
        print("\nnothing written. The anchor above does not appear verbatim in "
              "your file -- paste those lines as they actually read and the "
              "edit can be re-cut against them.")
        return 1

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    if not applied:
        print("\nall edits already present; nothing to do")
        return 0
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(src)
    path.write_text(out)
    print(f"\nwrote {path} ({len(applied)} edits), backup at {backup}")
    print("now run:  python -m py_compile", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
