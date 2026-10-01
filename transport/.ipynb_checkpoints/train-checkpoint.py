"""
Training loop for both gates, with checkpoint chaining.

CHAINING IS A HARD REQUIREMENT, NOT A CONVENIENCE. The free 8xA100 allocation
is capped at 6 hours per block and is repeatable. Every run must therefore be
resumable at the granularity of a single optimiser step, including RNG state
and dataloader position, or a multi-block run is not reproducible and its
learning-rate schedule silently restarts each block. `--max-hours` stops the
run cleanly before the wall and writes a resumable checkpoint; rerunning the
identical command picks it up.

DTYPE POLICY, inherited from Stage 0 and non-negotiable:

    transformer forward   bf16     (frozen, memory-bound)
    adapter parameters    fp32     (trained, accumulate)
    optimiser state       fp32
    ODE / path state      fp32     (accumulates across steps at inference)

Stage 0 bug 3 was `x + dt*v` in bf16 losing the increment against an 8-bit
mantissa, and it cost 9 dB on euler-200. Training an adapter in bf16 is the
same failure with a slower feedback loop. Autocast is used for the frozen
forward only.

GRADIENT THROUGH A FROZEN MODEL. The base weights do not require grad, but
activations still do, because the LoRA outputs feed into them. Gradient
checkpointing on the transformer blocks is what makes rank-16 adaptation of all
24 blocks fit on 24GB. Enable it; the ~30% step-time cost is cheaper than
halving the batch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def clip_threshold(grad_clip: float) -> float:
    """max_norm for clip_grad_norm_. grad_clip <= 0 disables clipping.

    Why disabling exists: the threshold is absolute, but the loss scale is
    not. With norm_p=1 every sample is divided by its own SSE (~2e4 at the
    flow-matching floor), so gradients sit near 0.05 and a threshold of 1.0
    never binds. With norm_p=0 the same objective's gradients are ~2e4x
    larger, the threshold binds on every step, and training silently becomes
    Adam on per-step unit-norm gradients -- which down-weights high-SSE
    (long-interval) batches. Adam is invariant to a CONSTANT gradient scale,
    so disabling the clip is what makes the two runs' optimisers match.
    """
    return float(grad_clip) if grad_clip > 0 else math.inf


@dataclass
class TrainConfig:
    gate: str                       # "a" (pairflow) | "b" (meanflow)
    out_dir: str = "runs/stage1"
    steps: int = 20_000
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    warmup: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 0

    # curriculum (gate B only)
    delta_start: float = 0.02
    #: Measured, not assumed. SD3-Medium's field is stiff: ||du/dt|| is ~15x
    #: ||u||, so the MeanFlow correction term (t-r)*du/dt overtakes v at
    #: Delta ~ 0.07. The previous default of 1.0 was a guess made before any
    #: measurement and drove target_norm to 5x the model's output scale.
    #: Set this from the `delta_usable` stat, not from ambition.
    delta_end: float = 0.15
    delta_warmup_frac: float = 0.6   # fraction of training spent widening

    #: Validation for best-checkpoint selection. Evaluated on a FIXED held-out
    #: batch at a FIXED interval width, independent of where the curriculum
    #: has reached, so the number means the same thing at step 500 and step
    #: 8000. Training loss cannot do this job: the curriculum widens Delta as
    #: training proceeds, so training loss measures a different (easier)
    #: problem early on and best-by-training-loss would always pick step ~0.
    val_every: int = 0            # 0 disables
    val_delta: float = 0.5        # fixed; NOT delta_schedule(step)

    # chaining
    max_hours: float | None = None
    ckpt_every: int = 500
    log_every: int = 25
    keep_last: int = 3


def delta_schedule(step: int, cfg: TrainConfig) -> float:
    """Widen the interval geometrically, then hold.

    Geometric rather than linear because the quantity that matters is the ratio
    of the interval to the region where the pretrained field is locally
    accurate, and that region does not grow linearly. Holding for the last 40%
    of training means the final checkpoint has actually converged at the widest
    interval rather than having just arrived there.
    """
    n = max(1, int(cfg.steps * cfg.delta_warmup_frac))
    if step >= n:
        return cfg.delta_end
    frac = step / n
    return float(cfg.delta_start * (cfg.delta_end / cfg.delta_start) ** frac)


class EMA:
    """Exponential moving average of the adapter parameters.

    Few-step samplers are famously sensitive to the exact weights; EMA is
    standard and cheap here because only the adapter is averaged (a few tens of
    MB), not the 2B-parameter base.
    """

    def __init__(self, params, decay: float):
        self.decay = decay
        self.shadow = [p.detach().clone().float() for p in params]

    @torch.no_grad()
    def update(self, params):
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, params):
        for s, p in zip(self.shadow, params):
            p.detach().copy_(s.to(device=p.device, dtype=p.dtype))

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd, params=None):
        """Restore the shadow, ON THE PARAMETERS' DEVICE.

        Trainer.load reads the checkpoint with map_location="cpu", so a
        restored shadow is on CPU while the parameters are on CUDA and the
        first update() raises "Expected all tensors to be on the same
        device". This only ever fires on RESUME -- a fresh run builds the
        shadow from live parameters -- so it survived every short test and
        broke the chained 8-hour run, which is the one path that needs it.
        """
        self.decay = sd["decay"]
        shadow = [t.clone() for t in sd["shadow"]]
        if params is not None:
            shadow = [t.to(p.device) for t, p in zip(shadow, params)]
        self.shadow = shadow


class Trainer:
    def __init__(self, adapter, loader, cfg: TrainConfig, loss_fn,
                 val_fn=None):
        """loss_fn(adapter, batch, step, cfg) -> (loss, stats dict).

        val_fn(adapter, cfg) -> float, lower is better. Called every
        cfg.val_every steps; the best-scoring checkpoint is kept as
        ckpt_valbest.pt alongside the numbered snapshots. Must be deterministic
        -- fixed batch, fixed seed, fixed Delta -- or it selects on noise.

        ckpt_valbest, NOT ckpt_best, and the rename is the point. On
        fd_distill_5k this val (the fixed-Delta interval SSE) ROSE 59.6k ->
        117.8k across the run while the independent field eval
        (eval_interval_field.py) improved monotonically to step 4000-5000, so
        `best_val` picked step 1000 -- and `ckpt_best.pt` was byte-for-byte
        `ckpt_step1000.pt` (same size, same mtime). Anything named "best" gets
        evaluated by someone eventually; this objective's own val is not a
        validated metric, so the file must not claim to be one.
        """
        self.adapter = adapter
        self.loader = loader
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.params = list(adapter.trainable_parameters())
        if not self.params:
            raise RuntimeError(
                "no trainable parameters. The adapter was not injected, and a "
                "run that trains nothing will still produce a loss curve."
            )
        self.opt = torch.optim.AdamW(
            self.params, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), eps=1e-8,
        )
        self.ema = EMA(self.params, cfg.ema_decay)
        self.val_fn = val_fn
        self.best_val = float("inf")
        self.step = 0
        self.t0 = time.time()
        self._log_path = self.out / "log.jsonl"

    # -------------------------------------------------------------- schedule --
    def _lr(self) -> float:
        c = self.cfg
        if self.step < c.warmup:
            return c.lr * (self.step + 1) / c.warmup
        p = (self.step - c.warmup) / max(1, c.steps - c.warmup)
        return c.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # ------------------------------------------------------------ chaining --
    def save(self, tag: str = "latest") -> Path:
        path = self.out / f"ckpt_{tag}.pt"
        torch.save({
            "step": self.step,
            "best_val": self.best_val,
            "config": asdict(self.cfg),
            "adapter": {k: v for k, v in self.adapter.state_dict().items()
                        if v.requires_grad or "lora" in k or "interval" in k
                        or "mask_enc" in k},
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }, path)
        return path

    def load(self, path: str | Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.adapter.load_state_dict(sd["adapter"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
        self.opt.load_state_dict(sd["opt"])
        self.ema.load_state_dict(sd["ema"], params=self.params)
        self.step = sd["step"]
        self.best_val = sd.get("best_val", float("inf"))
        torch.set_rng_state(sd["torch_rng"])
        if sd.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(sd["cuda_rng"])
        print(f"[resume] step {self.step} from {path}")

    def _out_of_time(self) -> bool:
        if self.cfg.max_hours is None:
            return False
        return (time.time() - self.t0) / 3600.0 >= self.cfg.max_hours

    # ---------------------------------------------------------------- loop --
    def run(self) -> Path:
        cfg = self.cfg
        self.adapter.train()
        it = iter(self.loader)

        win: dict = {}
        while self.step < cfg.steps:
            for g in self.opt.param_groups:
                g["lr"] = self._lr()

            self.opt.zero_grad(set_to_none=True)
            agg: dict = {}

            for _ in range(cfg.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    batch = next(it)
                loss, stats = self.loss_fn(self.adapter, batch, self.step, cfg)
                (loss / cfg.grad_accum).backward()
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0.0) + v / cfg.grad_accum

            max_norm = clip_threshold(cfg.grad_clip)
            # returns the PRE-clip total norm, so grad_norm stays comparable
            # across runs whether or not the clip binds
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, max_norm)
            self.opt.step()
            self.ema.update(self.params)
            self.step += 1

            # WINDOW AGGREGATES. The logged record is otherwise a snapshot of
            # ONE step in every log_every, so a single noisy batch reads as a
            # regime (a grad_norm spike at step 525 of the fd run was one
            # step in 25). The snapshot keys are unchanged; win_* are means
            # over every step since the last log, grad_norm_win_max the max.
            g = float(gnorm)
            win["n"] = win.get("n", 0) + 1
            win["g_sum"] = win.get("g_sum", 0.0) + g
            win["g_max"] = max(win.get("g_max", 0.0), g)
            win["clipped"] = win.get("clipped", 0) + int(g > max_norm)
            for k, v in agg.items():
                win.setdefault("sum", {})[k] = win.get("sum", {}).get(k, 0.0) + float(v)
                win.setdefault("cnt", {})[k] = win.get("cnt", {}).get(k, 0) + 1

            if self.step % cfg.log_every == 0:
                rec = {"step": self.step, "lr": self._lr(),
                       "grad_norm": float(gnorm),
                       "hours": (time.time() - self.t0) / 3600.0, **agg,
                       "grad_norm_win_mean": win["g_sum"] / win["n"],
                       "grad_norm_win_max": win["g_max"],
                       "grad_clip_frac": win["clipped"] / win["n"],
                       "win_steps": win["n"]}
                for k, v in win.get("sum", {}).items():
                    rec[f"win_{k}"] = v / win["cnt"][k]
                win.clear()
                with self._log_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()
                               if not k.startswith("win_")))

            if self.step % cfg.ckpt_every == 0:
                self.save("latest")
                # Numbered snapshots. `ckpt_latest.pt` is overwritten every
                # ckpt_every steps, so a finished run leaves exactly one
                # mid-training checkpoint: none. Gate A needed two points to
                # ask whether cos improvement bought anything, and step 4500
                # had already been overwritten by the time the question came
                # up. Keep the last `keep_last` snapshots; they are adapter
                # weights only, tens of MB each.
                if cfg.keep_last > 0 and self.step % (cfg.ckpt_every * 5) == 0:
                    self.save(f"step{self.step}")
                    snaps = sorted(
                        self.out.glob("ckpt_step*.pt"),
                        key=lambda q: int(q.stem.replace("ckpt_step", "")),
                    )
                    for old_ckpt in snaps[:-cfg.keep_last]:
                        old_ckpt.unlink()

            if self.val_fn is not None and cfg.val_every \
                    and self.step % cfg.val_every == 0:
                self.adapter.eval()
                with torch.no_grad():
                    v = float(self.val_fn(self.adapter, cfg))
                self.adapter.train()
                improved = v < self.best_val
                if improved:
                    self.best_val = v
                    self.save("valbest")
                extra = dict(getattr(self.val_fn, "last", {}) or {})
                with self._log_path.open("a") as fh:
                    fh.write(json.dumps({"step": self.step, "val": v,
                                         "best_val": self.best_val,
                                         "val_delta": cfg.val_delta,
                                         **extra}) + "\n")
                print(f"  val={v:.5f} best={self.best_val:.5f}"
                      + ("  <- saved ckpt_valbest (NOT the checkpoint to "
                         "evaluate: this val is not a validated metric)"
                         if improved else ""))

            if self._out_of_time():
                p = self.save("latest")
                print(f"[chain] wall limit reached at step {self.step}. "
                      f"Rerun the identical command to continue from {p}.")
                return p

        self.save("final")
        return self.out / "ckpt_final.pt"


def enable_memory_savers(transformer) -> None:
    """Gradient checkpointing plus attention slicing where available."""
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    else:
        transformer.gradient_checkpointing = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass