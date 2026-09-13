"""
Environment smoke test. Run this first, inside the container.

    python docker/smoke_test.py

The interesting checks are 4 and 5. MeanFlow -- and therefore any BiFM
reimplementation -- needs a Jacobian-vector product through the transformer,
which requires FORWARD-mode autodiff. Two things in a normal diffusion training
setup silently do not support it:

  * the flash and memory-efficient SDPA backends. torch.func.jvp through them
    either errors or falls back, so the JVP path must run under the math
    backend, which is slower and uses more memory.
  * torch.utils.checkpoint. Gradient checkpointing and forward-mode AD do not
    compose. On 24 GB this matters, because checkpointing is exactly what you
    would otherwise use to fit the batch.

Discovering either of these three weeks into Stage 2 is expensive. Discovering
them now costs one minute.
"""

from __future__ import annotations

import sys

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "
failures: list[str] = []


def line(status, name, detail=""):
    print(f"[{status}] {name:<38} {detail}")


def check_torch():
    import torch

    line(OK, "torch", torch.__version__)
    if not torch.cuda.is_available():
        failures.append("no CUDA")
        line(BAD, "cuda", "not available")
        return None
    dev = torch.cuda.get_device_properties(0)
    gb = dev.total_memory / 1024**3
    line(OK, "gpu", f"{dev.name}  {gb:.1f} GB  sm_{dev.major}{dev.minor}")
    if gb < 20:
        failures.append(f"only {gb:.0f} GB VRAM; SD3 + T5-XXL will not co-reside")
    if dev.major < 8:
        failures.append("pre-Ampere: no bf16, no flash attention")
    return torch


def check_bf16(torch):
    if torch.cuda.is_bf16_supported():
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        line(OK, "bf16 matmul", "supported")
    else:
        failures.append("no bf16")
        line(BAD, "bf16", "unsupported — use fp16 and watch for overflow")


def check_sdpa(torch):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 4, 512, 64, device="cuda", dtype=torch.bfloat16)
    available = []
    for name, backend in [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
            available.append(name)
        except Exception:
            pass
    line(OK, "sdpa backends", ", ".join(available) or "none")
    if "math" not in available:
        failures.append("math SDPA backend unavailable — the JVP path needs it")


_JVP_PROBE = """
import torch, torch.nn.functional as F
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel
backend = {backend}
def block(x):
    return F.scaled_dot_product_attention(x, x, x)
x = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float32)
with sdpa_kernel(backend):
    out, tan = jvp(block, (x,), (torch.randn_like(x),))
assert tan.shape == out.shape
print("PASS")
"""

_CKPT_PROBE = """
import torch
from torch.func import jvp
from torch.utils.checkpoint import checkpoint
net = torch.nn.Linear(64, 64).cuda()
x = torch.randn(8, 64, device="cuda")
jvp(lambda z: checkpoint(net, z, use_reentrant=False), (x,), (torch.randn_like(x),))
print("PASS")
"""


def _run_probe(src: str, timeout: int = 120):
    """Run a probe in a subprocess.

    Forward-mode AD through the fused attention kernels does not merely raise —
    it can segfault in native code, which no try/except in this process can
    catch. Isolating each probe turns a crashed test run into a reported
    result.

    Returns (status, detail) where status is 'pass', 'error' or 'crash'.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode == 0 and "PASS" in proc.stdout:
        return "pass", "works"
    if proc.returncode < 0:
        return "crash", f"native crash (signal {-proc.returncode})"
    tail = (proc.stderr.strip().splitlines() or ["unknown"])[-1]
    return "error", tail[:70]


def check_jvp_through_attention(torch):
    """The MeanFlow blocker. The math backend MUST pass."""
    # Math first: it is the one that decides whether Stage 2 is possible.
    for name, backend in [
        ("math", "SDPBackend.MATH"),
        ("mem_efficient", "SDPBackend.EFFICIENT_ATTENTION"),
        ("flash", "SDPBackend.FLASH_ATTENTION"),
    ]:
        status, detail = _run_probe(_JVP_PROBE.format(backend=backend))
        if status == "pass":
            line(OK, f"jvp via {name}", detail)
        elif name == "math":
            line(BAD, "jvp via math", detail)
            failures.append("JVP fails under the math backend — MeanFlow is blocked")
        else:
            line(WARN, f"jvp via {name}", f"{detail} — expected; force math for JVP")


def check_jvp_vs_checkpointing(torch):
    """Expected to fail. The point is to see it now, not in month two."""
    status, detail = _run_probe(_CKPT_PROBE)
    if status == "pass":
        line(OK, "jvp + grad checkpointing", "composes (unexpected — good)")
    else:
        line(
            WARN,
            "jvp + grad checkpointing",
            f"{detail} — budget VRAM without checkpointing",
        )


def check_memory_budget(torch):
    """Can T5-XXL and a 2B DiT co-reside? Allocate proxies and see."""
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    line(OK, "free vram", f"{free_gb:.1f} GB")
    budget = {
        "T5-XXL (bf16)": 9.4,
        "SD3-Medium DiT (bf16)": 4.3,
        "CLIP-L + CLIP-G": 1.6,
        "VAE": 0.3,
    }
    total = sum(budget.values())
    for k, v in budget.items():
        print(f"        {k:<26} {v:>5.1f} GB")
    print(f"        {'weights subtotal':<26} {total:>5.1f} GB")
    print(f"        {'left for activations':<26} {free_gb - total:>5.1f} GB")
    if free_gb - total < 4:
        failures.append(
            "under 4 GB for activations — cache text embeddings offline and "
            "unload T5 before the DiT loads"
        )


def check_deps():
    import importlib

    for mod, why in [
        ("diffusers", "SD3 pipeline"),
        ("transformers", "T5 / CLIP"),
        ("peft", "LoRA for stages 2-3"),
        ("sentencepiece", "T5 tokenizer — SD3 silently fails without it"),
        ("scipy", "distance transforms"),
        ("lpips", "background preservation metric"),
        ("open_clip", "CLIP scores"),
    ]:
        try:
            importlib.import_module(mod)
            line(OK, mod, why)
        except ImportError:
            failures.append(f"missing {mod}")
            line(BAD, mod, f"MISSING — needed for {why}")


def check_project():
    try:
        import numpy as np

        from bench.data import decode_rle, parse_brackets
        from region.alpha_field import build_retention_field, pin_weights
        from region.lambda_field import build_region_field

        m = np.zeros((256, 256), bool)
        m[100:150, 100:160] = True
        f = build_region_field(m)
        tau = build_retention_field(m)
        assert f.shape == (16, 16) and tau.shape == (16, 16)
        # target must be FREE early, background PINNED early
        assert pin_weights(tau, 0.9)[tau.argmax() // 16, tau.argmax() % 16] < 0.5
        assert decode_rle("0 64", (8, 8)).all()
        assert parse_brackets("a [red] car")[1] == ["red"]
        line(OK, "regionflow modules", "fields, rle and bracket parsing agree")
    except Exception as exc:
        failures.append(f"project import/logic: {exc}")
        line(BAD, "regionflow modules", str(exc)[:70])


def main():
    print("=" * 78)
    torch = check_torch()
    if torch is None:
        sys.exit(1)
    check_bf16(torch)
    check_sdpa(torch)
    print("-" * 78)
    check_jvp_through_attention(torch)
    check_jvp_vs_checkpointing(torch)
    print("-" * 78)
    check_memory_budget(torch)
    print("-" * 78)
    check_deps()
    check_project()
    print("=" * 78)

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nenvironment ok")


if __name__ == "__main__":
    main()
