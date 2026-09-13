"""
Triage for the signal-11 result in smoke_test.py.

Four probes crashed identically, including one that is only nn.Linear with no
attention involved. That points at a single shared cause rather than four
separate incompatibilities. The candidates, in order of likelihood:

  A. HARNESS. The parent process initialised CUDA (the gpu/bf16/sdpa checks all
     ran CUDA ops) before spawning. A CUDA context inherited across fork is a
     classic source of exactly this crash. If so, torch is fine and only
     smoke_test.py is wrong.
  B. CUDA FORWARD-AD. torch.func.jvp is broken for CUDA tensors in this
     torch/driver combination. Real, and it would block Stage 2.
  C. TORCH INSTALL. Something deeper is wrong.

Run this INSTEAD of smoke_test.py, from a fresh shell:

    python docker/triage_jvp.py

It runs each probe as a clean top-level process with no prior CUDA use, and
walks from the simplest case up. The first failure localises the problem.

NOTE: none of this blocks Stage 0, which is inference only and never touches
forward-mode autodiff. Stage 0 can proceed while this is unresolved.
"""

from __future__ import annotations

import subprocess
import sys

PROBES = [
    (
        "1. cpu jvp, linear only",
        "no CUDA at all. A crash here means the torch install is broken (C).",
        """
import torch
from torch.func import jvp
net = torch.nn.Linear(64, 64)
x = torch.randn(8, 64)
out, tan = jvp(net, (x,), (torch.randn_like(x),))
assert tan.shape == out.shape
print("PASS")
""",
    ),
    (
        "2. cuda basic op, no jvp",
        "does CUDA work at all in a fresh child process?",
        """
import torch
x = torch.randn(256, 256, device="cuda")
(x @ x).sum().item()
print("PASS")
""",
    ),
    (
        "3. cuda jvp, linear only",
        "forward-mode AD on CUDA, no attention. A crash HERE is cause (B).",
        """
import torch
from torch.func import jvp
net = torch.nn.Linear(64, 64).cuda()
x = torch.randn(8, 64, device="cuda")
out, tan = jvp(net, (x,), (torch.randn_like(x),))
assert tan.shape == out.shape
print("PASS")
""",
    ),
    (
        "4. cuda jvp, math attention",
        "the actual MeanFlow requirement.",
        """
import torch, torch.nn.functional as F
from torch.func import jvp
from torch.nn.attention import SDPBackend, sdpa_kernel
x = torch.randn(1, 4, 128, 64, device="cuda")
with sdpa_kernel(SDPBackend.MATH):
    out, tan = jvp(lambda z: F.scaled_dot_product_attention(z, z, z),
                   (x,), (torch.randn_like(x),))
assert tan.shape == out.shape
print("PASS")
""",
    ),
    (
        "5. cuda jvp, manual attention",
        "softmax(QK^T)V written out. Works even if the SDPA path does not,\n"
        "      and is a usable fallback for the JVP branch of a MeanFlow loss.",
        """
import torch, math
from torch.func import jvp
def attn(x):
    s = (x @ x.transpose(-2, -1)) / math.sqrt(x.shape[-1])
    return s.softmax(-1) @ x
x = torch.randn(1, 4, 128, 64, device="cuda")
out, tan = jvp(attn, (x,), (torch.randn_like(x),))
assert tan.shape == out.shape
print("PASS")
""",
    ),
]


def run(src: str, timeout: int = 180):
    p = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout
    )
    if p.returncode == 0 and "PASS" in p.stdout:
        return True, "pass"
    if p.returncode < 0:
        return False, f"CRASH signal {-p.returncode}"
    tail = (p.stderr.strip().splitlines() or ["?"])[-1]
    return False, tail[:80]


def main():
    print("=" * 78)
    print("JVP triage — no CUDA is touched in this parent process")
    print("=" * 78)

    results = []
    for name, why, src in PROBES:
        ok, detail = run(src)
        results.append((name, ok))
        print(f"\n{name}")
        print(f"      {why}")
        print(f"      -> {'ok' if ok else 'FAIL'}: {detail}")

    print("\n" + "=" * 78)
    passed = dict(results)
    p1, p2, p3, p4, p5 = (passed[n] for n, _ in results)

    if not p1:
        verdict = (
            "CAUSE C: torch install is broken at the CPU level. Reinstall torch "
            "before anything else."
        )
    elif not p2:
        verdict = (
            "CUDA does not work in a child process at all. Driver or container "
            "runtime problem, not a torch/AD problem."
        )
    elif not p3:
        verdict = (
            "CAUSE B: forward-mode AD is broken for CUDA tensors in this build.\n"
            "  Stage 2 (MeanFlow/BiFM) is blocked until torch is changed.\n"
            "  Try: pip install --index-url https://download.pytorch.org/whl/cu124 \\\n"
            "         torch==2.5.1 torchvision==0.20.1\n"
            "  Stage 0 and 1 are unaffected — they never use jvp."
        )
    elif p3 and not p4 and p5:
        verdict = (
            "Localised: SDPA is the problem, plain CUDA forward-AD is fine.\n"
            "  Use the manual softmax(QK^T)V path inside the JVP branch and keep\n"
            "  fused SDPA everywhere else. Costs memory, unblocks Stage 2."
        )
    elif p4:
        verdict = (
            "CAUSE A: everything passes standalone. The original crash was the\n"
            "  smoke test spawning after the parent had initialised CUDA.\n"
            "  Fix smoke_test.py to run its probes before any CUDA use.\n"
            "  Nothing is wrong with your environment."
        )
    else:
        verdict = "Mixed result — paste this output and we will work through it."

    print("VERDICT\n  " + verdict)
    print("=" * 78)
    print(
        "\nRegardless of outcome: Stage 0 is inference only and uses no "
        "forward-mode AD.\nThe r-sweep can start today."
    )


if __name__ == "__main__":
    main()
