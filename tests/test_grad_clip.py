"""CPU check for --grad-clip. Seconds.

    python tests/test_grad_clip.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.train import TrainConfig, clip_threshold  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


check("default clip is unchanged (1.0)", TrainConfig(gate="b").grad_clip == 1.0)
check("positive threshold passes through", clip_threshold(1.0) == 1.0)
check("0 disables", clip_threshold(0.0) == math.inf)
check("negative disables", clip_threshold(-1.0) == math.inf)


def grads(scale):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.zeros(5, 7)), torch.nn.Parameter(torch.zeros(3))]
    for p in ps:
        p.grad = torch.randn_like(p) * scale
    return ps


# disabled: gradients bit-for-bit untouched, and the returned norm is the true one
ps = grads(5e3)
before = [p.grad.clone() for p in ps]
true_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in ps))
n = torch.nn.utils.clip_grad_norm_(ps, clip_threshold(0.0))
check("disabled leaves gradients exactly unchanged",
      all(torch.equal(b, p.grad) for b, p in zip(before, ps)))
check("returned norm is the pre-clip norm",
      torch.allclose(n, true_norm, rtol=1e-6), f"{float(n):.4g}")

# enabled at 1.0 with large gradients: clips to unit norm, still reports pre-clip
ps = grads(5e3)
n = torch.nn.utils.clip_grad_norm_(ps, clip_threshold(1.0))
after = torch.sqrt(sum(p.grad.pow(2).sum() for p in ps))
check("enabled clip binds on large gradients",
      float(n) > 1.0 and abs(float(after) - 1.0) < 1e-4,
      f"pre {float(n):.4g} post {float(after):.4g}")

# the ratio that motivated this: norm_p=0 gradients vs norm_p=1 at sse ~2e4
ps = grads(0.05 / math.sqrt(38))          # total norm ~0.05, like the main run
n = torch.nn.utils.clip_grad_norm_(ps, clip_threshold(1.0))
check("main-run-sized gradients never reach the 1.0 clip", float(n) < 1.0,
      f"{float(n):.3g}")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
