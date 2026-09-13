"""CPU checks for CenteredIntervalEmbedding. No model, no GPU, seconds.

    python tests/test_centered_interval.py

Each check tests the property the run depends on, on the code path that
executes (forward and delta_part as TransportAdapter calls them), not a
neighbouring function. Cancellation tolerances are 1e-5 relative, not 0:
BLAS may route the two halves of the batched call through different
microkernels. A real failure is O(1).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.conditioning import (  # noqa: E402
    CenteredIntervalEmbedding, IntervalEmbedding, interval_kind_from_keys,
    make_interval_embedding, sinusoidal,
)

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


def randomise(module, scale=0.05):
    """Move every weight off init, so no check passes because of zeros."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * scale)


B = 16
t = torch.rand(B) * 0.8 + 0.2
delta = torch.rand(B) * t
r = t - delta

# 1. identity at init
emb = CenteredIntervalEmbedding()
out0 = emb(t, r)
check("zero output at init", out0.abs().max().item() == 0.0,
      f"max {out0.abs().max().item():.2e}")

# 2. with random weights: anchors cancel, intervals do not
randomise(emb)
with torch.no_grad():
    anchor_part = emb.delta_part(t, t.clone())
    interval_part = emb.delta_part(t, r)
    g = emb.t_branch(sinusoidal(t, emb.freq_dim))
    full_anchor = emb(t, t.clone())
rel_anchor = (anchor_part.norm() / interval_part.norm()).item()
check("delta part vanishes at r == t", rel_anchor < 1e-5,
      f"|anchor| / |interval| = {rel_anchor:.2e}")
check("delta part is live at r < t", interval_part.norm().item() > 1e-3,
      f"|interval| = {interval_part.norm().item():.3e}")
check("forward at r == t equals g(t)",
      torch.allclose(full_anchor, g, atol=1e-6),
      f"max {(full_anchor - g).abs().max().item():.2e}")

# 3. gradient: anchors train g but not f's readout
emb.zero_grad()
(emb(t, t.clone()) * torch.randn(B, 2048)).sum().backward()
gf = emb.delta_branch[-1].weight.grad.norm().item()
gg = emb.t_branch[-1].weight.grad.norm().item()
check("anchors give f-readout no gradient, g full gradient",
      gg > 0 and gf / gg < 1e-5, f"|grad f| {gf:.2e}  |grad g| {gg:.2e}")

# 4. gradient: from zero init, f's readout learns off-anchor
fresh = CenteredIntervalEmbedding()
(fresh(t, r) * torch.randn(B, 2048)).sum().backward()
gf0 = fresh.delta_branch[-1].weight.grad.norm().item()
check("f readout gets gradient at zero init when r < t", gf0 > 0,
      f"|grad| {gf0:.2e}")

# 5. forward-mode AD, as the jvp estimator uses it (tangent dt=1, dr=0)
fn = lambda tt, rr: emb(tt, rr)  # noqa: E731
_, tangent = torch.func.jvp(fn, (t, r), (torch.ones(B), torch.zeros(B)))
tangent = tangent.detach()
with torch.no_grad():
    h = 1e-4
    fd = (emb(t + h, r) - emb(t - h, r)) / (2 * h)
cos = torch.nn.functional.cosine_similarity(tangent.flatten(), fd.flatten(),
                                            dim=0).item()
rel = ((tangent - fd).norm() / fd.norm()).item()
check("jvp matches central difference", torch.isfinite(tangent).all().item()
      and cos > 0.99 and rel < 5e-2, f"cos {cos:.4f}  rel {rel:.2e}")

# 6. checkpoint-kind detection, from real state_dict names
ck_concat = {f"interval.{k}": v for k, v in IntervalEmbedding().state_dict().items()}
ck_cent = {f"interval.{k}": v for k, v in emb.state_dict().items()}
lora = {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": None}
check("detects concat", interval_kind_from_keys({**ck_concat, **lora}) == "concat")
check("detects centered", interval_kind_from_keys({**ck_cent, **lora}) == "centered")
check("no interval keys -> None", interval_kind_from_keys(lora) is None)
try:
    interval_kind_from_keys({**ck_concat, **ck_cent})
    check("mixture raises", False)
except ValueError:
    check("mixture raises", True)

# 7. factory validates
try:
    make_interval_embedding("centred", 2048)
    check("unknown kind raises", False)
except ValueError:
    check("unknown kind raises", True)
check("factory builds centered",
      isinstance(make_interval_embedding("centered", 2048),
                 CenteredIntervalEmbedding))

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
