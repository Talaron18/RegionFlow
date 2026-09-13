"""CPU checks for the MeanFlow target options. No model, seconds.

    python tests/test_meanflow_target.py

A toy u_fn with a closed-form total derivative stands in for the adapter.
The first two checks pin the REFERENCE path (tangent='cond', no caps) to a
verbatim re-implementation of the pre-change code, so the new options cannot
have altered the baseline run.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.meanflow_ref import (  # noqa: E402
    RefMeanFlowConfig, central_difference, ref_meanflow_loss, sample_tr,
    tangent_velocity,
)

torch.manual_seed(0)
DT = torch.float64
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


theta = torch.tensor(1.3, dtype=DT, requires_grad=True)
SH = (-1, 1, 1, 1)
EVAL_T = []


def u_fn(z, t, r):
    EVAL_T.append(t.detach().clone())
    h = (t - r).clamp(min=0.0)
    a = torch.cos(3 * t) + 0.5 * h
    b = torch.sin(2 * t)
    return theta * (a.reshape(SH) * z + b.reshape(SH))


def du_dt_exact(z, t, r, w):
    """Total derivative along (w, dt=1, dr=0), valid for t - r > 0."""
    a = torch.cos(3 * t) + 0.5 * (t - r)
    return theta.detach() * (a.reshape(SH) * w
                             + (-3 * torch.sin(3 * t) + 0.5).reshape(SH) * z
                             + (2 * torch.cos(2 * t)).reshape(SH))


def base_fn(z, t):
    return 0.7 * z + torch.sin(t).reshape(SH)


def batch(B=6, eps_margin=0.1):
    t = torch.rand(B, dtype=DT) * (1 - 2 * eps_margin) + eps_margin
    r = t - torch.rand(B, dtype=DT) * (t - 0.08) - 0.07   # Delta >= 0.07
    z = torch.randn(B, 2, 3, 3, dtype=DT)
    v_g = torch.randn_like(z)
    return z, t, r, v_g


def weighted(per_sample, cfg):
    return (per_sample / ((per_sample + cfg.norm_eps) ** cfg.norm_p).detach()).mean()


def original_loss(z, t, r, v_g, cfg):
    """Pre-change ref_meanflow_loss, verbatim in substance."""
    dt, dr = torch.ones_like(t), torch.zeros_like(r)
    if cfg.estimator == "jvp":
        with torch.no_grad():
            _, du = torch.func.jvp(u_fn, (z, t, r), (v_g, dt, dr))
        du = du.detach()
    else:
        e = cfg.fd_eps
        with torch.no_grad():
            du = ((u_fn(z + e * v_g, t + e * dt, r)
                   - u_fn(z - e * v_g, t - e * dt, r)) / (2 * e)).detach()
    u = u_fn(z, t, r)
    delta = (t - r).clamp(0.0, 1.0).reshape(SH)
    tgt = (v_g - delta * du).detach()
    return weighted((u - tgt).pow(2).flatten(1).sum(1), cfg)


def grad_of(loss):
    theta.grad = None
    loss.backward()
    return theta.grad.clone()


# 1-2. reference path unchanged -------------------------------------------------
z, t, r, v_g = batch()
for est in ("jvp", "fd"):
    cfg = RefMeanFlowConfig(estimator=est)
    l_old = original_loss(z, t, r, v_g, cfg)
    g_old = grad_of(l_old)
    l_new, st = ref_meanflow_loss(u_fn, z, t, r, v_g, cfg)
    g_new = grad_of(l_new)
    check(f"reference path unchanged ({est})",
          torch.allclose(l_old, l_new, rtol=1e-12, atol=0)
          and torch.allclose(g_old, g_new, rtol=1e-10, atol=0)
          and st["tangent_rel_vg"] == 0.0,
          f"loss {float(l_old):.6g} vs {float(l_new):.6g}")

# 3. central difference against the closed form ---------------------------------
w = base_fn(z, t)
for eps in (0.01, 0.025):
    d_fd = central_difference(u_fn, z, t, r, w, eps)
    d_ex = du_dt_exact(z, t, r, w)
    rel = float((d_fd - d_ex).norm() / d_ex.norm())
    check(f"central difference matches closed form (eps={eps})",
          rel < 3 * (3 * eps) ** 2, f"rel {rel:.2e}")

# 4. the loss wires the chosen tangent into the stencil and the JVP ---------------
for est, tan in (("fd", "frozen"), ("fd", "boundary"), ("jvp", "frozen"),
                 ("jvp", "boundary")):
    cfg = RefMeanFlowConfig(estimator=est, fd_eps=0.025, tangent=tan)
    with torch.no_grad():
        w_man = base_fn(z, t) if tan == "frozen" else u_fn(z, t, t)
    if est == "fd":
        du = central_difference(u_fn, z, t, r, w_man, cfg.fd_eps)
    else:
        with torch.no_grad():
            _, du = torch.func.jvp(u_fn, (z, t, r), (w_man, torch.ones_like(t),
                                                      torch.zeros_like(r)))
    delta = (t - r).clamp(0.0, 1.0).reshape(SH)
    tgt = (v_g - delta * du).detach()          # first term stays v_g
    l_man = weighted((u_fn(z, t, r) - tgt).pow(2).flatten(1).sum(1), cfg)
    g_man = grad_of(l_man)
    l_new, st = ref_meanflow_loss(u_fn, z, t, r, v_g, cfg, base_fn=base_fn)
    g_new = grad_of(l_new)
    rel_w = float(((w_man - v_g).flatten(1).norm(dim=1)
                   / v_g.flatten(1).norm(dim=1)).mean())
    check(f"{est}/{tan}: target, gradient and tangent stat",
          torch.allclose(l_man, l_new, rtol=1e-12, atol=0)
          and torch.allclose(g_man, g_new, rtol=1e-10, atol=0)
          and abs(st["tangent_rel_vg"] - rel_w) < 1e-12,
          f"tangent_rel_vg {st['tangent_rel_vg']:.3f}")

# 5. tangent carries no gradient ---------------------------------------------------
w_b = tangent_velocity(u_fn, None, z, t, v_g,
                       RefMeanFlowConfig(tangent="boundary"))
check("boundary tangent is detached", not w_b.requires_grad)

# 6. stencil near the ends of [0, 1] -----------------------------------------------
t_edge = torch.tensor([0.995, 0.003], dtype=DT)
r_edge = torch.tensor([0.5, 0.0], dtype=DT)
z_e = torch.randn(2, 2, 3, 3, dtype=DT)
w_e = base_fn(z_e, t_edge)
EVAL_T.clear()
central_difference(u_fn, z_e, t_edge, r_edge, w_e, 0.05)
t_seen = torch.cat(EVAL_T)
check("stencil never evaluates outside [0, 1]",
      float(t_seen.min()) >= 0.0 and float(t_seen.max()) <= 1.0,
      f"t in [{float(t_seen.min()):.3f}, {float(t_seen.max()):.3f}]")


def lin_fn(z, t, r):
    # linear along the path, so even the asymmetric edge stencil is exact --
    # which isolates the divisor: dividing by 2*eps instead of the actual span
    # would be off by (ep + em) / 2eps
    return theta * (0.3 * z + (t + 0.2 * (t - r)).reshape(SH))


d_lin = central_difference(lin_fn, z_e, t_edge, r_edge, w_e, 0.05)
d_lin_ex = theta.detach() * (0.3 * w_e + 1.2)
rel_lin = float((d_lin - d_lin_ex).norm() / d_lin_ex.norm())
check("edge stencil divides by the actual span", rel_lin < 1e-10,
      f"rel {rel_lin:.1e}")

# 7. sampler: caps, and unchanged at defaults -----------------------------------------
N = 20000
g = torch.Generator().manual_seed(7)
cfg = RefMeanFlowConfig(max_delta=0.3, max_t=0.9, data_proportion=0.25)
tc, rc, anc = sample_tr(N, cfg, "cpu", generator=g)
d = tc - rc
check("max_t and max_delta respected",
      float(tc.max()) <= 0.9 and float(d.max()) <= 0.3 + 1e-7
      and bool((rc[anc] == tc[anc]).all()),
      f"t max {float(tc.max()):.3f}  delta max {float(d.max()):.4f}")
check("anchor fraction unaffected by caps",
      abs(float(anc.float().mean()) - 0.25) < 0.02,
      f"{float(anc.float().mean()):.3f}")
z7, t7, r7, v7 = batch(B=64)
r7 = torch.maximum(r7, t7 - 0.1)           # cap 0.1 by hand
_, st7 = ref_meanflow_loss(u_fn, z7, t7, r7, v7,
                           RefMeanFlowConfig(estimator="fd", max_delta=0.1))
exp7 = float(((t7 - r7) >= 0.1 - 1e-6).float().mean())
check("frac_at_delta_cap stat", abs(st7["frac_at_delta_cap"] - exp7) < 1e-12
      and st7["frac_at_delta_cap"] > 0, f"{st7['frac_at_delta_cap']:.3f}")
print(f"        mass at delta cap: "
      f"{float(((d > 0.3 - 1e-6) & ~anc).float().mean()):.3f} of all samples")

g1 = torch.Generator().manual_seed(11)
t_new, r_new, a_new = sample_tr(64, RefMeanFlowConfig(), "cpu", generator=g1)
g2 = torch.Generator().manual_seed(11)
n1 = torch.sigmoid(torch.randn(64, generator=g2) * 1.0 - 0.4)
n2 = torch.sigmoid(torch.randn(64, generator=g2) * 1.0 - 0.4)
t_old, r_old = torch.maximum(n1, n2), torch.minimum(n1, n2)
a_old = torch.rand(64, generator=g2) < 0.75
r_old = torch.where(a_old, t_old, r_old)
check("sampler unchanged at defaults",
      torch.equal(t_new, t_old) and torch.equal(r_new, r_old))

# 8. config validation ------------------------------------------------------------------
for kwargs in ({"tangent": "marginal"}, {"max_delta": 0.0}, {"max_t": 1.5},
               {"fd_eps": 0.0}):
    try:
        RefMeanFlowConfig(**kwargs)
        check(f"rejects {kwargs}", False)
    except ValueError:
        check(f"rejects {kwargs}", True)
try:
    ref_meanflow_loss(u_fn, z, t, r, v_g, RefMeanFlowConfig(tangent="frozen"))
    check("frozen tangent without base_fn raises", False)
except ValueError:
    check("frozen tangent without base_fn raises", True)

# 9. first_term='frozen' ------------------------------------------------------------
for est, tan in (("fd", "frozen"), ("fd", "cond"), ("jvp", "boundary")):
    # kappa=0 explicitly: the config's reference default is kappa=0.5, and a
    # frozen first term refuses guidance
    cfg = RefMeanFlowConfig(estimator=est, fd_eps=0.025, tangent=tan,
                            first_term="frozen", norm_p=0.0, kappa=0.0)
    with torch.no_grad():
        w_man = {"frozen": base_fn(z, t), "cond": v_g,
                 "boundary": u_fn(z, t, t)}[tan]
        first_man = base_fn(z, t)
    if est == "fd":
        du = central_difference(u_fn, z, t, r, w_man, cfg.fd_eps)
    else:
        with torch.no_grad():
            _, du = torch.func.jvp(u_fn, (z, t, r), (w_man, torch.ones_like(t),
                                                      torch.zeros_like(r)))
    delta = (t - r).clamp(0.0, 1.0).reshape(SH)
    tgt = (first_man - delta * du).detach()
    l_man = weighted((u_fn(z, t, r) - tgt).pow(2).flatten(1).sum(1), cfg)
    g_man = grad_of(l_man)
    l_new, st = ref_meanflow_loss(u_fn, z, t, r, v_g, cfg, base_fn=base_fn)
    g_new = grad_of(l_new)
    rel_f = float(((first_man - v_g).flatten(1).norm(dim=1)
                   / v_g.flatten(1).norm(dim=1)).mean())
    check(f"first_term=frozen ({est}/{tan}): target, gradient, stat",
          torch.allclose(l_man, l_new, rtol=1e-12, atol=0)
          and torch.allclose(g_man, g_new, rtol=1e-10, atol=0)
          and abs(st["first_rel_vg"] - rel_f) < 1e-12)

# anchors distil exactly: with u_fn == base_fn at r == t the frozen target is
# matched and sse is 0 -- the property that makes norm_p=1 dangerous here
def u_eq_base(z_, t_, r_):
    return base_fn(z_, t_) + 0.0 * theta
zt_a, t_a, v_a = z[:3], t[:3], v_g[:3]
_, st_a = ref_meanflow_loss(u_eq_base, zt_a, t_a, t_a.clone(), v_a,
                            RefMeanFlowConfig(estimator="fd", fd_eps=0.025,
                                              tangent="frozen",
                                              first_term="frozen",
                                              kappa=0.0),
                            base_fn=base_fn)
check("frozen first term: anchors of the teacher have sse 0",
      st_a["sse"] < 1e-20, f"sse {st_a['sse']:.1e}")
check("cond first term: first_rel_vg is exactly 0",
      ref_meanflow_loss(u_fn, z, t, r, v_g, RefMeanFlowConfig())[1]
      ["first_rel_vg"] == 0.0)

# 10. adaptive-weight concentration stat ---------------------------------------------
_, st_p0 = ref_meanflow_loss(u_fn, z, t, r, v_g, RefMeanFlowConfig(norm_p=0.0))
check("adp_ess is 1.0 with norm_p=0", abs(st_p0["adp_ess"] - 1.0) < 1e-12,
      f"{st_p0['adp_ess']:.6f}")
_, st_p1 = ref_meanflow_loss(u_fn, z, t, r, v_g, RefMeanFlowConfig(norm_p=1.0))
check("adp_ess in (1/N, 1] with norm_p=1",
      1.0 / z.shape[0] - 1e-12 <= st_p1["adp_ess"] <= 1.0 + 1e-12,
      f"{st_p1['adp_ess']:.3f}")
check("t_at_sse_min is one of the batch's t",
      bool((t - st_p1["t_at_sse_min"]).abs().min() < 1e-12))

RefMeanFlowConfig(first_term="frozen", omega=1.0, kappa=0.0)   # must not raise
check("accepts first_term=frozen with omega=1, kappa=0", True)
for kwargs in ({"first_term": "teacher"},
               {"first_term": "frozen", "omega": 2.0, "kappa": 0.0},
               {"first_term": "frozen", "omega": 1.0, "kappa": 0.5}):
    try:
        RefMeanFlowConfig(**kwargs)
        check(f"rejects {kwargs}", False)
    except ValueError:
        check(f"rejects {kwargs}", True)

# 11. eval correction metrics ------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_interval_field import correction_metrics  # noqa: E402
ct = torch.randn(4, 2, 3, 3, dtype=DT)
m0 = correction_metrics(torch.zeros_like(ct), ct)
m1 = correction_metrics(ct.clone(), ct)
m2 = correction_metrics(2 * ct, ct)
ortho = torch.randn_like(ct)
ortho = ortho - (ortho.flatten(1) * ct.flatten(1)).sum(1).reshape(SH) \
    / ct.flatten(1).pow(2).sum(1).reshape(SH) * ct
m3 = correction_metrics(0.3 * ct + ortho, ct)
cos3 = torch.nn.functional.cosine_similarity(
    (0.3 * ct + ortho).flatten(1), ct.flatten(1), dim=1)
check("corr_err: zero correction scores 1, exact scores 0, 2x scores 1",
      abs(m0["corr_err"] - 1) < 1e-12 and m1["corr_err"] < 1e-24
      and abs(m2["corr_err"] - 1) < 1e-12 and abs(m1["corr_err_best"]) < 1e-12)
check("corr_err_best = 1 - cos^2 and is a lower bound",
      abs(m3["corr_err_best"] - float((1 - cos3 ** 2).mean())) < 1e-12
      and m3["corr_err"] >= m3["corr_err_best"] - 1e-12,
      f"err {m3['corr_err']:.3f} best {m3['corr_err_best']:.3f}")

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
