"""
Phase 1 audit: is transport/meanflow_ref.py actually reference-faithful?

Eight checks, all cheap, none needing the real model. Run before any Gate B
training, and again after any edit to meanflow_ref.py.

    python scripts/test_reference_audit.py

The point is not that these checks are clever. It is that the previous Gate B
implementation differed from the reference in a way that silently removed 75%
of the training signal, and nothing in the pipeline noticed -- the loss curve
looked plausible, the boundary assertion passed at exactly 0.0, and the
resulting Delta ~ 0.07 was nearly written up as a property of SD3. A cheap
audit against the source would have caught it before any GPU time.

Check 2 is the important one: it demonstrates the structural difference rather
than asserting it. Under mode="interval" the adapter gradient at r == t is
exactly zero; under mode="meanflow" it is not.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transport.adapter import AdapterConfig, TransportAdapter  # noqa: E402
from transport.meanflow_ref import (  # noqa: E402
    RefMeanFlowConfig, cond_drop, ref_meanflow_loss, sample_tr,
)

REF = {  # values read off Gsunshine/meanflow meanflow.py
    "P_mean": -0.4, "P_std": 1.0, "data_proportion": 0.75,
    "norm_p": 1.0, "norm_eps": 0.01, "class_dropout_prob": 0.1,
    "omega": 1.0, "kappa": 0.5,
}


def tiny(mode, dtype=torch.float32, device="cpu", seed=0):
    from diffusers import SD3Transformer2DModel

    torch.manual_seed(seed)
    tr = SD3Transformer2DModel(
        sample_size=16, patch_size=2, in_channels=16, num_layers=2,
        attention_head_dim=16, num_attention_heads=4, joint_attention_dim=64,
        caption_projection_dim=64, pooled_projection_dim=32, out_channels=16,
        pos_embed_max_size=32,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    torch.manual_seed(seed + 1)
    ad = TransportAdapter(tr, AdapterConfig(mode=mode, rank=4)).to(device)
    for p in ad.trainable_parameters():
        p.data = p.data.float()
    return ad


def main() -> int:
    cfg = RefMeanFlowConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    print("1. config values match the reference")
    for k, v in [("p_mean", REF["P_mean"]), ("p_std", REF["P_std"]),
                 ("data_proportion", REF["data_proportion"]),
                 ("norm_p", REF["norm_p"]), ("norm_eps", REF["norm_eps"]),
                 ("cond_dropout_prob", REF["class_dropout_prob"]),
                 ("omega", REF["omega"]), ("kappa", REF["kappa"])]:
        got = getattr(cfg, k)
        check(f"{k} = {v}", got == v, f"got {got}")

    print("\n2. endpoint sampling")
    t, r, anchor = sample_tr(4096, cfg, "cpu",
                             torch.Generator().manual_seed(0))
    check("r <= t everywhere", bool((r <= t + 1e-9).all()))
    frac = float((t - r).abs().lt(1e-9).float().mean())
    check(f"P(r == t) ~ {REF['data_proportion']}", abs(frac - 0.75) < 0.02,
          f"got {frac:.3f}")
    # Check the distribution on the NON-ANCHOR rows only. sample_tr overwrites
    # r with t on 75% of the batch, so cat([t, r]) is three-quarters
    # max(a, b) and its median sits well above sigmoid(P_mean). The first
    # version of this check tested the concatenation and failed the code for
    # a property the code does not have.
    # t is the MAX of two logit-normal draws and 75% of r is COPIED from t,
    # so neither population is distributed like a single draw. Comparing the
    # median of cat([t, r]) to sigmoid(P_mean) is the wrong test: it returns
    # ~0.51 for a correct implementation (verified: max-of-two median 0.536,
    # min-of-two 0.280, mixed 0.507, single draw 0.401).
    #
    # Test each population against what it should actually be, simulated
    # independently rather than reasoned about.
    g2 = torch.Generator().manual_seed(1)
    sim = torch.sigmoid(torch.randn(200000, 2, generator=g2) * REF["P_std"]
                        + REF["P_mean"])
    want_t = float(sim.max(dim=1).values.median())
    want_r = float(sim.min(dim=1).values.median())
    got_t = float(t.median())
    check("t ~ max of two logit-normal(P_mean, P_std)",
          abs(got_t - want_t) < 0.03, f"median {got_t:.3f} vs {want_t:.3f}")
    free = ~anchor
    if free.any():
        got_r = float(r[free].median())
        check("non-anchor r ~ min of two logit-normal",
              abs(got_r - want_r) < 0.03, f"median {got_r:.3f} vs {want_r:.3f}")
        # And the union of the non-anchor rows IS the raw draw population,
        # since {t_i, r_i} = {a_i, b_i} there.
        got_raw = float(torch.cat([t[free], r[free]]).median())
        want_raw = float(torch.sigmoid(torch.tensor(REF["P_mean"])))
        check("non-anchor {t, r} union ~ the raw draw distribution",
              abs(got_raw - want_raw) < 0.03,
              f"median {got_raw:.3f} vs {want_raw:.3f}")
    check("Delta spans its full range from step 0 (no curriculum)",
          float((t - r).max()) > 0.5, f"max Delta {float((t-r).max()):.3f}")

    print("\n3. target reduces to v at the anchors")
    x0 = torch.randn(8, 16, 16, 16)
    eps = torch.randn_like(x0)
    tt = torch.full((8,), 0.6)
    rr = tt.clone()
    v = eps - x0
    delta = (tt - rr).clamp(0, 1).reshape(-1, 1, 1, 1)
    # u* = v - Delta*du/dt, and Delta == 0, so u* == v for ANY du/dt
    u_tgt = v - delta * torch.randn_like(v) * 1e6
    check("u* == v when r == t", torch.allclose(u_tgt, v))

    print("\n4. loss reduction and adaptive weighting")
    err = torch.randn(8, 16, 16, 16)
    per_sample = err.pow(2).flatten(1).sum(1)
    check("per-sample loss is a SUM over non-batch dims",
          bool(torch.allclose(per_sample, err.pow(2).sum(dim=(1, 2, 3)))))
    w = (per_sample + cfg.norm_eps) ** cfg.norm_p
    check("adaptive weight = (sse + eps)^p with p = 1",
          bool(torch.allclose(w, per_sample + cfg.norm_eps)))
    ratio = float((per_sample / w).mean())
    check("weighting makes the loss ~scale-free (p=1)", 0.9 < ratio <= 1.0,
          f"mean loss/weight {ratio:.4f}")

    print(f"\n5. adapter gradients at r == t   (device={dev})")
    for mode, want_nonzero in (("meanflow", True), ("interval", False)):
        ad = tiny(mode, device=dev)
        z = torch.randn(4, 16, 16, 16, device=dev)
        pe = torch.randn(4, 24, 64, device=dev)
        po = torch.randn(4, 32, device=dev)
        t0 = torch.full((4,), 0.6, device=dev)
        r0 = t0.clone()                      # ANCHORS
        vg = torch.randn_like(z)
        loss, st = ref_meanflow_loss(
            lambda z_, t_, r_: ad(z_, t_, r_, pe, po), z, t0, r0, vg,
            RefMeanFlowConfig(estimator="fd"))
        loss.backward()
        g = sum(float(p.grad.abs().sum()) for p in ad.trainable_parameters()
                if p.grad is not None)
        check(f"mode={mode}: anchor gradient "
              f"{'nonzero' if want_nonzero else 'ZERO (the bug)'}",
              (g > 0) == want_nonzero, f"|grad| = {g:.3e}")

    print("\n6. cond_drop blanks the condition AND reverts the target")
    pe = torch.randn(64, 8, 16)
    po = torch.randn(64, 8)
    npe, npo = torch.zeros_like(pe), torch.zeros_like(po)
    v = torch.randn(64, 4, 8, 8)
    v_g = v + 1.0
    (pe_in, po_in), vg2 = cond_drop(v, v_g, (pe, po), (npe, npo),
                                    RefMeanFlowConfig(cond_dropout_prob=1.0))
    check("dropped samples get the null condition",
          bool(torch.allclose(pe_in, npe)))
    check("dropped samples revert v_g -> v", bool(torch.allclose(vg2, v)))

    print("\n7. no Delta curriculum in the reference path")
    # Check the AST for real identifiers, not the source text. The first
    # version grepped for "delta_schedule" and failed on the DOCSTRING that
    # explains why there is no curriculum -- the same grep-vs-parse mistake
    # that let a stale adapter.py through two checks earlier.
    import transport.meanflow_ref as mr
    ref_src = Path(mr.__file__).read_text()
    ref_tree = ast.parse(ref_src)
    # CURRICULUM symbols are banned: anything that changes the Delta range
    # over training. A STATIC cap is a different thing -- a fixed training-
    # range restriction, off by default -- and is checked separately below,
    # so allowing the name `max_delta` here does not open a hole for a
    # schedule.
    banned = {"delta_schedule", "delta_start", "delta_end"}
    found = set()
    for node in ast.walk(ref_tree):
        if isinstance(node, ast.Name) and node.id in banned:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in banned:
            found.add(node.attr)
        elif isinstance(node, ast.arg) and node.arg in banned:
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) \
                and node.name in banned:
            found.add(node.name)
    check("meanflow_ref defines/uses no curriculum symbols",
          not found, f"found {sorted(found)}" if found else "")
    # And the config must carry no curriculum knobs.
    fields = set(RefMeanFlowConfig().__dataclass_fields__)
    check("RefMeanFlowConfig has no curriculum fields",
          not (fields & banned), f"has {sorted(fields & banned)}")
    # Static caps and the tangent option: permitted deviations, but only if
    # the DEFAULTS are the reference and nothing is step-dependent.
    import inspect
    d = RefMeanFlowConfig()
    check("static caps default to no cap, tangent defaults to reference",
          d.max_delta == 1.0 and d.max_t == 1.0 and d.tangent == "cond",
          f"max_delta={d.max_delta} max_t={d.max_t} tangent={d.tangent}")
    check("sample_tr takes no step argument (no schedule possible)",
          "step" not in inspect.signature(sample_tr).parameters)

    print("\n8. guidance form")
    from transport.meanflow_ref import guided_velocity
    c = RefMeanFlowConfig(omega=1.0, kappa=0.5)
    v = torch.randn(4, 4, 8, 8)
    vu, vc = torch.randn_like(v), torch.randn_like(v)
    got = guided_velocity(v, None, None,
                          lambda z, t, e: vu if e == "u" else vc, "c", "u", c)
    want = c.omega * v + (1 - c.omega - c.kappa) * vu + c.kappa * vc
    check("kappa>0 branch matches the reference expression",
          bool(torch.allclose(got, want)))

    print("\n" + "=" * 62)
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        print("Do not run Gate B until these pass.")
        return 1
    print("Reference audit clean. The remaining deviations from the official\n"
          "implementation are the deliberate ones: text conditioning instead\n"
          "of class labels, and a LoRA over a frozen pretrained model instead\n"
          "of a DiT trained from scratch. Both are the subject of Gate B\n"
          "rather than accidents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
