"""Where does the model's prediction u live?

If u is diffuse (spread outside the mask), the mask multiplication
z_src + u*m creates a boundary jump that shows up as HF power. If u
is concentrated inside the mask, the HF excess has a different cause.
"""
import numpy as np


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="runs/synthetic_u_v.npz")
    args = ap.parse_args()
    d = np.load(args.npz)
    u = d["u"]          # (C, h, w)
    v = d["v"]          # (C, h, w)
    mask = d["mask"]    # (h, w) bool

    if mask.ndim == 3:
        mask = mask[0]
    if mask.dtype != bool:
        mask = mask > 0.5

    m_in = mask
    m_out = ~mask
    n_in = m_in.sum()
    n_out = m_out.sum()
    print(f"mask: {n_in} pixels in, {n_out} out, "
          f"fraction {n_in / (n_in + n_out):.3f}")

    C = u.shape[0]

    # RMS per element inside / outside
    u_in = u[:, m_in]
    u_out = u[:, m_out]
    v_in = v[:, m_in]
    v_out = v[:, m_out]

    u_in_rms = np.sqrt((u_in ** 2).mean())
    u_out_rms = np.sqrt((u_out ** 2).mean())
    v_in_rms = np.sqrt((v_in ** 2).mean())
    v_out_rms = np.sqrt((v_out ** 2).mean())

    print(f"\nRMS per element:")
    print(f"  u inside  mask: {u_in_rms:.5f}")
    print(f"  u outside mask: {u_out_rms:.5f}")
    print(f"  v inside  mask: {v_in_rms:.5f}")
    print(f"  v outside mask: {v_out_rms:.5f}")

    # the key ratio
    ratio_u = u_out_rms / u_in_rms if u_in_rms > 1e-8 else np.nan
    ratio_v = v_out_rms / v_in_rms if v_in_rms > 1e-8 else np.nan

    print(f"\noutside/inside RMS ratio:")
    print(f"  u: {ratio_u:.4f}")
    print(f"  v: {ratio_v:.4f}  (expected ~0 for synthetic)")

    # energy distribution
    u_in_energy = (u_in ** 2).sum()
    u_out_energy = (u_out ** 2).sum()
    u_total = u_in_energy + u_out_energy
    print(f"\nEnergy (sum of squares):")
    print(f"  u inside:  {u_in_energy:.2f}  "
          f"({100 * u_in_energy / u_total:.2f}%)")
    print(f"  u outside: {u_out_energy:.2f}  "
          f"({100 * u_out_energy / u_total:.2f}%)")

    # per-channel breakdown of outside energy
    print(f"\nPer-channel outside/inside RMS ratio:")
    for c in range(C):
        r = (np.sqrt((u[c, m_out] ** 2).mean())
             / np.sqrt((u[c, m_in] ** 2).mean() + 1e-12))
        print(f"  ch {c:2d}: {r:.4f}")

    # Also check: what fraction of mask is "the changed region"?
    # v should be near zero in most of the mask if the changed circle
    # is small relative to the mask
    v_abs = np.abs(v).mean(axis=0)  # (h, w)
    print(f"\nv mean |.| inside mask: {v_abs[m_in].mean():.5f}")
    print(f"v mean |.| outside mask: {v_abs[m_out].mean():.5f}")

    # interpretation
    print(f"\n{'=' * 60}")
    print(f"INTERPRETATION")
    print(f"{'=' * 60}")
    if ratio_u > 0.3:
        print(f"  outside/inside = {ratio_u:.3f} > 0.3")
        print(f"  -> u is DIFFUSE. The model's prediction extends well")
        print(f"     outside the mask, so z_src + u*m has a boundary jump.")
        print(f"     The HF excess in the spectrum comes from this jump.")
        print(f"     Spatial localisation is the real bottleneck.")
    elif ratio_u > 0.1:
        print(f"  outside/inside = {ratio_u:.3f} (moderate)")
        print(f"  -> partial diffusion. Boundary artifact contributes")
        print(f"     but is not the whole story.")
    else:
        print(f"  outside/inside = {ratio_u:.3f} < 0.1")
        print(f"  -> u is LOCALISED inside the mask. The HF excess is")
        print(f"     NOT a boundary artifact. Look at the within-mask")
        print(f"     spatial pattern instead.")
# 加到 synthetic_inside_outside.py 末尾

    print(f"\n{'=' * 60}")
    print(f"SPATIAL STRUCTURE OF u INSIDE MASK")
    print(f"{'=' * 60}")


    u_in = u[:, m_in]   # (C, N_in)
    v_in = v[:, m_in]

    u_mean_c = u_in.mean(axis=1)     # (C,) per-channel mean
    u_std_c  = u_in.std(axis=1)      # (C,) per-channel std
    v_mean_c = v_in.mean(axis=1)
    v_std_c  = v_in.std(axis=1)

    print(f"{'ch':>3}  {'u_mean':>10}  {'u_std':>10}  {'v_mean':>10}  "
      f"{'v_std':>10}  {'u_std/v_std':>12}")
    for c in range(C):
        ratio = u_std_c[c] / (v_std_c[c] + 1e-8)
        print(f"{c:>3}  {u_mean_c[c]:>10.4f}  {u_std_c[c]:>10.4f}  "
              f"{v_mean_c[c]:>10.4f}  {v_std_c[c]:>10.4f}  {ratio:>12.4f}")

# aggregate
    u_total_energy = (u_in ** 2).sum()
    u_mean_energy = (u_in.mean(axis=1, keepdims=True) ** 2).sum() * u_in.shape[1]
    u_fluc_energy = u_total_energy - u_mean_energy
    print(f"\nDecomposition of u energy inside mask:")
    print(f"  total:      {u_total_energy:>12.2f}")
    print(f"  from mean:  {u_mean_energy:>12.2f}  ({100*u_mean_energy/u_total_energy:.1f}%)")
    print(f"  from fluc:  {u_fluc_energy:>12.2f}  ({100*u_fluc_energy/u_total_energy:.1f}%)")

    v_total_energy = (v_in ** 2).sum()
    v_mean_energy = (v_in.mean(axis=1, keepdims=True) ** 2).sum() * v_in.shape[1]
    v_fluc_energy = v_total_energy - v_mean_energy
    print(f"\nSame for v (target):")
    print(f"  total:      {v_total_energy:>12.2f}")
    print(f"  from mean:  {v_mean_energy:>12.2f}  ({100*v_mean_energy/v_total_energy:.1f}%)")
    print(f"  from fluc:  {v_fluc_energy:>12.2f}  ({100*v_fluc_energy/v_total_energy:.1f}%)")
    print(f"\n{'=' * 60}")
    print(f"FOR REAL CACHE, compare to the synthetic run:")
    print(f"{'=' * 60}")
    print(f"  Expected fluc if u were pure 0.54*v:")
    print(f"    u_fluc_expected = (|u_mean| / |v_mean|)^2 * v_fluc")
    ratio_mag = (np.sqrt(u_mean_energy) / (np.sqrt(v_mean_energy) + 1e-8))
    u_fluc_expected = ratio_mag ** 2 * v_fluc_energy
    print(f"    ratio_mag = {ratio_mag:.4f}, "
          f"u_fluc_expected = {u_fluc_expected:.2f}")
    print(f"  Actual u_fluc = {u_fluc_energy:.2f}")
    print(f"  Excess = {u_fluc_energy / (u_fluc_expected + 1e-8):.2f}x")
    print(f"")
    print(f"  Synthetic task showed ~3.5x excess.")
    print(f"  Real cache: if < 1.5x, model has no extra noise in-distribution;")
    print(f"              if ~3x, the noise is intrinsic.")

# per-channel fluctuation ratio
    print(f"\nPer-channel fluctuation-to-total energy ratio (lower = more constant):")
    for c in range(C):
        u_tot = (u_in[c] ** 2).sum()
        u_fluct = ((u_in[c] - u_in[c].mean()) ** 2).sum()
        v_tot = (v_in[c] ** 2).sum()
        v_fluct = ((v_in[c] - v_in[c].mean()) ** 2).sum()
        print(f"  ch {c:2d}:  u {100*u_fluct/u_tot:.1f}%   v {100*v_fluct/v_tot:.1f}%")

if __name__ == "__main__":
    main()