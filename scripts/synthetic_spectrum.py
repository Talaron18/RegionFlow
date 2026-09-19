"""Frequency content of u vs v on the synthetic unambiguous task.

If u's amplitude falls off faster than v's at high frequency, the model
is producing a low-frequency approximation -- the conditional mean
signature. If u and v have comparable spectra, the problem is not
frequency-domain attenuation.
"""
import numpy as np
import torch


def radial_psd(x, mask, n_bins=10):
    """x: (C, h, w), mask: (h, w) bool. Returns (centers, psd)."""
    x = torch.from_numpy(x).float()
    m = torch.from_numpy(mask).float()
    xm = x * m.unsqueeze(0)
    F = torch.fft.fft2(xm.double(), dim=(-2, -1))
    P = F.abs().pow(2)
    C, h, w = x.shape
    ky = torch.fft.fftfreq(h).double()
    kx = torch.fft.fftfreq(w).double()
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K = (KY.pow(2) + KX.pow(2)).sqrt()
    K = K / K.max().clamp(min=1e-12)
    edges = torch.linspace(0, 1.0, n_bins + 1).double()
    psd = np.zeros(n_bins)
    centers = ((edges[:-1] + edges[1:]) / 2).numpy()
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        sel = (K >= lo) & (K <= hi) if b < n_bins - 1 else (K >= lo) & (K <= hi)
        n = int(sel.sum().item())
        if n == 0:
            continue
        psd[b] = P[:, sel].sum().item() / (C * n)
    return centers, psd


def main():
    d = np.load("runs/synthetic_u_v.npz")
    u = d["u"]          # (C, h, w)
    v = d["v"]
    mask = d["mask"].astype(bool)    # (h, w) bool
    print(f"u shape {u.shape}, v shape {v.shape}, mask sum {mask.sum()}")

    # global amplitudes
    u_in = u[:, mask]
    v_in = v[:, mask]
    print(f"\nglobal RMS inside mask:")
    print(f"  |u|_rms = {np.sqrt((u_in**2).mean()):.5f}")
    print(f"  |v|_rms = {np.sqrt((v_in**2).mean()):.5f}")
    print(f"  ratio = {np.sqrt((u_in**2).mean()) / np.sqrt((v_in**2).mean()):.3f}")

    # radial PSD
    centers, pu = radial_psd(u, mask)
    _, pv = radial_psd(v, mask)

    print(f"\nradial PSD:")
    print(f"  {'k (norm)':>10}  {'P_u':>12}  {'P_v':>12}  {'P_u/P_v':>10}  "
          f"{'amp_u/amp_v':>12}")
    for i, k in enumerate(centers):
        ratio = pu[i] / pv[i] if pv[i] > 1e-30 else np.nan
        amp_ratio = np.sqrt(ratio) if not np.isnan(ratio) else np.nan
        print(f"  {k:>10.4f}  {pu[i]:>12.4e}  {pv[i]:>12.4e}  "
              f"{ratio:>10.4f}  {amp_ratio:>12.4f}")

    # split low/high
    half = len(centers) // 2
    lf_ratio = pu[:half].sum() / pv[:half].sum() if pv[:half].sum() > 0 else np.nan
    hf_ratio = pu[half:].sum() / pv[half:].sum() if pv[half:].sum() > 0 else np.nan
    print(f"\nLF power ratio (k < 0.5): {lf_ratio:.4f}  "
          f"(amplitude: {np.sqrt(lf_ratio):.4f})")
    print(f"HF power ratio (k > 0.5): {hf_ratio:.4f}  "
          f"(amplitude: {np.sqrt(hf_ratio):.4f})")

    print(f"\nINTERPRETATION:")
    print(f"  If HF amplitude ratio is much lower than LF amplitude ratio,")
    print(f"  the model is producing a low-pass version of v -- conditional")
    print(f"  mean signature. Fix is architecture (higher-resolution output),")
    print(f"  not loss.")
    print(f"  If both ratios are similar and low (< 0.3), the model is")
    print(f"  globally under-amplitude -- fix is loss/objective.")
    print(f"  If both are close to 1, the problem is not frequency -- it's")
    print(f"  spatial alignment of high-frequency features.")


if __name__ == "__main__":
    main()
