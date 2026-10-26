"""
Focal spots of a collimated Gaussian beam through the Thorlabs LMM40X-P01
for a HeNe (632.8 nm) and a 2923 nm laser and several input beam diameters,
plus the best-fitting Gaussian for each spot.

Run:  .venv/Scripts/python.exe run_sim.py
Output: results/  (figures, summary.csv, summary.md)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize

import lmm40x as L

OUT = "results"
WAVELENGTHS = {"632.8nm": 632.8e-6, "2923nm": 2923e-6}   # mm
D_IN = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 8.0]   # input 1/e^2 beam diameters, mm
NPUP = 301                                     # pupil samples across 5.1 mm
NIMG = 241                                     # focal-plane samples per axis
HALF = 8.0                                     # half window in units of lambda
POL = (1.0, 0.0)                               # x-polarised input


def gauss2d(p, X, Y):
    A, x0, y0, wx, wy = p
    return A * np.exp(-2 * ((X - x0) ** 2 / wx ** 2 + (Y - y0) ** 2 / wy ** 2))


def fit_intensity(I, X, Y, w_guess):
    """Least-squares fit of a 2-D elliptical Gaussian to the total intensity
    (whole window, uniform weights)."""
    p0 = [I.max(), 0.0, 0.0, w_guess, w_guess]
    res = least_squares(lambda p: (gauss2d(p, X, Y) - I).ravel(), p0,
                        x_scale=[I.max(), w_guess, w_guess, w_guess, w_guess])
    return res.x


def fit_mode(E, X, Y, dA, w_guess):
    """Gaussian beam maximising the power coupling (mode overlap) with the
    full vector field at the focal plane:
      eta = |<G x_hat, E>|^2 / (<G,G> <E,E>),  G = exp(-x^2/wx^2 - y^2/wy^2)
    (flat phase: waist in this plane).  Returns wx, wy, eta."""
    P = np.sum(np.abs(E) ** 2) * dA

    def neg_eta(q):
        wx, wy = np.exp(q)
        G = np.exp(-X ** 2 / wx ** 2 - Y ** 2 / wy ** 2)
        ov = np.sum(G * E[0]) * dA
        return -np.abs(ov) ** 2 / (np.sum(G ** 2) * dA * P)

    r = minimize(neg_eta, np.log([w_guess, w_guess]), method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-8))
    wx, wy = np.exp(r.x)
    return wx, wy, -r.fun


def through_focus(pupil, amp, wl, zrange):
    """On-axis intensity vs defocus (vector)."""
    z = np.linspace(-zrange, zrange, 161)
    I = []
    for zi in z:
        E = L.focal_field(pupil, amp, wl, [0.0], [0.0], z=zi, pol=POL)
        I.append(np.sum(np.abs(E) ** 2))
    return z, np.array(I)


def plot_pupil(pupil, fname):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    g = pupil.grid
    ax[0].imshow(pupil.mask_full, extent=[g[0], g[-1], g[0], g[-1]],
                 origin="lower", cmap="gray")
    ax[0].set_title("Transmitting entrance-pupil area (white)")
    ax[0].set_xlabel("x$_0$ (mm)")
    ax[0].set_ylabel("y$_0$ (mm)")
    Wfull = np.full(pupil.mask_full.shape, np.nan)
    Wfull[pupil.mask_full] = pupil.W / 632.8e-6
    im = ax[1].imshow(Wfull, extent=[g[0], g[-1], g[0], g[-1]],
                      origin="lower", cmap="RdBu_r")
    ax[1].set_title("Residual wavefront at best focus (waves @ 632.8 nm)")
    ax[1].set_xlabel("x$_0$ (mm)")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_layout(fname):
    """Meridional ray layout from the real trace."""
    r = np.linspace(-2.55, 2.55, 21)
    tr = L.trace(r, 0 * r, spiders=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ri in enumerate(r):
        q =L.trace([ri], [0], spiders=False)
        # secondary hit, primary hit, focus
        sec_z = L.Z_SEC + L.C_SEC * ri ** 2 / (1 + np.sqrt(1 - (L.C_SEC * ri) ** 2))
        P = q["P"][0]
        s = q["s"][0]
        t = (L.Z_IMG - P[2]) / s[2]
        F = P + t * s
        col = "tab:red" if tr["ok"][i] else "0.75"
        ax.plot([L.Z_ENT, sec_z, P[2], F[2]], [ri, ri, P[0], F[0]], color=col,
                lw=0.8)
    zz = np.linspace(-10.75, 10.75, 200)
    for c, zv, lo, hi in [(L.C_PRI, L.Z_PRI, L.R_PRI_IN, L.R_PRI_OUT),
                          (L.C_SEC, L.Z_SEC, 0, L.R_SEC_CLEAR)]:
        y = zz[(np.abs(zz) >= lo) & (np.abs(zz) <= hi)]
        for sgn in (1, -1):
            yy = y[sgn * y > 0]
            ax.plot(zv + c * yy ** 2 / (1 + np.sqrt(1 - (c * yy) ** 2)), yy,
                    "k", lw=2)
    ax.axvline(L.Z_IMG, color="b", ls=":", lw=1)
    ax.plot([L.Z_S6, L.Z_S6], [-L.R_HOUSING, L.R_HOUSING], "k", lw=3)
    ax.set_aspect("equal")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("LMM40X-P01 meridional trace (red: transmitted, grey: "
                 "obscured)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    efl, zpar = L.paraxial_properties()
    zbest, rms, _ = L.best_focus()
    pupil = L.Pupil(n=NPUP)
    print(f"EFL {efl:.4f} mm, paraxial focus z={zpar:.4f}, best focus "
          f"z={zbest:.4f}, rms WFE {rms*1e6:.2f} nm")
    plot_pupil(pupil, f"{OUT}/pupil.png")
    plot_layout(f"{OUT}/layout.png")

    rows = []
    for wname, wl in WAVELENGTHS.items():
        x = np.linspace(-HALF * wl, HALF * wl, NIMG)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, x)
        fig_sum, axs = plt.subplots(len(D_IN), 4, figsize=(17, 3.6 * len(D_IN)))
        for iD, D in enumerate(D_IN):
            w_in = D / 2
            amp = L.gaussian_input(pupil, w_in)
            T = L.gaussian_power_fractions(w_in, pupil)
            E = L.focal_field(pupil, amp, wl, x, x, z=0.0, pol=POL)
            Ic = np.abs(E) ** 2
            I = Ic.sum(0)
            frac_comp = Ic.reshape(3, -1).sum(1) / I.sum()
            # ideal (unobscured, unclipped, paraxial) Gaussian focus
            w0_ideal = wl * efl / (np.pi * w_in)
            w_guess = max(0.4 * wl / 0.5, min(w0_ideal, 2 * wl))
            pI = fit_intensity(I / I.max(), X, Y, w_guess)
            wxm, wym, eta = fit_mode(E, X, Y, dx * dx, w_guess)
            # second-moment (D4sigma) widths inside the window (ring sensitive)
            Pn = I / I.sum()
            sx = np.sqrt(np.sum(Pn * X ** 2))
            sy = np.sqrt(np.sum(Pn * Y ** 2))
            # power in the central lobe: encircled to the first minimum on x
            cut = I[NIMG // 2, NIMG // 2:]
            i_min = np.argmax(np.diff(cut) > 0)
            r1 = x[NIMG // 2 + i_min]
            core = np.sum(I[np.hypot(X, Y) <= r1]) / I.sum()
            # FWHM along x and y
            def fwhm(c):
                h = c / c.max() - 0.5
                i = np.where(np.diff(np.sign(h)) != 0)[0]
                xs = [x[j] - h[j] * (x[j + 1] - x[j]) / (h[j + 1] - h[j])
                      for j in i]
                return xs[-1] - xs[0] if len(xs) >= 2 else np.nan
            fx = fwhm(I[NIMG // 2, :])
            fy = fwhm(I[:, NIMG // 2])
            zf, If = through_focus(pupil, amp, wl, 4 * wl / 0.5 ** 2)
            dz_peak = zf[np.argmax(If)]
            row = dict(
                wavelength=wname, D_in_mm=D, transmission=T,
                w0_ideal_um=w0_ideal * 1e3,
                fwhm_x_um=fx * 1e3, fwhm_y_um=fy * 1e3,
                fitI_wx_um=abs(pI[3]) * 1e3, fitI_wy_um=abs(pI[4]) * 1e3,
                mode_wx_um=wxm * 1e3, mode_wy_um=wym * 1e3,
                mode_coupling=eta,
                D4s_x_um=4 * sx * 1e3, D4s_y_um=4 * sy * 1e3,
                first_min_um=r1 * 1e3, core_power=core,
                frac_Ex=frac_comp[0], frac_Ey=frac_comp[1],
                frac_Ez=frac_comp[2], focus_shift_um=dz_peak * 1e3)
            rows.append(row)
            print(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else
                            f"{k}={v}" for k, v in row.items()))

            # ---- per-case figure
            um = x * 1e3
            ext = [um[0], um[-1], um[0], um[-1]]
            fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
            im = ax[0, 0].imshow(I / I.max(), extent=ext, origin="lower",
                                 cmap="inferno")
            ax[0, 0].set_title("|E|$^2$ (linear)")
            fig.colorbar(im, ax=ax[0, 0])
            im = ax[0, 1].imshow(np.log10(I / I.max() + 1e-6), extent=ext,
                                 origin="lower", cmap="inferno", vmin=-5)
            ax[0, 1].set_title("log$_{10}$ |E|$^2$")
            fig.colorbar(im, ax=ax[0, 1])
            ph = np.angle(E[0] * np.exp(-1j * np.angle(E[0, NIMG // 2,
                                                           NIMG // 2])))
            ph = np.ma.masked_where(Ic[0] < 1e-4 * Ic[0].max(), ph)
            im = ax[0, 2].imshow(ph, extent=ext, origin="lower",
                                 cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax[0, 2].set_title("phase of E$_x$ (rad), masked < 1e-4")
            fig.colorbar(im, ax=ax[0, 2])
            for a in ax[0]:
                a.set_xlabel("x (µm)")
                a.set_ylabel("y (µm)")
            G = gauss2d([1, pI[1], pI[2], pI[3], pI[4]], X, Y)
            Gm = np.exp(-2 * (X ** 2 / wxm ** 2 + Y ** 2 / wym ** 2))
            c = NIMG // 2
            ax[1, 0].plot(um, I[c] / I.max(), "k", label="sim x-cut")
            ax[1, 0].plot(um, I[:, c] / I.max(), "k--", label="sim y-cut")
            ax[1, 0].plot(um, G[c], "r", label="LSQ Gauss x")
            ax[1, 0].plot(um, Gm[c], "b", label="mode-matched Gauss x")
            ax[1, 0].set_xlim(-4 * wl * 1e3 / 1, 4 * wl * 1e3)
            ax[1, 0].legend(fontsize=8)
            ax[1, 0].set_xlabel("µm")
            ax[1, 1].semilogy(um, I[c] / I.max(), "k")
            ax[1, 1].semilogy(um, I[:, c] / I.max(), "k--")
            ax[1, 1].semilogy(um, G[c], "r")
            ax[1, 1].set_ylim(1e-6, 1.5)
            ax[1, 1].set_xlabel("µm")
            ax[1, 1].set_title("cuts (log)")
            ax[1, 2].plot(zf * 1e3, If / If.max(), "k")
            ax[1, 2].set_xlabel("defocus z (µm)")
            ax[1, 2].set_title("on-axis intensity through focus")
            fig.suptitle(
                f"LMM40X-P01, {wname}, input 1/e² diameter {D} mm "
                f"(T = {T*100:.1f}%)\nLSQ Gauss w = {pI[3]*1e3:.3f} × "
                f"{pI[4]*1e3:.3f} µm; mode-matched w = {wxm*1e3:.3f} × "
                f"{wym*1e3:.3f} µm, coupling {eta*100:.1f}%")
            fig.tight_layout()
            fig.savefig(f"{OUT}/spot_{wname}_D{D:g}mm.png", dpi=110)
            plt.close(fig)

            # ---- summary grid
            a = axs[iD]
            a[0].imshow(I / I.max(), extent=ext, origin="lower", cmap="inferno")
            a[0].set_ylabel(f"D = {D} mm\nT = {T*100:.1f}%")
            a[1].imshow(np.log10(I / I.max() + 1e-6), extent=ext,
                        origin="lower", cmap="inferno", vmin=-5)
            a[2].imshow(ph, extent=ext, origin="lower", cmap="twilight",
                        vmin=-np.pi, vmax=np.pi)
            a[3].plot(um, I[c] / I.max(), "k", um, G[c], "r", um, Gm[c], "b")
            a[3].set_xlim(-4 * wl * 1e3, 4 * wl * 1e3)
        for j, t in enumerate(["|E|² linear", "log10 |E|²", "phase Ex",
                               "x-cut: sim (k), LSQ (r), mode (b)"]):
            axs[0, j].set_title(t)
        fig_sum.suptitle(f"LMM40X-P01 focal spots, {wname} (axes in µm)")
        fig_sum.tight_layout()
        fig_sum.savefig(f"{OUT}/summary_{wname}.png", dpi=80)
        plt.close(fig_sum)

    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
        f.write("| λ | D_in (mm) | T | LSQ w_x × w_y (µm) | mode w_x × w_y "
                "(µm) | coupling η | T·η | FWHM x × y (µm) | ideal w0 (µm) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['wavelength']} | {r['D_in_mm']:g} | "
                    f"{r['transmission']*100:.2f}% | {r['fitI_wx_um']:.3f} × "
                    f"{r['fitI_wy_um']:.3f} | {r['mode_wx_um']:.3f} × "
                    f"{r['mode_wy_um']:.3f} | {r['mode_coupling']*100:.1f}% | "
                    f"{r['transmission']*r['mode_coupling']*100:.1f}% | "
                    f"{r['fwhm_x_um']:.3f} × {r['fwhm_y_um']:.3f} | "
                    f"{r['w0_ideal_um']:.3f} |\n")


if __name__ == "__main__":
    main()
