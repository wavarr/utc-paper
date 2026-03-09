#!/usr/bin/env python3
"""
pergalaxy_regime.py -- Per-galaxy chi2 decomposition with acceleration-regime analysis
=====================================================================================

Generates pergalaxy_regime.png for the "Per-galaxy decomposition and
acceleration-regime dependence" subsection of Paper I.

Three panels:
  (a) Delta_chi2 vs median log10(g_bar/a0) with Spearman correlation
  (b) Cumulative Delta_chi2 sorted by Delta_chi2
  (c) Histogram split by median acceleration

Here Delta_chi2_i = chi2_Simple_i - chi2_UCT_i  (positive = UCT better).
Uses TOTAL chi2 per galaxy (not reduced), since the per-galaxy comparison
should not divide by N_pts (which varies across galaxies and would obscure
the contribution of data-rich galaxies).

Author: Gabriel Steiger
"""

import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Physical constants (identical to reproduce_sparc.py)
# ---------------------------------------------------------------------------
C_LIGHT    = 2.99792458e8
H0_KMS_MPC = 67.30
MPC_M      = 3.0856775814913673e22
KPC_M      = 3.0856775814913673e19
H0_SI      = H0_KMS_MPC * 1e3 / MPC_M

PHI        = (1.0 + np.sqrt(5.0)) / 2.0
N_UCT      = np.log(2.0) / np.log(PHI**2)
A0_UCT     = C_LIGHT * H0_SI / (2.0 * np.pi)

UPSILON_DISK  = 0.5
UPSILON_BULGE = 0.7


# ---------------------------------------------------------------------------
#  Data loading (identical to reproduce_sparc.py)
# ---------------------------------------------------------------------------
def find_sparc_data():
    candidates = [
        Path(__file__).parent / "SPARC_table2.dat",
        Path(__file__).parent / "data" / "SPARC_table2.dat",
        Path("data/SPARC_table2.dat"),
    ]
    for p in candidates:
        if p.exists():
            return p
    print("ERROR: Cannot find SPARC_table2.dat.")
    sys.exit(1)


def load_sparc(filepath=None):
    if filepath is None:
        filepath = find_sparc_data()
    names_list = []
    R_kpc, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            names_list.append(parts[0])
            R_kpc.append(float(parts[2]))
            V_obs.append(float(parts[3]))
            V_err.append(float(parts[4]))
            V_gas.append(float(parts[5]))
            V_disk.append(float(parts[6]))
            V_bulge.append(float(parts[7]))
    R  = np.array(R_kpc) * KPC_M
    Vo = np.array(V_obs) * 1e3
    Ve = np.array(V_err) * 1e3
    Vg = np.array(V_gas) * 1e3
    Vd = np.array(V_disk) * 1e3
    Vb = np.array(V_bulge) * 1e3
    names = np.array(names_list)
    g_obs = Vo**2 / R
    g_bar = (UPSILON_DISK * Vd**2 + UPSILON_BULGE * Vb**2
             + Vg * np.abs(Vg)) / R
    sigma_g = 2.0 * Vo * Ve / R
    mask = (g_bar > 0) & (g_obs > 0) & (sigma_g > 0) & (sigma_g < g_obs)
    return g_obs[mask], g_bar[mask], sigma_g[mask], names[mask]


# ---------------------------------------------------------------------------
#  Interpolation functions (identical to reproduce_sparc.py)
# ---------------------------------------------------------------------------
def _solve_aqual_family(g_bar, a0, n):
    g_pred = np.empty_like(g_bar)
    for i, gb in enumerate(g_bar):
        x_N = gb / a0
        def f(x, _xN=x_N, _n=n):
            return x**2 / (1.0 + x**_n)**(1.0/_n) - _xN
        x_hi = max(x_N * 2.0, np.sqrt(x_N) * 3.0, 10.0)
        while f(x_hi) < 0:
            x_hi *= 5.0
        g_pred[i] = brentq(f, 1e-15, x_hi, xtol=1e-12) * a0
    return g_pred


def g_pred_uct(g_bar, a0):
    return _solve_aqual_family(g_bar, a0, N_UCT)


def g_pred_simple(g_bar, a0):
    x_N = g_bar / a0
    x = 0.5 * (x_N + np.sqrt(x_N**2 + 4.0 * x_N))
    return x * a0


def fit_a0(g_obs, g_bar, sigma_g, pred_func, a0_range=(5e-11, 2e-10)):
    def objective(a0):
        g_pred = pred_func(g_bar, a0)
        log_res = np.log10(g_obs) - np.log10(g_pred)
        sig_log = sigma_g / (g_obs * np.log(10.0))
        return np.sum(log_res**2 / sig_log**2) / (len(g_obs) - 1)
    result = minimize_scalar(objective, bounds=a0_range, method="bounded")
    return result.x, result.fun


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    print("Loading SPARC data...")
    g_obs, g_bar, sigma_g, names = load_sparc()
    unique_gals = np.unique(names)
    N_gal = len(unique_gals)
    print(f"  {len(g_obs)} data points, {N_gal} galaxies")

    # Fit a0 for Simple
    a0_simple, _ = fit_a0(g_obs, g_bar, sigma_g, g_pred_simple)
    print(f"  UCT a0    = {A0_UCT:.4e} m/s^2  (predicted)")
    print(f"  Simple a0 = {a0_simple:.4e} m/s^2  (fitted)")

    # Compute per-galaxy TOTAL chi2 (not reduced) for both models
    gal_data = {}
    print("\nComputing per-galaxy chi2...")
    for gal in unique_gals:
        mask = names == gal
        n_pts = mask.sum()
        if n_pts < 2:
            continue

        go = g_obs[mask]
        gb = g_bar[mask]
        sg = sigma_g[mask]

        # UCT prediction
        gp_uct = g_pred_uct(gb, A0_UCT)
        log_res_uct = np.log10(go) - np.log10(gp_uct)
        sig_log = sg / (go * np.log(10.0))
        chi2_uct = np.sum(log_res_uct**2 / sig_log**2)

        # Simple prediction
        gp_sim = g_pred_simple(gb, a0_simple)
        log_res_sim = np.log10(go) - np.log10(gp_sim)
        chi2_sim = np.sum(log_res_sim**2 / sig_log**2)

        # Median baryonic acceleration for this galaxy
        med_gbar = np.median(gb)

        gal_data[gal] = {
            "n_pts": n_pts,
            "chi2_uct": chi2_uct,
            "chi2_sim": chi2_sim,
            "delta_chi2": chi2_sim - chi2_uct,  # positive = UCT better
            "med_gbar": med_gbar,
            "med_log_gbar_a0": np.log10(med_gbar / A0_UCT),
        }

    gal_names = sorted(gal_data.keys())
    delta_chi2 = np.array([gal_data[g]["delta_chi2"] for g in gal_names])
    med_log_x = np.array([gal_data[g]["med_log_gbar_a0"] for g in gal_names])
    n_pts_arr = np.array([gal_data[g]["n_pts"] for g in gal_names])
    N = len(gal_names)

    # Statistics
    n_uct_better = np.sum(delta_chi2 > 0)
    n_sim_better = np.sum(delta_chi2 < 0)
    sum_delta = np.sum(delta_chi2)

    rho, p_val = spearmanr(med_log_x, delta_chi2)

    print(f"\n  UCT better: {n_uct_better}/{N} galaxies")
    print(f"  Simple better: {n_sim_better}/{N} galaxies")
    print(f"  Sum(Delta chi2) = {sum_delta:.0f}")
    print(f"  Spearman rho = {rho:.3f}, p = {p_val:.2e}")

    # Key galaxies
    for gal in ["NGC5055", "UGC02953"]:
        if gal in gal_data:
            d = gal_data[gal]
            print(f"  {gal}: Delta_chi2 = {d['delta_chi2']:.0f}, "
                  f"n_pts = {d['n_pts']}, "
                  f"chi2_UCT = {d['chi2_uct']:.0f}, "
                  f"chi2_Simple = {d['chi2_sim']:.0f}")

    # Excluding NGC5055
    mask_no5055 = np.array([g != "NGC5055" for g in gal_names])
    sum_no5055 = np.sum(delta_chi2[mask_no5055])
    rho_no5055, p_no5055 = spearmanr(med_log_x[mask_no5055], delta_chi2[mask_no5055])
    print(f"\n  Excluding NGC5055:")
    print(f"    Sum(Delta chi2) = {sum_no5055:.0f}")
    print(f"    Spearman rho = {rho_no5055:.3f}, p = {p_no5055:.2e}")

    # Split by median acceleration
    med_split = np.median(med_log_x)
    low_mask = med_log_x <= med_split
    high_mask = med_log_x > med_split
    n_low = low_mask.sum()
    n_high = high_mask.sum()
    n_uct_low = np.sum(delta_chi2[low_mask] > 0)
    n_uct_high = np.sum(delta_chi2[high_mask] > 0)
    print(f"\n  Split at median log10(g_bar/a0) = {med_split:.2f}:")
    print(f"    Low-acc: N={n_low}, UCT wins {n_uct_low}")
    print(f"    High-acc: N={n_high}, UCT wins {n_uct_high}")

    # Surface brightness correlation
    # Use median g_bar as a proxy for surface brightness
    rho_sb, p_sb = spearmanr(med_log_x, delta_chi2)
    print(f"\n  Surface brightness proxy correlation: rho={rho_sb:.3f}, p={p_sb:.2e}")

    # -----------------------------------------------------------------------
    #  Figure
    # -----------------------------------------------------------------------
    print("\nGenerating figure...")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        r"Per-Galaxy $\chi^2$ Decomposition: UCT ($n = 0.72$, predicted $a_0$) "
        r"vs Simple ($n = 1$, fitted $a_0$)",
        fontsize=14, fontweight="bold", y=1.02
    )

    # --- Panel (a): Delta_chi2 vs median acceleration ---
    colors_a = np.where(delta_chi2 > 0, "#4CAF50", "#E57373")
    sizes = 15 + 120 * (n_pts_arr / n_pts_arr.max())

    ax_a.scatter(med_log_x, delta_chi2, c=colors_a, s=sizes,
                 alpha=0.65, edgecolors="none", zorder=3)
    ax_a.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax_a.axvline(0, color="gray", ls=":", lw=0.5, alpha=0.5)

    ax_a.set_xlabel(r"Median $\log_{10}\,(g_{\rm bar}/a_0)$", fontsize=12)
    ax_a.set_ylabel(r"$\Delta\chi^2 = \chi^2_{\rm Simple} - \chi^2_{\rm UCT}$",
                     fontsize=12)
    ax_a.set_title(f"(a) UCT advantage vs acceleration\n"
                   f"(Spearman $\\rho$ = {rho:.3f}, p = {p_val:.1e})",
                   fontsize=11)
    ax_a.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
                       markersize=8, label=f"Green: UCT better ({n_uct_better}/{N})"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E57373',
                       markersize=8, label=f"Red: Simple better ({n_sim_better}/{N})"),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markersize=4, label=r"Dot size $\propto$ N_points"),
        ],
        fontsize=8, loc="upper left", framealpha=0.9
    )
    ax_a.grid(alpha=0.15)

    # --- Panel (b): Cumulative Delta_chi2 ---
    sorted_delta = np.sort(delta_chi2)[::-1]  # largest (UCT best) first
    cumsum = np.cumsum(sorted_delta)
    ranks = np.arange(1, N + 1)

    ax_b.fill_between(ranks, cumsum, 0, where=cumsum >= 0,
                       color="#4CAF50", alpha=0.3)
    ax_b.fill_between(ranks, cumsum, 0, where=cumsum < 0,
                       color="#E57373", alpha=0.3)
    ax_b.plot(ranks, cumsum, "k-", lw=1.5)
    ax_b.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    ax_b.set_xlabel(r"Galaxy rank (sorted by $\Delta\chi^2$)", fontsize=12)
    ax_b.set_ylabel(r"Cumulative $\Delta\chi^2$", fontsize=12)
    ax_b.set_title("(b) Cumulative advantage", fontsize=11)
    ax_b.grid(alpha=0.15)

    # --- Panel (c): Histogram split by median acceleration ---
    bins_c = np.linspace(
        max(delta_chi2.min(), -10000),
        min(delta_chi2.max(), 10000),
        40
    )

    ax_c.hist(delta_chi2[low_mask], bins=bins_c, alpha=0.5,
              color="#E57373", edgecolor="none",
              label=f"Low-acc (N={n_low}, UCT wins {n_uct_low})")
    ax_c.hist(delta_chi2[high_mask], bins=bins_c, alpha=0.5,
              color="#5C9BD5", edgecolor="none",
              label=f"High-acc (N={n_high}, UCT wins {n_uct_high})")
    ax_c.axvline(0, color="k", ls="--", lw=0.8, alpha=0.7)

    ax_c.set_xlabel(r"$\Delta\chi^2$ (positive = UCT better)", fontsize=12)
    ax_c.set_ylabel("Number of galaxies", fontsize=12)
    ax_c.set_title("(c) Split by median acceleration", fontsize=11)
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(alpha=0.15)

    fig.tight_layout()

    outpath = Path(__file__).parent.parent / "figures" / "pergalaxy_regime.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")

    # Also save to Paper_Pub2 for cross-reference
    outpath2 = Path(__file__).resolve().parent.parent.parent / "Paper_Pub2" / "figures" / "pergalaxy_regime.png"
    if outpath2.parent.exists():
        fig.savefig(outpath2, dpi=200, bbox_inches="tight")
        print(f"  Saved: {outpath2}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
