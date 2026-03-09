#!/usr/bin/env python3
"""
cluster_retest.py
-----------------
Re-run the XXL cluster mass test with the PREDICTED UCT parameters:
  alpha_S = 1/phi = 0.618
  lambda  = 182 kpc
  Gamma   = K1 Bessel (covariant), NOT Gaussian

Compare with the original analysis (alpha_S = 0.71, Gaussian Gamma)
to see how the 8.0-sigma claim holds up.

Data: Umetsu et al. (2020, ApJ 890:148), 136 X-ray-selected systems.
"""

import numpy as np
from scipy.special import k1 as bessel_k1
from scipy.stats import spearmanr
import os

# ── Physical constants (SI) ──
G_SI    = 6.674e-11
MSUN    = 1.989e30
MPC_M   = 3.086e22
KPC_M   = 3.086e19
H0_SI   = 70e3 / MPC_M
OMEGA_M = 0.2793
OMEGA_L = 0.7207

# ── UCT parameters ──
PHI_GOLD    = (1.0 + np.sqrt(5)) / 2.0
A0_UCT      = 1.042e-10     # m/s^2
LAMBDA_PHI  = 182.0          # kpc
F_STELLAR   = 0.02

# The two alpha_S values to compare
ALPHA_PREDICTED = 1.0 / PHI_GOLD   # 0.618
ALPHA_ORIGINAL  = 0.71              # from original analysis


def E_z(z):
    return np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)

def rho_crit(z):
    H = H0_SI * E_z(z)
    return 3.0 * H**2 / (8.0 * np.pi * G_SI)

def f_gas(T_keV):
    return 0.11 * (T_keV / 3.0)**0.28

def R500_from_M500(M500_Msun, z):
    rho_c = rho_crit(z)
    R500_m = (3.0 * M500_Msun * MSUN / (4.0 * np.pi * 500.0 * rho_c))**(1.0/3.0)
    return R500_m / KPC_M

def M_MOND_deep(M_bary_Msun, R500_kpc):
    M_bary_kg = M_bary_Msun * MSUN
    R500_m    = R500_kpc * KPC_M
    M_MOND_kg = R500_m * np.sqrt(M_bary_kg * A0_UCT / G_SI)
    return M_MOND_kg / MSUN


# ── Gamma profiles ──
def gamma_gauss(R_kpc, lam=LAMBDA_PHI):
    return 1.0 - np.exp(-(R_kpc / lam)**2)

def gamma_k1(R_kpc, lam=LAMBDA_PHI):
    """Covariant K1 Bessel: Gamma = 1 - (R/lam)*K1(R/lam)"""
    x = np.asarray(R_kpc, dtype=float) / lam
    x = np.clip(x, 1e-10, None)
    return np.clip(1.0 - x * bessel_k1(x), 0.0, 1.0)


def load_umetsu2020(filepath):
    def T_from_M500(M_1e14Msun, z_val):
        return 3.0 * M_1e14Msun**0.65 * E_z(z_val)**0.4

    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('-') or not line:
                continue
            parts = line.split('\t')
            if parts[0] == 'recno' or parts[0] == ' ':
                continue
            try:
                z    = float(parts[4])
                M500 = float(parts[15])
                e_M500 = float(parts[16])
            except (ValueError, IndexError):
                continue
            try:
                T_keV = float(parts[6])
            except ValueError:
                T_keV = T_from_M500(M500, z)
            rows.append((z, T_keV, M500, e_M500))

    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2] * 1e14, arr[:, 3] * 1e14


def run_analysis(z, T_keV, M500_wl, e_M500_wl, alpha_s, gamma_fn, label):
    """Run the full cluster analysis with given alpha_S and Gamma profile."""
    N = len(z)
    R500_kpc = np.array([R500_from_M500(M, zz) for M, zz in zip(M500_wl, z)])
    fg       = f_gas(T_keV)
    M_bary   = (fg + F_STELLAR) * M500_wl
    M_MOND   = np.array([M_MOND_deep(Mb, R) for Mb, R in zip(M_bary, R500_kpc)])
    Gam      = gamma_fn(R500_kpc)
    corr     = 1.0 + alpha_s * Gam
    M_UCT    = M_MOND * corr

    resid_MOND = np.log10(M500_wl / M_MOND)
    resid_UCT  = np.log10(M500_wl / M_UCT)
    sigma_resid = e_M500_wl / (M500_wl * np.log(10))
    log_M = np.log10(M500_wl)

    # Group/cluster split
    grp = log_M < 13.5
    clu = log_M >= 13.5
    wt_g = 1.0 / sigma_resid[grp]**2
    wt_c = 1.0 / sigma_resid[clu]**2

    rm_grp = np.average(resid_MOND[grp], weights=wt_g)
    rm_clu = np.average(resid_MOND[clu], weights=wt_c)
    ru_grp = np.average(resid_UCT[grp],  weights=wt_g)
    ru_clu = np.average(resid_UCT[clu],  weights=wt_c)
    rm_grp_err = 1.0 / np.sqrt(wt_g.sum())
    rm_clu_err = 1.0 / np.sqrt(wt_c.sum())

    # Chi-squared
    chi2_MOND = np.sum(resid_MOND**2 / sigma_resid**2)
    chi2_UCT  = np.sum(resid_UCT**2 / sigma_resid**2)
    delta_chi2 = chi2_MOND - chi2_UCT

    # BIC: UCT has 0 extra params (alpha_S is predicted), MOND has 0
    # But in the comparison, UCT adds the K1 correction with alpha_S fixed
    # so delta_BIC = delta_chi2 (same number of parameters)
    # If we treat alpha_S as fitted: delta_BIC = delta_chi2 - ln(N)
    delta_BIC_0param = delta_chi2            # alpha_S predicted
    delta_BIC_1param = delta_chi2 - np.log(N)  # alpha_S fitted

    # Significance: sqrt(delta_chi2) for nested models with 0 extra params
    sigma_improvement = np.sqrt(abs(delta_chi2)) * np.sign(delta_chi2)

    # Overall UCT residual
    wt_all = 1.0 / sigma_resid**2
    ru_all = np.average(resid_UCT, weights=wt_all)
    ru_all_err = 1.0 / np.sqrt(wt_all.sum())

    # Spearman
    rho_mond, p_mond = spearmanr(log_M, resid_MOND)
    rho_uct, p_uct   = spearmanr(log_M, resid_UCT)

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  alpha_S = {alpha_s:.4f}, lambda = {LAMBDA_PHI} kpc")
    print(f"{'='*65}")
    print(f"  N = {N}")
    print(f"  R500 range: {R500_kpc.min():.0f} -- {R500_kpc.max():.0f} kpc")
    print(f"  Gamma range: {Gam.min():.4f} -- {Gam.max():.4f}")
    print(f"  Correction range: {corr.min():.4f} -- {corr.max():.4f}")
    print(f"\n  Overall:")
    print(f"    MOND residual: {resid_MOND.mean():+.4f} +/- {resid_MOND.std():.4f} dex")
    print(f"    UCT  residual: {ru_all:+.4f} +/- {ru_all_err:.4f} dex (weighted)")
    print(f"\n  Groups (log M < 13.5, N={grp.sum()}):")
    print(f"    MOND: {rm_grp:+.4f} +/- {rm_grp_err:.4f} dex")
    print(f"    UCT:  {ru_grp:+.4f} +/- {rm_grp_err:.4f} dex")
    print(f"\n  Clusters (log M >= 13.5, N={clu.sum()}):")
    print(f"    MOND: {rm_clu:+.4f} +/- {rm_clu_err:.4f} dex")
    print(f"    UCT:  {ru_clu:+.4f} +/- {rm_clu_err:.4f} dex")
    print(f"\n  chi2 MOND: {chi2_MOND:.1f}")
    print(f"  chi2 UCT:  {chi2_UCT:.1f}")
    print(f"  Delta chi2: {delta_chi2:.1f}")
    print(f"  Significance: {sigma_improvement:.1f} sigma")
    print(f"  Delta BIC (0 extra param): {-delta_BIC_0param:.1f}")
    print(f"  Delta BIC (1 extra param): {-delta_BIC_1param:.1f}")
    print(f"\n  Spearman (residual vs log M):")
    print(f"    MOND: rho={rho_mond:.3f}, p={p_mond:.4f}")
    print(f"    UCT:  rho={rho_uct:.3f}, p={p_uct:.4f}")

    return dict(
        alpha_s=alpha_s, chi2_MOND=chi2_MOND, chi2_UCT=chi2_UCT,
        delta_chi2=delta_chi2, sigma=sigma_improvement,
        ru_all=ru_all, ru_all_err=ru_all_err,
        ru_clu=ru_clu, ru_clu_err=rm_clu_err,
        ru_grp=ru_grp, ru_grp_err=rm_grp_err,
    )


def main():
    filepath = 'data/umetsu2020_table2.tsv'
    z, T_keV, M500_wl, e_M500_wl = load_umetsu2020(filepath)
    print(f"Loaded {len(z)} systems from Umetsu+2020")

    # Run all four combinations
    results = []
    results.append(run_analysis(z, T_keV, M500_wl, e_M500_wl,
                                ALPHA_ORIGINAL, gamma_gauss,
                                "ORIGINAL: Gaussian Gamma, alpha_S = 0.71"))
    results.append(run_analysis(z, T_keV, M500_wl, e_M500_wl,
                                ALPHA_PREDICTED, gamma_gauss,
                                "Gaussian Gamma, alpha_S = 1/phi = 0.618"))
    results.append(run_analysis(z, T_keV, M500_wl, e_M500_wl,
                                ALPHA_ORIGINAL, gamma_k1,
                                "K1 Bessel Gamma, alpha_S = 0.71"))
    results.append(run_analysis(z, T_keV, M500_wl, e_M500_wl,
                                ALPHA_PREDICTED, gamma_k1,
                                "PREDICTED: K1 Bessel Gamma, alpha_S = 1/phi = 0.618"))

    print("\n\n" + "="*65)
    print("  COMPARISON SUMMARY")
    print("="*65)
    labels = [
        "Gauss, a=0.71 (original)",
        "Gauss, a=0.618 (predicted)",
        "K1,    a=0.71",
        "K1,    a=0.618 (predicted)",
    ]
    print(f"{'Configuration':30s} {'chi2_UCT':>10s} {'Dchi2':>8s} {'sigma':>8s} "
          f"{'resid_clu':>12s} {'resid_all':>12s}")
    for lab, r in zip(labels, results):
        print(f"  {lab:28s} {r['chi2_UCT']:10.1f} {r['delta_chi2']:8.1f} "
              f"{r['sigma']:8.1f} "
              f"{r['ru_clu']:+.4f}+/-{r['ru_clu_err']:.4f} "
              f"{r['ru_all']:+.4f}+/-{r['ru_all_err']:.4f}")


if __name__ == '__main__':
    main()
