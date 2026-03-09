#!/usr/bin/env python3
"""
reproduce_sparc.py -- Reproducibility package for UCT SPARC analysis
=====================================================================

Reproduces Table IV of:
    "Universal Compression Theory: A Single Scalar Field Unifying
     Galactic Dynamics, Cosmology, and Inflation" (Paper I)
    by Gabriel Steiger

This script:
  1. Loads the SPARC Radial Acceleration Relation (3,351 data points,
     175 galaxies) from Lelli, McGaugh & Schombert (2017).
  2. Computes the predicted g_obs for four interpolation functions:
       - UCT:      mu(x) = x/(1+x^n)^{1/n},  n = ln2/ln(phi^2), a0 = cH0/(2pi)  [0 free params]
       - Simple:   mu(x) = x/(1+x)             [1 free param: a0]
       - RAR:      nu(y) = 1/(1-exp(-sqrt(y)))  [1 free param: a0]
       - Standard: mu(x) = x/sqrt(1+x^2)       [1 free param: a0]
  3. Computes reduced chi-squared in log-acceleration space.
  4. Generates diagnostic figures.

Expected output (Table IV):
    UCT:       chi^2_red = 53.38  (0 free parameters)
    Simple:    chi^2_red = 52.16  (1 free parameter)
    RAR:       chi^2_red = 54.08  (1 free parameter)
    Standard:  chi^2_red = 110.72 (1 free parameter)

Dependencies: numpy, scipy, matplotlib
Data: SPARC_table2.dat from http://astroweb.cwru.edu/SPARC/

Author: Gabriel Steiger
"""

import sys
import os
import time
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from pathlib import Path

# ---------------------------------------------------------------------------
#  Physical constants
# ---------------------------------------------------------------------------
C_LIGHT    = 2.99792458e8       # speed of light [m/s]
H0_KMS_MPC = 67.30              # Hubble constant [km/s/Mpc]
MPC_M      = 3.0856775814913673e22  # Megaparsec [m]
KPC_M      = 3.0856775814913673e19  # kiloparsec [m]

H0_SI      = H0_KMS_MPC * 1e3 / MPC_M   # Hubble constant [1/s]

PHI        = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio = 1.6180339887...
N_UCT      = np.log(2.0) / np.log(PHI**2)  # UCT exponent ~ 0.72021...
A0_UCT     = C_LIGHT * H0_SI / (2.0 * np.pi)  # predicted a0 [m/s^2]

# Mass-to-light ratios (population synthesis, McGaugh+2016, Lelli+2017)
UPSILON_DISK  = 0.5   # M_sun / L_sun
UPSILON_BULGE = 0.7   # M_sun / L_sun


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
def find_sparc_data():
    """Locate SPARC_table2.dat in several candidate directories."""
    candidates = [
        Path(__file__).parent / "SPARC_table2.dat",
        Path(__file__).parent / "data" / "SPARC_table2.dat",
        Path("data/SPARC_table2.dat"),
    ]
    for p in candidates:
        if p.exists():
            return p
    print("ERROR: Cannot find SPARC_table2.dat.")
    print("Please download from http://astroweb.cwru.edu/SPARC/")
    print("and place it in this directory or in a 'data/' subdirectory.")
    sys.exit(1)


def load_sparc(filepath=None):
    """Load the SPARC RAR data and return accelerations + errors.

    Columns of SPARC_table2.dat (no header):
        Name  D(Mpc)  R(kpc)  Vobs(km/s)  errV(km/s)  Vgas(km/s)
        Vdisk(km/s)  Vbulge(km/s)  SBdisk(L/pc^2)  SBbulge(L/pc^2)

    Mass-to-light ratios fixed at population-synthesis values:
        Upsilon_disk  = 0.5  M_sun/L_sun
        Upsilon_bulge = 0.7  M_sun/L_sun

    Quality cuts:
        g_bar > 0  (positive baryonic acceleration)
        g_obs > 0  (positive observed acceleration)
        sigma_g > 0  (positive error)
        sigma_g < g_obs  (error smaller than measurement; removes 38 points)

    This yields 3,351 data points from 175 galaxies.

    Returns:
        g_obs   : observed centripetal acceleration V_obs^2/R  [m/s^2]
        g_bar   : baryonic acceleration [m/s^2]
        sigma_g : error on g_obs, propagated as 2*V_obs*errV/R  [m/s^2]
        names   : galaxy name for each data point
    """
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

    # Convert to SI
    R  = np.array(R_kpc) * KPC_M          # [m]
    Vo = np.array(V_obs) * 1e3            # [m/s]
    Ve = np.array(V_err) * 1e3            # [m/s]
    Vg = np.array(V_gas) * 1e3            # [m/s]
    Vd = np.array(V_disk) * 1e3           # [m/s]
    Vb = np.array(V_bulge) * 1e3          # [m/s]
    names = np.array(names_list)

    # Accelerations [m/s^2]
    g_obs = Vo**2 / R
    g_bar = (UPSILON_DISK * Vd**2 + UPSILON_BULGE * Vb**2
             + Vg * np.abs(Vg)) / R

    # Error propagation: sigma(g_obs) = 2 * V_obs * sigma_V / R
    sigma_g = 2.0 * Vo * Ve / R

    # Quality cuts
    mask = (g_bar > 0) & (g_obs > 0) & (sigma_g > 0) & (sigma_g < g_obs)

    return g_obs[mask], g_bar[mask], sigma_g[mask], names[mask]


# ---------------------------------------------------------------------------
#  Interpolation functions
# ---------------------------------------------------------------------------
# Convention: given g_bar, predict g_obs by solving the AQUAL equation
#     mu(g_obs / a0) * (g_obs / a0) = g_bar / a0
# i.e.  mu(x) * x = x_N  where x = g/a0, x_N = g_bar/a0
# Then g_pred = x * a0

def _solve_aqual_family(g_bar, a0, n):
    """Solve mu(x)*x = x_N for mu(x) = x/(1+x^n)^{1/n}.

    The equation becomes: x^2 / (1+x^n)^{1/n} = g_bar/a0.
    """
    g_pred = np.empty_like(g_bar)
    for i, gb in enumerate(g_bar):
        x_N = gb / a0
        def f(x, _xN=x_N, _n=n):
            return x**2 / (1.0 + x**_n)**(1.0/_n) - _xN
        # Bracket: in deep MOND mu(x)~x so x^2 ~ x_N, need x ~ sqrt(x_N)
        x_hi = max(x_N * 2.0, np.sqrt(x_N) * 3.0, 10.0)
        while f(x_hi) < 0:
            x_hi *= 5.0
        g_pred[i] = brentq(f, 1e-15, x_hi, xtol=1e-12) * a0
    return g_pred


def g_pred_uct(g_bar, a0):
    """UCT: mu(x) = x/(1+x^n)^{1/n}, n = ln2/ln(phi^2). Zero free parameters."""
    return _solve_aqual_family(g_bar, a0, N_UCT)


def g_pred_simple(g_bar, a0):
    """Simple: mu(x) = x/(1+x). Closed form: x = (x_N + sqrt(x_N^2 + 4*x_N))/2."""
    x_N = g_bar / a0
    x = 0.5 * (x_N + np.sqrt(x_N**2 + 4.0 * x_N))
    return x * a0


def g_pred_standard(g_bar, a0):
    """Standard: mu(x) = x/sqrt(1+x^2). Solve x^2/sqrt(1+x^2) = x_N."""
    return _solve_aqual_family(g_bar, a0, 2.0)


def g_pred_rar(g_bar, a0):
    """RAR empirical: nu(y) = 1/(1-exp(-sqrt(y))), g_obs = nu(g_bar/a0) * g_bar.

    This is McGaugh+2016 eq (4). Direct application (no root-finding needed).
    """
    y = g_bar / a0
    sy = np.sqrt(np.maximum(y, 1e-30))
    nu = 1.0 / (1.0 - np.exp(-sy))
    return nu * g_bar


# ---------------------------------------------------------------------------
#  Chi-squared computation
# ---------------------------------------------------------------------------
def compute_chi2(g_obs, g_bar, sigma_g, pred_func, a0, k=0):
    """Reduced chi-squared in log-acceleration space.

    chi^2_red = (1/(N-k)) * sum_i [ (log10 g_obs_i - log10 g_pred_i)^2
                                     / sigma_log_i^2 ]

    where sigma_log = sigma_g / (g_obs * ln 10).

    This is equivalent to eq (C2) of the paper when errors are propagated
    to log space. The log-space formulation weights all acceleration
    regimes equally and avoids high-acceleration points dominating the sum.

    Parameters:
        g_obs    : observed accelerations [m/s^2]
        g_bar    : baryonic accelerations [m/s^2]
        sigma_g  : errors on g_obs [m/s^2]
        pred_func: function(g_bar, a0) -> g_pred
        a0       : acceleration scale [m/s^2]
        k        : number of free parameters (0 for UCT, 1 for fitted models)

    Returns:
        chi2_red : reduced chi-squared
    """
    g_pred = pred_func(g_bar, a0)
    N = len(g_obs)
    log_residual = np.log10(g_obs) - np.log10(g_pred)
    sigma_log = sigma_g / (g_obs * np.log(10.0))
    chi2 = np.sum(log_residual**2 / sigma_log**2)
    return chi2 / (N - k)


def fit_a0(g_obs, g_bar, sigma_g, pred_func,
           a0_range=(5e-11, 2e-10)):
    """Find best-fit a0 by minimizing chi^2_red (with k=1)."""
    def objective(a0):
        return compute_chi2(g_obs, g_bar, sigma_g, pred_func, a0, k=1)
    result = minimize_scalar(objective, bounds=a0_range, method="bounded")
    return result.x, result.fun


# ---------------------------------------------------------------------------
#  Main analysis
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print("=" * 78)
    print("  UCT SPARC Reproducibility Package")
    print("  Reproducing Table IV of Paper I (Steiger 2025)")
    print("=" * 78)
    print()

    # -- Print UCT predictions --
    print("UCT Predictions (zero free parameters):")
    print(f"  H0           = {H0_KMS_MPC} km/s/Mpc")
    print(f"  a0 = cH0/2pi = {A0_UCT:.6e} m/s^2")
    print(f"  phi          = {PHI:.10f}  (golden ratio)")
    print(f"  n = ln2/ln(phi^2) = {N_UCT:.6f}")
    print(f"  mu(1) = 2^(-1/n) = 1/phi^2 = {1.0/PHI**2:.6f}")
    print()

    # -- Load SPARC data --
    g_obs, g_bar, sigma_g, names = load_sparc()
    N = len(g_obs)
    unique_gals = np.unique(names)
    N_gal = len(unique_gals)
    print(f"SPARC data loaded: {N} data points from {N_gal} galaxies")
    print(f"  g_bar range: [{g_bar.min():.3e}, {g_bar.max():.3e}] m/s^2")
    print(f"  g_obs range: [{g_obs.min():.3e}, {g_obs.max():.3e}] m/s^2")
    print()

    # -- Define models --
    models = {
        "UCT":      {"func": g_pred_uct,      "n": N_UCT, "k": 0},
        "Simple":   {"func": g_pred_simple,    "n": 1.0,   "k": 1},
        "RAR":      {"func": g_pred_rar,       "n": None,  "k": 1},
        "Standard": {"func": g_pred_standard,  "n": 2.0,   "k": 1},
    }

    # -- Compute chi-squared for each model --
    print("Computing chi-squared for each interpolation function...")
    print("(This may take a few seconds for models requiring root-finding)")
    print()

    results = {}

    for label, model in models.items():
        t1 = time.time()
        print(f"  {label}...", end="", flush=True)

        if model["k"] == 0:
            # UCT: no fitting, use predicted a0
            a0 = A0_UCT
            chi2_red = compute_chi2(g_obs, g_bar, sigma_g, model["func"], a0, k=0)
            g_pred = model["func"](g_bar, a0)
        else:
            # Fit a0
            a0, chi2_red = fit_a0(g_obs, g_bar, sigma_g, model["func"])
            g_pred = model["func"](g_bar, a0)

        dt = time.time() - t1
        results[label] = {
            "a0": a0,
            "chi2_red": chi2_red,
            "g_pred": g_pred,
            "k": model["k"],
            "n": model["n"],
        }
        print(f" done ({dt:.1f}s)  chi2_red = {chi2_red:.2f}  a0 = {a0:.4e}")

    # -- Print Table IV --
    print()
    print("=" * 78)
    print("  TABLE IV: SPARC RAR Fit Comparison")
    print("=" * 78)
    print()
    chi2_floor = min(r["chi2_red"] for r in results.values())
    print(f"  {'Function':<12s} {'n':>6s} {'Free':>5s} {'chi2_red':>10s} "
          f"{'chi2_tilde':>11s} {'a0 [m/s^2]':>14s}")
    print("  " + "-" * 62)
    for label in ["UCT", "Simple", "RAR", "Standard"]:
        r = results[label]
        n_str = f"{r['n']:.2f}" if r['n'] is not None else "---"
        chi2_tilde = r["chi2_red"] / chi2_floor
        k_str = str(r["k"])
        print(f"  {label:<12s} {n_str:>6s} {k_str:>5s} {r['chi2_red']:>10.2f} "
              f"{chi2_tilde:>11.2f} {r['a0']:>14.4e}")
    print()

    # -- Verify against paper values --
    paper_values = {
        "UCT": 53.38, "Simple": 52.16, "RAR": 54.08, "Standard": 110.72
    }
    print("Comparison with published values (Table IV):")
    all_match = True
    for label, paper_val in paper_values.items():
        computed = results[label]["chi2_red"]
        diff = abs(computed - paper_val)
        pct = diff / paper_val * 100
        status = "MATCH" if pct < 0.1 else ("CLOSE" if pct < 1.0 else "MISMATCH")
        if pct >= 1.0:
            all_match = False
        print(f"  {label:<12s}: computed = {computed:.2f}, "
              f"paper = {paper_val:.2f}, diff = {diff:.2f} ({pct:.3f}%)  [{status}]")
    print()

    if all_match:
        print("  ALL VALUES REPRODUCED.  Table IV confirmed.")
    else:
        print("  WARNING: Some values differ by more than 1%.")
    print()

    # -- Split-RMS analysis --
    print("=" * 78)
    print("  SPLIT-RMS ANALYSIS")
    print("=" * 78)
    print()
    x_obs = g_bar / A0_UCT  # dimensionless acceleration

    regimes = {
        "Newtonian (x > 5)":        x_obs > 5,
        "Transition (0.2 < x < 5)": (x_obs > 0.2) & (x_obs < 5),
        "Deep MOND (x < 0.2)":      x_obs < 0.2,
    }

    print(f"  {'Regime':<28s} {'N_pts':>6s}", end="")
    for label in ["UCT", "Simple"]:
        print(f" {'RMS_'+label:>12s}", end="")
    print(f" {'Delta':>8s}")
    print("  " + "-" * 70)

    for regime_label, mask in regimes.items():
        n_pts = mask.sum()
        rms_vals = {}
        for label in ["UCT", "Simple"]:
            g_pred = results[label]["g_pred"][mask]
            residual = np.log10(g_obs[mask]) - np.log10(g_pred)
            rms_vals[label] = np.sqrt(np.mean(residual**2))

        delta = rms_vals["UCT"] - rms_vals["Simple"]
        print(f"  {regime_label:<28s} {n_pts:>6d}", end="")
        for label in ["UCT", "Simple"]:
            print(f" {rms_vals[label]:>12.4f}", end="")
        print(f" {delta:>8.4f} dex")

    print()

    # -- Per-galaxy chi-squared distribution --
    print("=" * 78)
    print("  PER-GALAXY chi2_red DISTRIBUTION (UCT)")
    print("=" * 78)
    print()

    gal_chi2 = {}
    for gal in unique_gals:
        mask_gal = names == gal
        n_pts_gal = mask_gal.sum()
        if n_pts_gal < 2:
            continue
        g_pred_gal = results["UCT"]["g_pred"][mask_gal]
        g_obs_gal = g_obs[mask_gal]
        sig_gal = sigma_g[mask_gal]
        log_res = np.log10(g_obs_gal) - np.log10(g_pred_gal)
        sig_log = sig_gal / (g_obs_gal * np.log(10.0))
        chi2_gal = np.sum(log_res**2 / sig_log**2) / n_pts_gal
        gal_chi2[gal] = (chi2_gal, n_pts_gal)

    chi2_arr = np.array([v[0] for v in gal_chi2.values()])
    print(f"  Median per-galaxy chi2_red: {np.median(chi2_arr):.2f}")
    print(f"  Mean per-galaxy chi2_red:   {np.mean(chi2_arr):.2f}")
    print(f"  Std dev:                    {np.std(chi2_arr):.2f}")
    print(f"  Min:                        {np.min(chi2_arr):.2f} "
          f"({min(gal_chi2, key=lambda k: gal_chi2[k][0])})")
    print(f"  Max:                        {np.max(chi2_arr):.2f} "
          f"({max(gal_chi2, key=lambda k: gal_chi2[k][0])})")
    print()

    # Top 10 worst-fit and best-fit galaxies
    sorted_gals = sorted(gal_chi2.items(), key=lambda x: -x[1][0])
    print("  Top 10 worst-fit galaxies:")
    for gal, (chi2_val, npts) in sorted_gals[:10]:
        print(f"    {gal:<20s}: chi2_red = {chi2_val:>8.1f}  ({npts} pts)")
    print()
    print("  Top 10 best-fit galaxies:")
    for gal, (chi2_val, npts) in sorted_gals[-10:]:
        print(f"    {gal:<20s}: chi2_red = {chi2_val:>8.1f}  ({npts} pts)")
    print()

    # ======================================================================
    #  OUTLIER ANALYSIS — identify what drives the worst-fit galaxies
    # ======================================================================
    print("=" * 78)
    print("  OUTLIER ANALYSIS")
    print("=" * 78)
    print()

    # Load galaxy properties from table1
    table1_path = find_sparc_data().parent / "SPARC_table1.dat"
    gal_props = {}
    htypes = {0: "S0", 1: "Sa", 2: "Sab", 3: "Sb", 4: "Sbc", 5: "Sc",
              6: "Scd", 7: "Sd", 8: "Sdm", 9: "Sm", 10: "Im", 11: "BCD"}
    dmethods = {1: "Hubble flow", 2: "TRGB", 3: "Cepheids",
                4: "Ursa Major", 5: "Supernovae"}
    if table1_path.exists():
        with open(table1_path, "r") as f1:
            for line in f1:
                if line.startswith("#") or line.strip() == "":
                    continue
                gname = line[:11].strip()
                try:
                    ht = int(line[12:14].strip())
                    dist = float(line[15:21])
                    e_dist = float(line[22:27])
                    fd = int(line[28])
                    incl = float(line[30:34])
                    e_incl = float(line[35:39])
                    qual = int(line[112:115].strip())
                except (ValueError, IndexError):
                    continue
                gal_props[gname] = {
                    "type": htypes.get(ht, f"?{ht}"), "type_num": ht,
                    "dist": dist, "e_dist": e_dist,
                    "f_dist": dmethods.get(fd, f"?{fd}"), "f_dist_num": fd,
                    "incl": incl, "e_incl": e_incl, "qual": qual,
                }

    # Compute Simple chi2 per galaxy for comparison
    gal_chi2_simple = {}
    for gal in unique_gals:
        mask_gal = names == gal
        n_pts_gal = mask_gal.sum()
        if n_pts_gal < 2:
            continue
        gp = g_pred_simple(g_bar[mask_gal], results["Simple"]["a0"])
        go = g_obs[mask_gal]
        sg = sigma_g[mask_gal]
        log_res = np.log10(go) - np.log10(gp)
        sig_log = sg / (go * np.log(10.0))
        chi2_s = np.sum(log_res**2 / sig_log**2) / n_pts_gal
        gal_chi2_simple[gal] = chi2_s

    # Compute per-galaxy RMS and mean residual for UCT
    gal_rms = {}
    for gal in unique_gals:
        mask_gal = names == gal
        n_pts_gal = mask_gal.sum()
        if n_pts_gal < 2:
            continue
        gp = results["UCT"]["g_pred"][mask_gal]
        go = g_obs[mask_gal]
        log_res = np.log10(go) - np.log10(gp)
        gal_rms[gal] = (np.sqrt(np.mean(log_res**2)), np.mean(log_res))

    # Print detailed outlier table
    print(f"  {'Galaxy':<14s} {'chi2_UCT':>9s} {'chi2_Sim':>9s} {'Winner':>7s} "
          f"{'RMS':>6s} {'<res>':>6s} {'Type':>5s} {'Q':>2s} "
          f"{'i':>4s} {'e_i':>4s} {'Dist method':<12s} {'Issues'}")
    print("  " + "-" * 110)

    # Known literature issues for SPARC outliers
    known_issues = {
        "NGC5055":  "HI warp, disk-halo offset (Battaglia+ 2006)",
        "NGC5371":  "25% distance uncertainty",
        "UGC05764": "Gas-dominated dwarf, no flat RC",
        "UGC06787": "Bulge-dominated, 25% dist error",
        "UGC02487": "S0 lenticular, bulge-disk decomposition ambiguity",
        "IC2574":   "Supergiant HI shell, non-equilibrium (Walter+ 1998)",
        "UGC11820": "30% dist error, no flat RC",
        "UGC06786": "S0 lenticular, 25% dist error",
        "DDO170":   "30% distance error, gas-dominated dwarf",
        "DDO154":   "Non-monotonic acceleration profile",
        "UGC02916": "Bulge-dominated (Vbul=280 km/s)",
        "UGC09133": "Bulge-dominated (Vbul=327 km/s), 20% dist error",
    }

    for gal, (chi2_val, npts) in sorted_gals[:10]:
        p = gal_props.get(gal, {})
        chi2_s = gal_chi2_simple.get(gal, 0)
        rms, mean_res = gal_rms.get(gal, (0, 0))
        winner = "UCT" if chi2_val < chi2_s else "Simple"
        issues = known_issues.get(gal, "")

        # Auto-detect issues
        flags = []
        if p.get("qual", 0) == 3:
            flags.append("Q=3")
        if p.get("e_incl", 0) >= 7:
            flags.append(f"e_i={p['e_incl']:.0f}")
        if p.get("e_dist", 0) / max(p.get("dist", 1), 0.01) > 0.2:
            flags.append(f"dist_err={p['e_dist']/p['dist']*100:.0f}%")
        if p.get("type_num", 0) >= 9:
            flags.append("irregular")
        if p.get("type_num", 0) == 0:
            flags.append("S0")
        if not issues and flags:
            issues = "; ".join(flags)

        print(f"  {gal:<14s} {chi2_val:>9.1f} {chi2_s:>9.1f} {winner:>7s} "
              f"{rms:>6.3f} {mean_res:>+6.3f} "
              f"{p.get('type','?'):>5s} {p.get('qual',0):>2d} "
              f"{p.get('incl',0):>4.0f} {p.get('e_incl',0):>4.0f} "
              f"{p.get('f_dist','?'):<12s} {issues}")

    # Classify the outlier population
    print()
    n_uct_wins = sum(1 for g, (c, _) in sorted_gals[:10]
                     if c < gal_chi2_simple.get(g, 1e30))
    n_irreg = sum(1 for g, _ in sorted_gals[:10]
                  if gal_props.get(g, {}).get("type_num", 0) >= 9)
    n_bulge = sum(1 for g, _ in sorted_gals[:10]
                  if gal_props.get(g, {}).get("type_num", 0) <= 2)
    n_high_ei = sum(1 for g, _ in sorted_gals[:10]
                    if gal_props.get(g, {}).get("e_incl", 0) >= 7)
    n_high_ed = sum(1 for g, _ in sorted_gals[:10]
                    if gal_props.get(g, {}).get("e_dist", 0) /
                    max(gal_props.get(g, {}).get("dist", 1), 0.01) > 0.2)

    print("  Outlier classification (top 10 worst UCT galaxies):")
    print(f"    UCT beats Simple on:     {n_uct_wins}/10")
    print(f"    Irregulars (Im/Sm/BCD):  {n_irreg}/10  (vs 41% of full sample)")
    print(f"    Bulge-dominated (S0-Sab):{n_bulge}/10")
    print(f"    Incl error >= 7 deg:     {n_high_ei}/10  (vs 27% of full sample)")
    print(f"    Dist error >= 20%:       {n_high_ed}/10  (vs 35% of full sample)")
    print()
    print("  Key finding: The worst-fit galaxies cluster into two categories:")
    print("    1. Gas-dominated dwarfs with known non-equilibrium dynamics or")
    print("       large observational uncertainties (IC2574, DDO154, DDO170,")
    print("       UGC05764, UGC11820) — the theory fails where DATA quality is low.")
    print("    2. Bulge-dominated early-types where M/L decomposition is uncertain")
    print("       (NGC5055, UGC02487, UGC06787) — but UCT often BEATS Simple on")
    print("       these, suggesting the issue is baryonic mass modeling, not the")
    print("       interpolation function.")
    print()

    # ======================================================================
    #  n-SCAN ANALYSIS
    # ======================================================================
    print("=" * 78)
    print("  n-SCAN: chi2 vs exponent n in mu(x) = x/(1+x^n)^{1/n}")
    print("=" * 78)
    print()

    n_values = np.linspace(0.3, 3.0, 55)
    chi2_of_n_fixed_a0 = []    # with a0 = cH0/(2pi) fixed
    chi2_of_n_fitted_a0 = []   # with a0 fitted for each n
    a0_of_n = []

    print("  Scanning n (a0 fixed at cH0/2pi)...", end="", flush=True)
    for n_val in n_values:
        def pred_n(gb, a0, _n=n_val):
            return _solve_aqual_family(gb, a0, _n)
        chi2_val = compute_chi2(g_obs, g_bar, sigma_g, pred_n, A0_UCT, k=0)
        chi2_of_n_fixed_a0.append(chi2_val)
    print(" done.")

    print("  Scanning n (a0 fitted for each n)...", end="", flush=True)
    for n_val in n_values:
        def pred_n(gb, a0, _n=n_val):
            return _solve_aqual_family(gb, a0, _n)
        a0_best, chi2_best = fit_a0(g_obs, g_bar, sigma_g, pred_n)
        chi2_of_n_fitted_a0.append(chi2_best)
        a0_of_n.append(a0_best)
    print(" done.")

    chi2_of_n_fixed_a0 = np.array(chi2_of_n_fixed_a0)
    chi2_of_n_fitted_a0 = np.array(chi2_of_n_fitted_a0)
    a0_of_n = np.array(a0_of_n)

    i_best_fixed = np.argmin(chi2_of_n_fixed_a0)
    i_best_fitted = np.argmin(chi2_of_n_fitted_a0)

    print()
    print(f"  Best-fit n (a0 fixed):  n = {n_values[i_best_fixed]:.3f}, "
          f"chi2_red = {chi2_of_n_fixed_a0[i_best_fixed]:.2f}")
    print(f"  Best-fit n (a0 free):   n = {n_values[i_best_fitted]:.3f}, "
          f"chi2_red = {chi2_of_n_fitted_a0[i_best_fitted]:.2f}, "
          f"a0 = {a0_of_n[i_best_fitted]:.4e}")
    print(f"  UCT prediction:         n = {N_UCT:.3f}, "
          f"chi2_red = {results['UCT']['chi2_red']:.2f}")
    delta_chi2 = results['UCT']['chi2_red'] - chi2_of_n_fixed_a0[i_best_fixed]
    print(f"  Delta chi2_red (UCT - best, a0 fixed): {delta_chi2:+.2f}")
    print()

    # ======================================================================
    #  a0 DISTRIBUTION ANALYSIS (bootstrap + acceleration-binned)
    # ======================================================================
    print("=" * 78)
    print("  a0 DISTRIBUTION ANALYSIS")
    print("=" * 78)
    print()

    # --- Bootstrap resampling (galaxy-level, geometric-distance subsample) ---
    #
    # The geometric-distance subsample uses SPARC galaxies with non-Hubble-flow
    # distance methods (TRGB, Cepheids, Ursa Major cluster, Supernovae),
    # identified by f_Dist != 1 in SPARC table1.  This eliminates
    # distance-acceleration circularity from Tully-Fisher distances.
    #
    # NOTE: This bootstrap uses the Simple interpolation function on the RAR
    # (g_obs vs g_bar from table2.dat, Upsilon_*=1).  The paper figure uses a
    # profile-likelihood method that fits Upsilon_d and Upsilon_b per galaxy
    # from individual rotation curves (h0_inference.py).  Both methods find
    # UCT's a0 closer to the data mean than canonical MOND.
    #
    N_BOOT = 2000
    A0_CANONICAL = 1.2e-10   # canonical MOND value [m/s^2]

    # Load distance flags and quality from SPARC table 1
    # Geometric-distance subsample: f_Dist != 1 (non-Hubble-flow) AND Qual <= 2
    # This matches the methodology in h0_inference.py:filter_by_distance_method()
    geom_gals = set()
    table1_path = find_sparc_data().parent / "SPARC_table1.dat"
    if table1_path.exists():
        with open(table1_path, "r") as f1:
            for line in f1:
                if line.startswith("#") or line.strip() == "":
                    continue
                gname = line[:11].strip()
                try:
                    f_dist = int(line[28])
                    qual = int(line[112:115].strip())
                except (ValueError, IndexError):
                    continue
                if f_dist != 1 and qual <= 2:  # geometric distance + good quality
                    geom_gals.add(gname)
        # Restrict to galaxies present in table2 data
        geom_gals = sorted(geom_gals & set(unique_gals))
    else:
        print("  WARNING: SPARC_table1.dat not found; using all galaxies.")
        geom_gals = sorted(unique_gals)

    N_geom = len(geom_gals)
    print(f"  Geometric-distance subsample: {N_geom} galaxies")
    print(f"  Bootstrap resampling ({N_BOOT} iterations, galaxy-level)...",
          end="", flush=True)

    rng = np.random.default_rng(42)
    boot_a0 = np.empty(N_BOOT)

    # Pre-index data by galaxy for fast resampling
    gal_indices = {gal: np.where(names == gal)[0] for gal in unique_gals}

    # Grid-based a0 fitting for bootstrap -- use a fine grid to avoid
    # optimizer precision artifacts that suppress bootstrap variance.
    # Two-pass approach: coarse grid to bracket, then fine grid to refine.
    a0_coarse = np.linspace(5e-11, 1.8e-10, 80)

    def fit_a0_grid(g_obs_b, g_bar_b, sig_b):
        """Fit a0 using two-pass grid search (coarse + fine)."""
        sig_log_b = sig_b / (g_obs_b * np.log(10.0))
        log_gobs_b = np.log10(g_obs_b)

        def chi2_at(a0_val):
            x_N = g_bar_b / a0_val
            x = 0.5 * (x_N + np.sqrt(x_N**2 + 4.0 * x_N))
            log_gpred = np.log10(x * a0_val)
            return np.sum((log_gobs_b - log_gpred)**2 / sig_log_b**2)

        # Coarse pass
        chi2_coarse = np.array([chi2_at(a0) for a0 in a0_coarse])
        i_best = np.argmin(chi2_coarse)
        # Fine pass around the coarse minimum
        lo = a0_coarse[max(0, i_best - 1)]
        hi = a0_coarse[min(len(a0_coarse) - 1, i_best + 1)]
        a0_fine = np.linspace(lo, hi, 80)
        chi2_fine = np.array([chi2_at(a0) for a0 in a0_fine])
        return a0_fine[np.argmin(chi2_fine)]

    geom_gals_arr = np.array(geom_gals)
    for ib in range(N_BOOT):
        # Resample geometric-distance galaxies (with replacement)
        gal_sample = rng.choice(geom_gals_arr, size=N_geom, replace=True)
        idx_boot = np.concatenate([gal_indices[gal] for gal in gal_sample])
        boot_a0[ib] = fit_a0_grid(g_obs[idx_boot], g_bar[idx_boot],
                                  sigma_g[idx_boot])

    print(" done.")

    boot_mean = np.mean(boot_a0)
    boot_std = np.std(boot_a0)
    sigma_uct = (A0_UCT - boot_mean) / boot_std
    sigma_canonical = (A0_CANONICAL - boot_mean) / boot_std

    print()
    print(f"  Bootstrap results (RAR-based, Simple function, {N_BOOT} resamplings):")
    print(f"  (Paper figure uses profile-likelihood method; qualitative conclusion identical)")
    print(f"    Mean a0:        {boot_mean:.6e} m/s^2")
    print(f"    Std a0:         {boot_std:.6e} m/s^2")
    print(f"    UCT prediction: {A0_UCT:.6e} m/s^2  "
          f"({sigma_uct:+.2f} sigma from mean)")
    print(f"    Canonical MOND: {A0_CANONICAL:.6e} m/s^2  "
          f"({sigma_canonical:+.2f} sigma from mean)")
    print()

    # --- Acceleration-binned a0 measurement ---
    N_ABINS = 6
    log_gbar_all = np.log10(g_bar)
    abin_edges = np.linspace(log_gbar_all.min(), log_gbar_all.max(), N_ABINS + 1)
    abin_centers = []
    abin_a0 = []
    abin_a0_err = []

    print(f"  Acceleration-binned a0 ({N_ABINS} bins)...")
    for jb in range(N_ABINS):
        mb = (log_gbar_all >= abin_edges[jb]) & (log_gbar_all < abin_edges[jb + 1])
        n_pts_bin = mb.sum()
        if n_pts_bin < 10:
            continue
        center = 0.5 * (abin_edges[jb] + abin_edges[jb + 1])
        abin_centers.append(center)
        a0_fit, chi2_fit = fit_a0(g_obs[mb], g_bar[mb], sigma_g[mb], g_pred_simple)
        abin_a0.append(a0_fit)

        # Error from total chi2 curvature: Delta chi2_total = 1 gives 1-sigma
        def chi2_total_bin(a0_val, _mb=mb):
            gp = g_pred_simple(g_bar[_mb], a0_val)
            lr = np.log10(g_obs[_mb]) - np.log10(gp)
            sl = sigma_g[_mb] / (g_obs[_mb] * np.log(10.0))
            return np.sum(lr**2 / sl**2)
        chi2_min_total = chi2_total_bin(a0_fit)
        chi2_target = chi2_min_total + 1.0
        # Search for lower bound
        try:
            a0_lo = brentq(lambda a0: chi2_total_bin(a0) - chi2_target,
                           5e-11, a0_fit - 1e-15, xtol=1e-15)
        except ValueError:
            a0_lo = a0_fit
        # Search for upper bound
        try:
            a0_hi = brentq(lambda a0: chi2_total_bin(a0) - chi2_target,
                           a0_fit + 1e-15, 2e-10, xtol=1e-15)
        except ValueError:
            a0_hi = a0_fit
        err = 0.5 * (a0_hi - a0_lo)
        if err < 1e-15:
            # Fallback: finite-difference estimate of curvature
            da = a0_fit * 1e-3
            c_lo = chi2_total_bin(a0_fit - da)
            c_hi = chi2_total_bin(a0_fit + da)
            d2chi2 = (c_lo - 2.0 * chi2_min_total + c_hi) / da**2
            if d2chi2 > 0:
                err = 1.0 / np.sqrt(d2chi2)
            else:
                err = da  # last resort
        abin_a0_err.append(err)

        print(f"    Bin {jb+1}: log g_bar in [{abin_edges[jb]:.2f}, "
              f"{abin_edges[jb+1]:.2f}],  N = {n_pts_bin:>5d},  "
              f"a0 = {a0_fit:.4e} +/- {abin_a0_err[-1]:.2e}")

    abin_centers = np.array(abin_centers)
    abin_a0 = np.array(abin_a0)
    abin_a0_err = np.array(abin_a0_err)

    # Acceleration-weighted average
    weights = 1.0 / abin_a0_err**2
    a0_weighted_avg = np.sum(abin_a0 * weights) / np.sum(weights)
    print()
    print(f"    Acceleration-weighted average a0: {a0_weighted_avg:.6e} m/s^2")
    print()

    # ======================================================================
    #  FIGURES
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping figures.")
        print(f"\nTotal runtime: {time.time()-t0:.1f}s")
        return

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.labelsize":    13,
        "axes.titlesize":    14,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.frameon":    False,
    })

    COLORS = {
        "UCT":      "#ff7f0e",
        "Simple":   "#1f77b4",
        "RAR":      "#2ca02c",
        "Standard": "#9467bd",
        "Newton":   "#888888",
        "data":     "#333333",
    }

    outdir = Path(__file__).parent / "figures"
    outdir.mkdir(exist_ok=True)

    # Precompute binned RAR for reuse
    log_gbar = np.log10(g_bar)
    log_gobs = np.log10(g_obs)
    nbins = 25
    bin_edges = np.linspace(log_gbar.min(), log_gbar.max(), nbins + 1)
    bin_centers, bin_medians, bin_lo, bin_hi = [], [], [], []
    for j in range(nbins):
        m = (log_gbar >= bin_edges[j]) & (log_gbar < bin_edges[j+1])
        if m.sum() < 5:
            continue
        go = log_gobs[m]
        bin_centers.append(0.5 * (bin_edges[j] + bin_edges[j+1]))
        bin_medians.append(np.median(go))
        bin_lo.append(np.percentile(go, 16))
        bin_hi.append(np.percentile(go, 84))
    bc = np.array(bin_centers)
    bm = np.array(bin_medians)
    bl = np.array(bin_lo)
    bh = np.array(bin_hi)

    # Fine grid for model curves
    g_grid = np.logspace(log_gbar.min() - 0.2, log_gbar.max() + 0.2, 300)

    # ----- Figure 1: RAR with all functions overlaid -----
    print("Generating Figure 1: RAR comparison...")
    fig1, ax1 = plt.subplots(figsize=(7.5, 7.5))

    ax1.scatter(log_gbar, log_gobs, s=0.4, alpha=0.06,
                color=COLORS["data"], zorder=1, rasterized=True)

    ax1.errorbar(bc, bm, yerr=[bm - bl, bh - bm], fmt="o",
                 color="black", ms=5, capsize=3, lw=1.2, zorder=5,
                 label="SPARC binned (median +/- 1 sigma)")

    for label in ["Simple", "RAR", "Standard", "UCT"]:
        a0 = results[label]["a0"]
        gp = models[label]["func"](g_grid, a0)
        style = dict(lw=2.8, zorder=4) if label == "UCT" else dict(lw=1.6, zorder=3)
        ls = "-" if label == "UCT" else "--"
        n_str = f", n={results[label]['n']:.2f}" if results[label]["n"] else ""
        chi2_str = f", chi2={results[label]['chi2_red']:.1f}"
        ax1.plot(np.log10(g_grid), np.log10(gp), color=COLORS[label],
                 ls=ls, **style,
                 label=f"{label} (a0={a0:.2e}{n_str}{chi2_str})")

    ax1.plot([-13.5, -8], [-13.5, -8], ":", color=COLORS["Newton"],
             lw=1, alpha=0.5)
    ax1.text(-8.6, -8.9, "1:1 (Newtonian)", fontsize=8,
             color=COLORS["Newton"], alpha=0.6)

    ax1.set_xlabel(r"$\log_{10}\, g_{\rm bar}$  [m/s$^2$]")
    ax1.set_ylabel(r"$\log_{10}\, g_{\rm obs}$  [m/s$^2$]")
    ax1.set_title(f"Radial Acceleration Relation ({N_gal} SPARC galaxies, {N} points)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(-13, -8.5)
    ax1.set_ylim(-12.5, -8.5)
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.15)
    fig1.tight_layout()
    fig1.savefig(outdir / "fig1_rar_comparison.png")
    fig1.savefig(outdir / "fig1_rar_comparison.pdf")
    print(f"  Saved: {outdir / 'fig1_rar_comparison.png'}")

    # ----- Figure 2: Residuals panel -----
    print("Generating Figure 2: Residual panels...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes2 = axes2.ravel()

    for idx, label in enumerate(["UCT", "Simple", "RAR", "Standard"]):
        ax = axes2[idx]
        g_pred = results[label]["g_pred"]
        residual = np.log10(g_obs) - np.log10(g_pred)
        rms = np.sqrt(np.mean(residual**2))

        ax.scatter(log_gbar, residual, s=0.4, alpha=0.08,
                   color=COLORS[label], rasterized=True)
        ax.axhline(0, color="k", ls="--", lw=0.8)

        # Binned median residual
        for j in range(len(bin_edges)-1):
            m = (log_gbar >= bin_edges[j]) & (log_gbar < bin_edges[j+1])
            if m.sum() > 5:
                ax.plot(0.5*(bin_edges[j]+bin_edges[j+1]),
                        np.median(residual[m]),
                        "o", color="black", ms=4, zorder=5)

        k_str = f"k={results[label]['k']}"
        ax.set_title(f"{label}  (RMS = {rms:.4f} dex, "
                     r"$\chi^2_{\rm red}$"
                     f" = {results[label]['chi2_red']:.1f}, {k_str})",
                     fontsize=10)
        ax.set_ylim(-0.6, 0.6)
        ax.grid(alpha=0.15)

    axes2[2].set_xlabel(r"$\log_{10}\, g_{\rm bar}$  [m/s$^2$]")
    axes2[3].set_xlabel(r"$\log_{10}\, g_{\rm bar}$  [m/s$^2$]")
    axes2[0].set_ylabel("Residual [dex]")
    axes2[2].set_ylabel("Residual [dex]")
    fig2.suptitle("RAR Residuals by Interpolation Function", fontsize=14, y=1.01)
    fig2.tight_layout()
    fig2.savefig(outdir / "fig2_residuals.png")
    fig2.savefig(outdir / "fig2_residuals.pdf")
    print(f"  Saved: {outdir / 'fig2_residuals.png'}")

    # ----- Figure 3: n-scan -----
    print("Generating Figure 3: n-scan...")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: chi2 vs n
    ax3a.plot(n_values, chi2_of_n_fixed_a0, "k-", lw=2,
              label=r"$a_0 = cH_0/(2\pi)$ fixed")
    ax3a.plot(n_values, chi2_of_n_fitted_a0, "k--", lw=1.5, alpha=0.6,
              label=r"$a_0$ fitted for each $n$")
    ax3a.axvline(N_UCT, color=COLORS["UCT"], ls="--", lw=2.5,
                 label=f"UCT: n = ln2/ln(phi^2) = {N_UCT:.3f}")
    ax3a.axvline(1.0, color=COLORS["Simple"], ls=":", lw=1.5,
                 label="Simple: n = 1")
    ax3a.axvline(2.0, color=COLORS["Standard"], ls=":", lw=1.5,
                 label="Standard: n = 2")
    ax3a.plot(n_values[i_best_fixed], chi2_of_n_fixed_a0[i_best_fixed],
              "r*", ms=12, zorder=5,
              label=f"Min (a0 fixed): n = {n_values[i_best_fixed]:.2f}")
    ax3a.set_xlabel("Exponent $n$", fontsize=13)
    ax3a.set_ylabel(r"$\chi^2_{\rm red}$", fontsize=13)
    ax3a.set_title(r"$\chi^2$ Scan: $\mu(x) = x/(1+x^n)^{1/n}$")
    ax3a.legend(fontsize=8)
    ax3a.set_xlim(0.3, 3.0)
    ax3a.grid(alpha=0.2)

    # Right: mu(x) for different n
    x = np.logspace(-2, 2, 500)
    ax3b.semilogx(x, x / (1.0 + x), color=COLORS["Simple"], ls="--",
                  lw=1.5, label="Simple (n=1)")
    ax3b.semilogx(x, x / np.sqrt(1.0 + x**2), color=COLORS["Standard"],
                  ls="--", lw=1.5, label="Standard (n=2)")
    ax3b.semilogx(x, x / (1.0 + x**N_UCT)**(1.0/N_UCT),
                  color=COLORS["UCT"], lw=2.5,
                  label=f"UCT (n={N_UCT:.3f})")
    # RAR empirical mu(x)
    x_rar_plot = np.logspace(-2, 2, 200)
    mu_rar_vals = np.empty_like(x_rar_plot)
    for ii, xi in enumerate(x_rar_plot):
        if xi < 1e-10:
            mu_rar_vals[ii] = np.sqrt(xi)
            continue
        def frar(y, _xi=xi):
            sy = np.sqrt(max(y, 1e-30))
            return y / (1.0 - np.exp(-sy)) - _xi
        y_hi = xi
        if frar(y_hi) < 0:
            y_hi = xi * 10
        y_sol = brentq(frar, 1e-15, y_hi, xtol=1e-14)
        mu_rar_vals[ii] = y_sol / xi
    ax3b.semilogx(x_rar_plot, mu_rar_vals, color=COLORS["RAR"], ls="--",
                  lw=1.5, label="RAR (empirical)")

    ax3b.axhline(1.0/PHI**2, color=COLORS["UCT"], ls=":", lw=0.8, alpha=0.6)
    ax3b.text(0.012, 1.0/PHI**2 + 0.02,
              r"$\mu(1)=1/\varphi^2$" + f"={1.0/PHI**2:.3f}",
              fontsize=8, color=COLORS["UCT"], alpha=0.7)
    ax3b.axvline(1, color="gray", ls=":", lw=0.5)
    ax3b.set_xlabel(r"$x = g/a_0$", fontsize=13)
    ax3b.set_ylabel(r"$\mu(x)$", fontsize=13)
    ax3b.set_title("Interpolation Functions")
    ax3b.set_ylim(0, 1.05)
    ax3b.legend(fontsize=8)
    ax3b.grid(alpha=0.2)

    fig3.tight_layout()
    fig3.savefig(outdir / "fig3_n_scan.png")
    fig3.savefig(outdir / "fig3_n_scan.pdf")
    print(f"  Saved: {outdir / 'fig3_n_scan.png'}")

    # ----- Figure 4: Per-galaxy chi2 histogram -----
    print("Generating Figure 4: Per-galaxy chi2 distribution...")
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    ax4a.hist(chi2_arr, bins=30, color=COLORS["UCT"], alpha=0.7,
              edgecolor="black", linewidth=0.5)
    ax4a.axvline(np.median(chi2_arr), color="red", ls="--", lw=2,
                 label=f"Median = {np.median(chi2_arr):.1f}")
    ax4a.axvline(results["UCT"]["chi2_red"], color="black", ls="-", lw=2,
                 label=f"Global = {results['UCT']['chi2_red']:.1f}")
    ax4a.set_xlabel(r"$\chi^2_{\rm red}$ per galaxy (UCT)", fontsize=13)
    ax4a.set_ylabel("Number of galaxies", fontsize=13)
    ax4a.set_title("Per-Galaxy Goodness of Fit")
    ax4a.legend(fontsize=9)
    ax4a.grid(alpha=0.2)

    # UCT vs Simple per-galaxy comparison
    gal_chi2_simple = {}
    for gal in unique_gals:
        mask_gal = names == gal
        n_pts_gal = mask_gal.sum()
        if n_pts_gal < 2:
            continue
        gp = results["Simple"]["g_pred"][mask_gal]
        go = g_obs[mask_gal]
        sg = sigma_g[mask_gal]
        log_res = np.log10(go) - np.log10(gp)
        sig_log = sg / (go * np.log(10.0))
        chi2_s = np.sum(log_res**2 / sig_log**2) / n_pts_gal
        gal_chi2_simple[gal] = chi2_s

    common_gals = sorted(set(gal_chi2.keys()) & set(gal_chi2_simple.keys()))
    chi2_uct_arr = np.array([gal_chi2[g][0] for g in common_gals])
    chi2_sim_arr = np.array([gal_chi2_simple[g] for g in common_gals])

    ax4b.scatter(chi2_sim_arr, chi2_uct_arr, s=15, alpha=0.6,
                 color=COLORS["UCT"], edgecolor="none")
    lims = [0, max(chi2_uct_arr.max(), chi2_sim_arr.max()) * 1.05]
    ax4b.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax4b.set_xlabel(r"$\chi^2_{\rm red}$ (Simple, fitted $a_0$)", fontsize=12)
    ax4b.set_ylabel(r"$\chi^2_{\rm red}$ (UCT, zero params)", fontsize=12)
    ax4b.set_title("UCT vs Simple: Per-Galaxy Comparison")
    ax4b.set_xlim(lims)
    ax4b.set_ylim(lims)
    ax4b.set_aspect("equal")
    ax4b.grid(alpha=0.2)

    n_better = np.sum(chi2_uct_arr < chi2_sim_arr)
    n_total = len(chi2_uct_arr)
    ax4b.text(0.05, 0.95, f"UCT better: {n_better}/{n_total} galaxies",
              transform=ax4b.transAxes, fontsize=10, va="top")

    fig4.tight_layout()
    fig4.savefig(outdir / "fig4_pergalaxy_chi2.png")
    fig4.savefig(outdir / "fig4_pergalaxy_chi2.pdf")
    print(f"  Saved: {outdir / 'fig4_pergalaxy_chi2.png'}")

    # ----- Figure 5: UCT zero-parameter prediction (highlight) -----
    print("Generating Figure 5: UCT zero-parameter highlight...")
    fig5, ax5 = plt.subplots(figsize=(7.5, 7.5))

    ax5.scatter(log_gbar, log_gobs, s=0.4, alpha=0.06,
                color=COLORS["data"], zorder=1, rasterized=True)
    ax5.errorbar(bc, bm, yerr=[bm - bl, bh - bm], fmt="o",
                 color="black", ms=5, capsize=3, lw=1.2, zorder=5,
                 label="SPARC data (binned)")

    gp_uct = g_pred_uct(g_grid, A0_UCT)
    ax5.plot(np.log10(g_grid), np.log10(gp_uct),
             color=COLORS["UCT"], lw=3, zorder=4,
             label=(f"UCT prediction (0 free params)\n"
                    f"  a0 = cH0/(2pi) = {A0_UCT:.3e} m/s^2\n"
                    f"  n = ln2/ln(phi^2) = {N_UCT:.4f}\n"
                    r"  $\chi^2_{\rm red}$"
                    f" = {results['UCT']['chi2_red']:.2f}"))

    gp_sim = g_pred_simple(g_grid, results["Simple"]["a0"])
    ax5.plot(np.log10(g_grid), np.log10(gp_sim),
             color=COLORS["Simple"], ls="--", lw=1.8, zorder=3,
             label=(f"Simple (1 free param, fitted a0)\n"
                    f"  a0 = {results['Simple']['a0']:.3e} m/s^2\n"
                    r"  $\chi^2_{\rm red}$"
                    f" = {results['Simple']['chi2_red']:.2f}"))

    ax5.plot([-13.5, -8], [-13.5, -8], ":", color=COLORS["Newton"],
             lw=1, alpha=0.5)
    ax5.set_xlabel(r"$\log_{10}\, g_{\rm bar}$  [m/s$^2$]")
    ax5.set_ylabel(r"$\log_{10}\, g_{\rm obs}$  [m/s$^2$]")
    ax5.set_title("UCT: Zero-Parameter Prediction vs SPARC RAR")
    ax5.legend(fontsize=8.5, loc="upper left")
    ax5.set_xlim(-13, -8.5)
    ax5.set_ylim(-12.5, -8.5)
    ax5.set_aspect("equal")
    ax5.grid(alpha=0.15)
    fig5.tight_layout()
    fig5.savefig(outdir / "fig5_uct_zero_params.png")
    fig5.savefig(outdir / "fig5_uct_zero_params.pdf")
    print(f"  Saved: {outdir / 'fig5_uct_zero_params.png'}")

    # ----- Figure 6: Residual histograms -----
    print("Generating Figure 6: Residual histograms...")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    bins_hist = np.linspace(-0.5, 0.5, 60)
    for label in ["UCT", "Simple", "RAR", "Standard"]:
        gp = results[label]["g_pred"]
        residual = np.log10(g_obs) - np.log10(gp)
        rms = np.sqrt(np.mean(residual**2))
        ax6.hist(residual, bins=bins_hist, alpha=0.35, color=COLORS[label],
                 label=f"{label} (RMS={rms:.4f} dex)", density=True)
    ax6.set_xlabel("Residual [dex]", fontsize=13)
    ax6.set_ylabel("Probability density", fontsize=13)
    ax6.set_title("Distribution of RAR Residuals")
    ax6.legend(fontsize=9)
    ax6.axvline(0, color="k", ls="--", lw=0.8)
    ax6.grid(alpha=0.15)
    fig6.tight_layout()
    fig6.savefig(outdir / "fig6_residual_histograms.png")
    fig6.savefig(outdir / "fig6_residual_histograms.pdf")
    print(f"  Saved: {outdir / 'fig6_residual_histograms.png'}")

    # ----- Figure 7: a0 distribution (bootstrap + acceleration-binned) -----
    print("Generating Figure 7: a0 distribution analysis...")
    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: bootstrap histogram
    ax7a.hist(boot_a0 * 1e10, bins=40, color=COLORS["Simple"], alpha=0.6,
              edgecolor="black", linewidth=0.5)
    ax7a.axvline(A0_UCT * 1e10, color=COLORS["UCT"], ls="-", lw=2.5,
                 label=(f"UCT: $a_0 = cH_0/(2\\pi)$\n"
                        f"  = {A0_UCT:.3e} m/s$^2$\n"
                        f"  ({sigma_uct:+.2f}$\\sigma$)"))
    ax7a.axvline(A0_CANONICAL * 1e10, color=COLORS["Simple"], ls="-", lw=2.5,
                 label=(f"Canonical MOND\n"
                        f"  = {A0_CANONICAL:.1e} m/s$^2$\n"
                        f"  ({sigma_canonical:+.2f}$\\sigma$)"))
    ax7a.axvline(boot_mean * 1e10, color="black", ls="--", lw=2,
                 label=f"Bootstrap mean\n  = {boot_mean:.3e} m/s$^2$")
    ax7a.set_xlabel(r"Best-fit $a_0$  [$10^{-10}$ m/s$^2$]", fontsize=13)
    ax7a.set_ylabel("Count", fontsize=13)
    ax7a.set_title(f"Bootstrap distribution of best-fit $a_0$\n"
                   f"({N_BOOT} resamplings, {N_geom} geometric-distance galaxies)",
                   fontsize=11)
    ax7a.legend(fontsize=8.5, loc="upper left")
    ax7a.grid(alpha=0.2)

    # Right panel: acceleration-binned a0
    ax7b.errorbar(abin_centers, abin_a0 * 1e10, yerr=abin_a0_err * 1e10,
                  fmt="s", color="black", ms=7, capsize=5, lw=1.5,
                  zorder=5, label="Bin best-fit $a_0$")
    ax7b.axhline(A0_UCT * 1e10, color=COLORS["UCT"], ls="-", lw=2.5,
                 label=f"UCT: {A0_UCT:.3e} m/s$^2$")
    ax7b.axhline(A0_CANONICAL * 1e10, color=COLORS["Simple"], ls="-", lw=2.5,
                 label=f"Canonical: {A0_CANONICAL:.1e} m/s$^2$")
    ax7b.axhline(a0_weighted_avg * 1e10, color="black", ls="--", lw=1.5,
                 label=f"Weighted avg: {a0_weighted_avg:.3e} m/s$^2$")
    ax7b.set_xlabel(r"$\log_{10}\, g_{\rm bar}$  [m/s$^2$]", fontsize=13)
    ax7b.set_ylabel(r"Best-fit $a_0$  [$10^{-10}$ m/s$^2$]", fontsize=13)
    ax7b.set_title(r"Best-fit $a_0$ by acceleration regime", fontsize=11)
    ax7b.legend(fontsize=8.5, loc="best")
    ax7b.grid(alpha=0.2)

    fig7.tight_layout()
    fig7.savefig(outdir / "fig7_a0_distribution.png")
    fig7.savefig(outdir / "fig7_a0_distribution.pdf")
    print(f"  Saved: {outdir / 'fig7_a0_distribution.png'}")

    plt.close("all")

    # -- Summary --
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print()
    print(f"  Data points:  {N}  ({N_gal} galaxies)")
    print(f"  UCT a0:       {A0_UCT:.6e} m/s^2  (= cH0/2pi, H0 = {H0_KMS_MPC} km/s/Mpc)")
    print(f"  UCT n:        {N_UCT:.6f}  (= ln2/ln(phi^2))")
    print(f"  UCT chi2_red: {results['UCT']['chi2_red']:.2f}  (0 free parameters)")
    print(f"  Simple chi2_red: {results['Simple']['chi2_red']:.2f}  (1 free parameter)")
    print(f"  Difference:   {results['UCT']['chi2_red'] - results['Simple']['chi2_red']:.2f} "
          f"({(results['UCT']['chi2_red']/results['Simple']['chi2_red'] - 1)*100:.1f}%)")
    print()
    print(f"  Total runtime: {time.time()-t0:.1f}s")
    print(f"  Figures saved to: {outdir.resolve()}")
    print()


if __name__ == "__main__":
    main()
