#!/usr/bin/env python3
"""
Individual Galaxy Rotation Curve Fits — UCT Paper (Steiger 2026)
=================================================================

Reproduces the observational test of Section V of the paper:

    Protocol
    --------
    (i)  Fix the RAR interpolation to the McGaugh et al. (2016) form.
    (ii) Fix a₀ to either the UCT value (cH₀/2π) or the MOND value.
    (iii) For each galaxy minimise χ² over (Υ_d, Υ_b) ∈ [0.05, 10].
    (iv) Compare paired reduced χ² values via Wilcoxon signed-rank test.

This tests the *normalization* of the RAR scale under a fixed empirical
interpolation, not Bayesian model selection or a first-principles
derivation of the interpolation.

SPARC data: Lelli et al. 2016, AJ 152 157.
V_disk and V_bulge in the rotmod files are normalised to Υ_* = 1;
scaling to an arbitrary Υ is V_component(Υ) = V_file × √Υ.

No synthetic data are used.

Author: Gabriel Steiger
Date:   February 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import wilcoxon
from scipy.linalg import inv as matrix_inv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Physical Constants
# =============================================================================

c         = 299792.458    # km/s
H0_PLANCK = 67.4          # km/s/Mpc  (Planck-2018)
H0_SHOES  = 73.0          # km/s/Mpc  (SH0ES)

# UCT a₀ = c·H₀/(2π):  c [km/s]×1e3→m/s, H₀ [km/s/Mpc]×1e3/3.086e22→s⁻¹
A0_UCT_PLANCK = c * H0_PLANCK * 1e6 / (2.0 * np.pi * 3.086e22)   # m/s²
A0_UCT_SHOES  = c * H0_SHOES  * 1e6 / (2.0 * np.pi * 3.086e22)   # m/s²
A0_MOND       = 1.2e-10                                            # m/s²

KPC_TO_M = 3.086e19   # m kpc⁻¹

# Υ_* bounds used in all fits (matching paper Section V B)
UPSILON_MIN = 0.05
UPSILON_MAX = 10.0

# =============================================================================
# SPARC Data Structures
# =============================================================================

@dataclass
class SPARCGalaxy:
    """
    Container for one SPARC rotation-curve file.

    V_disk and V_bulge are as tabulated in the rotmod file (Υ_* = 1).
    Scale to arbitrary Υ via:  V_component(Υ) = V_file × √Υ
    V_gas already includes the standard 1.33 helium correction.
    """
    name:        str
    distance:    float        # Mpc
    r:           np.ndarray   # kpc
    v_obs:       np.ndarray   # km/s
    v_err:       np.ndarray   # km/s
    v_gas:       np.ndarray   # km/s
    v_disk:      np.ndarray   # km/s  (Υ_* = 1 normalisation)
    v_bulge:     np.ndarray   # km/s  (Υ_* = 1 normalisation)

    def has_bulge(self) -> bool:
        return np.any(self.v_bulge > 0)


# =============================================================================
# Data I/O
# =============================================================================

def download_sparc_data(data_dir: str = "./sparc_data") -> Path:
    """Download and extract SPARC rotmod files if not present."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    rotmod_path = data_path / "Rotmod_LTG"
    if rotmod_path.exists():
        print(f"SPARC data already present at {data_path}")
        return data_path
    rotmod_url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
    print("Downloading SPARC rotation curve data…")
    try:
        zip_path = data_path / "Rotmod_LTG.zip"
        urllib.request.urlretrieve(rotmod_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_path)
        zip_path.unlink()
        print(f"Downloaded to {data_path}")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Obtain data from http://astroweb.cwru.edu/SPARC/")
    return data_path


def parse_sparc_galaxy(filepath: Path) -> SPARCGalaxy:
    """
    Parse one SPARC *_rotmod.dat file.

    Header: '# Distance = X Mpc'
    Data columns (≥6): R  Vobs  errV  Vgas  Vdisk  Vbulge  [SBdisk  SBbulge]
    """
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    name     = filepath.stem.replace('_rotmod', '')
    distance = 10.0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('# Distance'):
            try:
                distance = float(stripped.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
            break

    r_lst, vo_lst, ve_lst, vg_lst, vd_lst, vb_lst = [], [], [], [], [], []

    for line in lines:
        s = line.strip()
        if s.startswith('#') or not s:
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        try:
            r_lst.append(float(parts[0]))
            vo_lst.append(float(parts[1]))
            ve_lst.append(abs(float(parts[2])))
            vg_lst.append(float(parts[3]))
            vd_lst.append(float(parts[4]))
            vb_lst.append(float(parts[5]))
        except ValueError:
            continue

    if not r_lst:
        raise ValueError(f"No data rows parsed from {filepath}")

    r = np.array(r_lst)
    mask = (r > 0) & np.isfinite(r)
    return SPARCGalaxy(
        name=name,
        distance=distance,
        r=r[mask],
        v_obs=np.array(vo_lst)[mask],
        v_err=np.array(ve_lst)[mask],
        v_gas=np.array(vg_lst)[mask],
        v_disk=np.array(vd_lst)[mask],
        v_bulge=np.array(vb_lst)[mask],
    )


def load_all_sparc_galaxies(data_dir: str = "./sparc_data",
                             min_points: int = 5) -> List[SPARCGalaxy]:
    """Load all SPARC galaxies from Rotmod_LTG/."""
    data_path = Path(data_dir)
    rotmod_path = data_path / "Rotmod_LTG"
    if not rotmod_path.exists():
        rotmod_path = data_path
    if not rotmod_path.exists():
        raise FileNotFoundError(f"SPARC data not found at {data_path}")

    galaxies = []
    for fp in sorted(rotmod_path.glob("*_rotmod.dat")):
        try:
            gal = parse_sparc_galaxy(fp)
            if len(gal.r) >= min_points:
                galaxies.append(gal)
        except Exception as exc:
            print(f"  [skip] {fp.name}: {exc}")

    print(f"Loaded {len(galaxies)} SPARC galaxies (≥{min_points} data points)")
    return galaxies


# =============================================================================
# Physics
# =============================================================================

def rar_interpolation(g_bar: np.ndarray, a0: float) -> np.ndarray:
    """
    McGaugh et al. (2016) RAR interpolation function.

        g_obs = g_bar / (1 − exp(−√(g_bar / a₀)))

    Numerically stable at g_bar → 0: ν → 1/√x → g_obs → √(g_bar · a₀).
    """
    g = np.asarray(g_bar, dtype=float)
    out = np.zeros_like(g)
    pos = g > 0
    x   = g[pos] / a0
    sq  = np.sqrt(np.maximum(x, 0.0))
    denom = 1.0 - np.exp(-sq)
    out[pos] = np.where(denom < 1e-14,
                        np.sqrt(g[pos] * a0),
                        g[pos] / denom)
    return out


def accel_from_rotation(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    g = v²/r  with r in kpc, v in km/s → g in m/s².
    """
    return (v * v / r) * (1e6 / KPC_TO_M)


def rar_rotation_curve(r: np.ndarray,
                       v_gas: np.ndarray,
                       v_disk: np.ndarray,
                       v_bulge: np.ndarray,
                       upsilon_disk: float,
                       upsilon_bulge: float,
                       a0: float) -> np.ndarray:
    """
    Predicted rotation curve from the RAR with fixed a₀.

    V_disk_file and V_bulge_file are normalised to Υ=1; scaling:
        V_component(Υ) = V_file × √Υ

    Returns V_pred in km/s.
    """
    v_d   = v_disk  * np.sqrt(max(upsilon_disk,  1e-9))
    v_b   = v_bulge * np.sqrt(max(upsilon_bulge, 1e-9))
    v_bar_sq = np.maximum(v_gas**2 + v_d**2 + v_b**2, 0.0)
    v_bar    = np.sqrt(v_bar_sq)
    g_bar    = accel_from_rotation(r, v_bar)
    g_obs    = rar_interpolation(g_bar, a0)
    return np.sqrt(np.maximum(g_obs * r * KPC_TO_M / 1e6, 0.0))


# =============================================================================
# Fitting
# =============================================================================

def _chi2_rar(params: np.ndarray, galaxy: SPARCGalaxy, a0: float) -> float:
    """χ² objective for fixed-a₀ RAR fit with free (Υ_d, Υ_b)."""
    ud, ub = params
    if ud <= 0 or ub <= 0:
        return 1e30
    v_pred = rar_rotation_curve(
        galaxy.r, galaxy.v_gas, galaxy.v_disk, galaxy.v_bulge, ud, ub, a0
    )
    if not np.all(np.isfinite(v_pred)):
        return 1e30
    res = (galaxy.v_obs - v_pred) / galaxy.v_err
    return float(np.sum(res * res))


def fit_galaxy(galaxy: SPARCGalaxy, a0: float,
               label: str = '') -> Dict:
    """
    Fit one galaxy under the fixed-a₀ RAR protocol.

    Free parameters: Υ_disk, Υ_bulge  ∈  [UPSILON_MIN, UPSILON_MAX]

    Uses differential_evolution for global exploration followed by
    L-BFGS-B refinement.

    Returns dict with fit results.
    """
    bounds = [(UPSILON_MIN, UPSILON_MAX),
              (UPSILON_MIN, UPSILON_MAX)]

    de = differential_evolution(
        _chi2_rar, bounds=bounds, args=(galaxy, a0),
        maxiter=300, popsize=10, tol=1e-8,
        seed=0, polish=True
    )
    res = minimize(
        _chi2_rar, x0=de.x, args=(galaxy, a0),
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 2000}
    )

    ud, ub = res.x
    v_best = rar_rotation_curve(
        galaxy.r, galaxy.v_gas, galaxy.v_disk, galaxy.v_bulge, ud, ub, a0
    )
    ndof = max(len(galaxy.r) - 2, 1)

    # Hessian-based uncertainties (parameter-relative step)
    epsvec = np.maximum(1e-3 * np.abs(res.x), 1e-10)
    sigmas = _hessian_sigmas(_chi2_rar, res.x, bounds, epsvec, galaxy, a0)

    return {
        'galaxy':         galaxy.name,
        'label':          label,
        'a0':             a0,
        'upsilon_disk':   ud,
        'upsilon_bulge':  ub,
        'upsilon_disk_err':  sigmas[0],
        'upsilon_bulge_err': sigmas[1],
        'chi2':           res.fun,
        'ndof':           ndof,
        'chi2_reduced':   res.fun / ndof,
        'n_data':         len(galaxy.r),
        'success':        res.success,
        'v_model':        v_best,
    }


def _hessian_sigmas(fun, x_opt, bounds, epsvec, *args):
    """
    1-σ parameter uncertainties from the finite-difference Hessian.
    Returns NaN for parameters at a bound (constrained minimum).
    """
    n = len(x_opt)
    on_bound = np.array([
        abs(x_opt[i] - lo) < 1e-3 * (hi - lo) or
        abs(x_opt[i] - hi) < 1e-3 * (hi - lo)
        for i, (lo, hi) in enumerate(bounds)
    ])
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = epsvec[i], epsvec[j]
            xpp = x_opt.copy(); xpp[i] += ei; xpp[j] += ej
            xpm = x_opt.copy(); xpm[i] += ei; xpm[j] -= ej
            xmp = x_opt.copy(); xmp[i] -= ei; xmp[j] += ej
            xmm = x_opt.copy(); xmm[i] -= ei; xmm[j] -= ej
            H[i, j] = (fun(xpp, *args) - fun(xpm, *args)
                       - fun(xmp, *args) + fun(xmm, *args)) / (4 * ei * ej)
            H[j, i] = H[i, j]
    try:
        C = 2.0 * matrix_inv(H)
        sigmas = np.sqrt(np.maximum(np.diag(C), 0.0))
    except Exception:
        sigmas = np.full(n, np.nan)
    sigmas[on_bound] = np.nan
    return sigmas


# =============================================================================
# Batch Analysis
# =============================================================================

def run_paired_comparison(galaxies: List[SPARCGalaxy],
                          a0_uct:  float = A0_UCT_PLANCK,
                          a0_mond: float = A0_MOND
                          ) -> Tuple[List[Dict], List[Dict]]:
    """
    Fit all galaxies under both fixed-a₀ protocols.

    Returns (uct_results, mond_results) as lists of dicts.
    """
    uct_results  = []
    mond_results = []
    n = len(galaxies)
    print(f"\nFitting {n} galaxies under UCT (a₀={a0_uct:.4e}) "
          f"and MOND (a₀={a0_mond:.4e})…")

    for i, gal in enumerate(galaxies):
        try:
            ur = fit_galaxy(gal, a0_uct,  label='UCT')
            mr = fit_galaxy(gal, a0_mond, label='MOND')
            uct_results.append(ur)
            mond_results.append(mr)
            if (i + 1) % 25 == 0 or i == 0:
                print(f"  [{i+1:3d}/{n}] {gal.name:15s}  "
                      f"UCT χ²ᵣ={ur['chi2_reduced']:.2f}  "
                      f"MOND χ²ᵣ={mr['chi2_reduced']:.2f}  "
                      f"Υ_d(UCT)={ur['upsilon_disk']:.3f}")
        except Exception as exc:
            print(f"  [{i+1:3d}/{n}] {gal.name}: ERROR — {exc}")

    return uct_results, mond_results


def wilcoxon_comparison(uct_results: List[Dict],
                        mond_results: List[Dict]) -> Dict:
    """
    Paired Wilcoxon signed-rank test on per-galaxy reduced χ².

    Tests H₁: χ²ᵣ(UCT) < χ²ᵣ(MOND)  (one-sided).

    Returns dict of summary statistics matching paper Table II.
    """
    uct_chi2r  = np.array([r['chi2_reduced'] for r in uct_results])
    mond_chi2r = np.array([r['chi2_reduced'] for r in mond_results])
    uct_ud     = np.array([r['upsilon_disk'] for r in uct_results])
    mond_ud    = np.array([r['upsilon_disk'] for r in mond_results])

    diff = mond_chi2r - uct_chi2r
    stat, p_one = wilcoxon(diff, alternative='greater')

    n = len(uct_chi2r)
    return {
        'n_galaxies':         n,
        'uct_median':         float(np.median(uct_chi2r)),
        'uct_mean':           float(np.mean(uct_chi2r)),
        'uct_std':            float(np.std(uct_chi2r)),
        'mond_median':        float(np.median(mond_chi2r)),
        'mond_mean':          float(np.mean(mond_chi2r)),
        'mond_std':           float(np.std(mond_chi2r)),
        'uct_lt2':            int(np.sum(uct_chi2r  < 2)),
        'mond_lt2':           int(np.sum(mond_chi2r < 2)),
        'uct_lt5':            int(np.sum(uct_chi2r  < 5)),
        'mond_lt5':           int(np.sum(mond_chi2r < 5)),
        'uct_wins':           int(np.sum(diff > 0)),
        'mond_wins':          int(np.sum(diff < 0)),
        'median_delta_chi2r': float(np.median(diff)),
        'mean_delta_chi2r':   float(np.mean(diff)),
        'wilcoxon_stat':      float(stat),
        'wilcoxon_p_onesided': float(p_one),
        'uct_median_upsilon_d':  float(np.median(uct_ud)),
        'mond_median_upsilon_d': float(np.median(mond_ud)),
    }


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != 'v_model'}
                         for r in results])


# =============================================================================
# Plotting
# =============================================================================

def plot_rar_curves(galaxies: List[SPARCGalaxy],
                   uct_results: List[Dict],
                   mond_results: List[Dict],
                   a0_uct: float = A0_UCT_PLANCK,
                   a0_mond: float = A0_MOND,
                   save_path: Optional[str] = None):
    """
    Reproduce Figure 1 of the paper: RAR scatter with UCT and MOND curves.
    All 175-galaxy data points; solid UCT curve; dashed MOND curve.
    """
    g_bar_pts, g_obs_pts = [], []
    for gal, ur in zip(galaxies, uct_results):
        ud = ur['upsilon_disk']
        ub = ur['upsilon_bulge']
        v_d   = gal.v_disk  * np.sqrt(ud)
        v_b   = gal.v_bulge * np.sqrt(ub)
        v_bar = np.sqrt(np.maximum(gal.v_gas**2 + v_d**2 + v_b**2, 0))
        g_bar = accel_from_rotation(gal.r, v_bar)
        g_obs = accel_from_rotation(gal.r, gal.v_obs)
        mask  = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_bar + g_obs)
        g_bar_pts.extend(g_bar[mask])
        g_obs_pts.extend(g_obs[mask])

    g_bar_arr = np.array(g_bar_pts)
    g_obs_arr = np.array(g_obs_pts)

    g_range   = np.logspace(
        np.floor(np.log10(g_bar_arr.min())) - 0.5,
        np.ceil (np.log10(g_bar_arr.max())) + 0.5,
        300
    )
    g_uct_curve  = rar_interpolation(g_range, a0_uct)
    g_mond_curve = rar_interpolation(g_range, a0_mond)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(g_bar_arr, g_obs_arr, s=4, c='#999999', alpha=0.25,
               rasterized=True, label='SPARC data')
    ax.plot(g_range, g_uct_curve,  '-',  color='#c0392b', lw=2.0,
            label=f'UCT  $a_0={a0_uct*1e10:.3f}\\times10^{{-10}}$')
    ax.plot(g_range, g_mond_curve, '--', color='#2980b9', lw=2.0,
            label=f'MOND $a_0={a0_mond*1e10:.2f}\\times10^{{-10}}$')
    ax.plot(g_range, g_range, ':', color='black', lw=1.0, label='Unity')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-13, 1e-8); ax.set_ylim(1e-13, 1e-8)
    ax.set_xlabel('$g_{\\rm bar}$ (m s$^{-2}$)', fontsize=12)
    ax.set_ylabel('$g_{\\rm obs}$ (m s$^{-2}$)', fontsize=12)
    ax.set_title('Radial Acceleration Relation — SPARC', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='both', alpha=0.15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_chi2_distributions(uct_df: pd.DataFrame,
                            mond_df: pd.DataFrame,
                            save_path: Optional[str] = None):
    """
    Reproduce Figure 2 of the paper: per-galaxy χ²ᵣ histograms.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, min(uct_df['chi2_reduced'].quantile(0.97),
                              mond_df['chi2_reduced'].quantile(0.97)) * 1.05,
                       40)
    ax.hist(mond_df['chi2_reduced'].clip(upper=bins[-1]), bins=bins,
            alpha=0.55, color='#2980b9', edgecolor='black', lw=0.4,
            label=f"MOND  median={mond_df['chi2_reduced'].median():.2f}")
    ax.hist(uct_df['chi2_reduced'].clip(upper=bins[-1]),  bins=bins,
            alpha=0.55, color='#c0392b', edgecolor='black', lw=0.4,
            label=f"UCT   median={uct_df['chi2_reduced'].median():.2f}")
    ax.set_xlabel('Per-galaxy reduced $\\tilde{\\chi}^2$', fontsize=11)
    ax.set_ylabel('Number of galaxies', fontsize=11)
    ax.set_title('Distribution of reduced $\\tilde{\\chi}^2$ — 175 SPARC galaxies',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_individual_rotcurves(galaxies: List[SPARCGalaxy],
                              uct_results: List[Dict],
                              mond_results: List[Dict],
                              n_plot: int = 4,
                              save_path: Optional[str] = None):
    """
    Reproduce Figure 3 of the paper: example rotation-curve fits.

    Selects n_plot galaxies spanning the fit-quality distribution
    (best, upper-quartile, median, worst).
    """
    chi2r_uct = np.array([r['chi2_reduced'] for r in uct_results])
    idx_sorted = np.argsort(chi2r_uct)
    n = len(galaxies)
    picks = [
        idx_sorted[0],
        idx_sorted[n // 4],
        idx_sorted[n // 2],
        idx_sorted[3 * n // 4],
    ][:n_plot]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, picks):
        gal = galaxies[idx]
        ur  = uct_results[idx]
        mr  = mond_results[idx]
        r   = gal.r

        ax.errorbar(r, gal.v_obs, yerr=gal.v_err,
                    fmt='ko', ms=3.5, lw=0.7, label='SPARC data', zorder=5)
        ax.plot(r, ur['v_model'], '-',  color='#c0392b', lw=2.0,
                label=f"UCT  $\\tilde{{\\chi}}^2$={ur['chi2_reduced']:.2f}")
        ax.plot(r, mr['v_model'], '--', color='#2980b9', lw=2.0,
                label=f"MOND $\\tilde{{\\chi}}^2$={mr['chi2_reduced']:.2f}")

        # Baryonic components (UCT Υ)
        ud, ub = ur['upsilon_disk'], ur['upsilon_bulge']
        v_d = gal.v_disk  * np.sqrt(ud)
        v_b = gal.v_bulge * np.sqrt(ub)
        ax.plot(r, gal.v_gas, ':', color='steelblue',  lw=1.0, alpha=0.7)
        ax.plot(r, v_d,       ':', color='darkorange',  lw=1.0, alpha=0.7)
        if np.any(v_b > 0):
            ax.plot(r, v_b,   ':', color='firebrick',  lw=1.0, alpha=0.7)

        ax.set_xlabel('$R$ (kpc)', fontsize=10)
        ax.set_ylabel('$V$ (km s$^{-1}$)', fontsize=10)
        ax.set_title(gal.name, fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Representative SPARC rotation-curve fits', fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_upsilon_distribution(uct_df: pd.DataFrame,
                              mond_df: pd.DataFrame,
                              save_path: Optional[str] = None):
    """Distribution of best-fit Υ_disk for both protocols."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(UPSILON_MIN, min(3.0, UPSILON_MAX), 40)
    ax.hist(mond_df['upsilon_disk'], bins=bins, alpha=0.55,
            color='#2980b9', edgecolor='black', lw=0.4,
            label=f"MOND  median={mond_df['upsilon_disk'].median():.3f}")
    ax.hist(uct_df['upsilon_disk'],  bins=bins, alpha=0.55,
            color='#c0392b', edgecolor='black', lw=0.4,
            label=f"UCT   median={uct_df['upsilon_disk'].median():.3f}")
    ax.axvline(0.5, color='k', ls='--', lw=1.2,
               label='SPS prediction Υ=0.5')
    ax.set_xlabel('$\\Upsilon_*$ (disk, best-fit)', fontsize=11)
    ax.set_ylabel('Galaxies', fontsize=11)
    ax.set_title('Stellar M/L ratios (3.6 μm)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_h0_sensitivity(galaxies: List[SPARCGalaxy],
                        h0_values: Optional[List[float]] = None,
                        save_path: Optional[str] = None):
    """
    Reproduce the H₀-sensitivity analysis (Section V F).

    Runs the full 175-galaxy fit at each H₀ value and plots
    median χ²ᵣ vs H₀, analogous to the broad-minimum figure.
    This is computationally intensive; use a coarse H₀ grid.
    """
    if h0_values is None:
        h0_values = np.linspace(60, 80, 9)

    medians = []
    a0_values = []
    for h0 in h0_values:
        a0 = c * h0 * 1e6 / (2.0 * np.pi * 3.086e22)
        a0_values.append(a0)
        chi2r_list = []
        for gal in galaxies:
            try:
                r = fit_galaxy(gal, a0)
                chi2r_list.append(r['chi2_reduced'])
            except Exception:
                pass
        medians.append(np.median(chi2r_list))
        print(f"  H₀={h0:.1f}: a₀={a0*1e10:.4f}×10⁻¹⁰  "
              f"median χ²ᵣ={medians[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h0_values, medians, 'ko-', lw=1.5, ms=5)
    ax.axvline(H0_PLANCK, color='#c0392b', ls='--', lw=1.5,
               label=f'Planck H₀={H0_PLANCK}')
    ax.axvline(H0_SHOES,  color='#2980b9', ls='--', lw=1.5,
               label=f'SH0ES H₀={H0_SHOES}')
    ax.axhline(A0_MOND * 2 * np.pi * 3.086e22 / (c * 1e6),
               color='gray', ls=':', lw=1.0, label='_nolegend_')
    ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=11)
    ax.set_ylabel('Median $\\tilde{\\chi}^2$', fontsize=11)
    ax.set_title('Sensitivity of median $\\tilde{\\chi}^2$ to $H_0$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return dict(zip(h0_values.tolist(), medians))


# =============================================================================
# Summary Statistics Printer
# =============================================================================

def print_summary(stats: Dict, a0_uct: float, a0_mond: float):
    """Print the summary statistics table (paper Table II)."""
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS  (Section V, Table II)")
    print("=" * 65)
    print(f"\n  UCT  a₀ = {a0_uct:.4e} m/s²  (c·H₀_Planck / 2π)")
    print(f"  MOND a₀ = {a0_mond:.4e} m/s²  (empirical)")
    print(f"  Galaxies fitted = {stats['n_galaxies']}\n")

    hdr = f"  {'Metric':<35s}  {'UCT':>10s}  {'MOND':>10s}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    rows = [
        ("Median χ̃²",
         f"{stats['uct_median']:.2f}", f"{stats['mond_median']:.2f}"),
        ("Mean χ̃² ± std",
         f"{stats['uct_mean']:.2f}±{stats['uct_std']:.2f}",
         f"{stats['mond_mean']:.2f}±{stats['mond_std']:.2f}"),
        (f"χ̃² < 2  (of {stats['n_galaxies']})",
         f"{stats['uct_lt2']}", f"{stats['mond_lt2']}"),
        (f"χ̃² < 5  (of {stats['n_galaxies']})",
         f"{stats['uct_lt5']}", f"{stats['mond_lt5']}"),
        ("Median Υ_d",
         f"{stats['uct_median_upsilon_d']:.3f}",
         f"{stats['mond_median_upsilon_d']:.3f}"),
        (f"Per-galaxy wins (of {stats['n_galaxies']})",
         f"{stats['uct_wins']}", f"{stats['mond_wins']}"),
    ]
    for name, u, m in rows:
        print(f"  {name:<35s}  {u:>10s}  {m:>10s}")

    print(sep)
    print(f"\n  Wilcoxon one-sided p  = {stats['wilcoxon_p_onesided']:.3e}")
    print(f"  Median Δχ̃² (MOND−UCT) = {stats['median_delta_chi2r']:.3f}")
    print(f"  Mean   Δχ̃² (MOND−UCT) = {stats['mean_delta_chi2r']:.3f}")


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(data_dir:   str = "./sparc_data",
                 output_dir: str = "./fit_results",
                 run_h0_sensitivity: bool = False) -> Tuple:
    """
    Complete paper-protocol analysis pipeline.

    Steps
    -----
    1. Load all SPARC galaxies.
    2. Fit each galaxy under both fixed-a₀ protocols (UCT and MOND).
    3. Compute Wilcoxon statistics (Table II equivalent).
    4. Save CSV tables and diagnostic plots.
    5. Optionally run H₀ sensitivity sweep (Section V F).
    """
    print("=" * 65)
    print("UCT vs MOND — Fixed-a₀ RAR Protocol")
    print("Reproducing Section V of Steiger (2026)")
    print("=" * 65)
    print(f"  A0_UCT_PLANCK = {A0_UCT_PLANCK:.4e} m/s²")
    print(f"  A0_MOND       = {A0_MOND:.4e} m/s²")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    galaxies = load_all_sparc_galaxies(data_dir)
    if not galaxies:
        raise RuntimeError("No SPARC galaxies loaded — check data_dir.")

    uct_results, mond_results = run_paired_comparison(
        galaxies, a0_uct=A0_UCT_PLANCK, a0_mond=A0_MOND
    )
    if not uct_results:
        raise RuntimeError("No successful fits.")

    uct_df  = results_to_dataframe(uct_results)
    mond_df = results_to_dataframe(mond_results)
    uct_df.to_csv( output_path / "uct_fit_results.csv",  index=False)
    mond_df.to_csv(output_path / "mond_fit_results.csv", index=False)
    print(f"\nFit tables saved to {output_path}/")

    stats = wilcoxon_comparison(uct_results, mond_results)
    print_summary(stats, A0_UCT_PLANCK, A0_MOND)

    pd.Series(stats).to_csv(output_path / "wilcoxon_stats.csv", header=True)

    print("\nGenerating plots…")
    plot_rar_curves(
        galaxies, uct_results, mond_results,
        save_path=str(output_path / "rar_curves.png")
    )
    plot_chi2_distributions(
        uct_df, mond_df,
        save_path=str(output_path / "chi2_distributions.png")
    )
    plot_individual_rotcurves(
        galaxies, uct_results, mond_results,
        save_path=str(output_path / "example_rotcurves.png")
    )
    plot_upsilon_distribution(
        uct_df, mond_df,
        save_path=str(output_path / "upsilon_distribution.png")
    )

    if run_h0_sensitivity:
        print("\nRunning H₀ sensitivity sweep (Section V F)…")
        h0_grid = np.linspace(60, 80, 9)
        plot_h0_sensitivity(
            galaxies, h0_values=h0_grid,
            save_path=str(output_path / "h0_sensitivity.png")
        )

    print(f"\nAll outputs saved to {output_path}/")
    return uct_df, mond_df, galaxies, uct_results, mond_results, stats


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    run_analysis(
        data_dir   = str(script_dir / "sparc_data"),
        output_dir = str(script_dir / "fit_results"),
        run_h0_sensitivity = False,
    )
