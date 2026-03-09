# Reproducibility Package: UCT SPARC Analysis

This package reproduces **Table IV** of:

> **Universal Compression Theory: A Single Scalar Field Unifying
> Galactic Dynamics, Cosmology, and Inflation** (Paper I)
> by Gabriel Steiger (March 2026)

## What it reproduces

The paper tests four MOND interpolation functions against the SPARC Radial
Acceleration Relation (3,351 data points from 175 galaxies):

| Function | n    | Free params | chi2_red | chi2_tilde |
|----------|------|-------------|----------|------------|
| UCT      | 0.72 | 0           | 53.38    | 1.03       |
| Simple   | 1    | 1           | 52.16    | 1.00       |
| RAR      | ---  | 1           | 54.08    | 1.04       |
| Standard | 2    | 1           | 110.72   | 2.13       |

The key result: **UCT with zero free parameters** achieves chi2_red = 53.4,
competitive with the best one-parameter fit (Simple at 52.2), and
outperforming the empirical RAR function (54.1).

## The physics in brief

MOND (Modified Newtonian Dynamics) describes galactic rotation curves using
an interpolation function mu(x) that transitions between Newtonian gravity
(mu -> 1 for large x) and deep MOND (mu -> x for small x), where
x = g/a0 is the acceleration in units of a characteristic scale a0.

UCT predicts both parameters from first principles:
- **a0 = cH0/(2pi) = 1.041e-10 m/s^2** (derived from the Hubble parameter)
- **n = ln2/ln(phi^2) = 0.7202** (derived from the golden ratio phi)
- The interpolation function is: **mu(x) = x / (1 + x^n)^{1/n}**

No parameters are fitted to the data.

## Dependencies

```
pip install numpy scipy matplotlib
```

Or:

```
pip install -r requirements.txt
```

## SPARC data

The script expects `SPARC_table2.dat` from the SPARC database:

**Download from:** http://astroweb.cwru.edu/SPARC/

Place the file either:
- In this directory, or
- In a `data/` subdirectory

The file contains rotation curve data for 175 late-type galaxies
(Lelli, McGaugh & Schombert 2017, ApJ 836, 152).

## How to run

```bash
python reproduce_sparc.py
```

Runtime: approximately 10 seconds on a modern laptop.

## Expected output

The script prints:
1. UCT predicted constants (a0, n, phi)
2. Data summary (3,351 points, 175 galaxies)
3. **Table IV** with chi2_red for all four functions
4. Verification against published values (all match to <0.1%)
5. Split-RMS analysis across acceleration regimes
6. Per-galaxy chi2 distribution
7. n-scan analysis (chi2 as function of exponent n)

Six diagnostic figures are saved to `figures/`:

- `fig1_rar_comparison` -- RAR with all four model curves overlaid
- `fig2_residuals` -- Residual panels for each function
- `fig3_n_scan` -- chi2 vs exponent n, showing UCT lies near the minimum
- `fig4_pergalaxy_chi2` -- Per-galaxy chi2 distribution and UCT vs Simple scatter
- `fig5_uct_zero_params` -- UCT zero-parameter prediction highlighted
- `fig6_residual_histograms` -- Distribution of log-space residuals

All figures are saved in both PNG (300 dpi) and PDF formats.

## Methodology details

**Data processing:**
- Baryonic acceleration: g_bar = (Y_d * V_d^2 + Y_b * V_b^2 + V_g|V_g|) / R
- Mass-to-light ratios: Y_disk = 0.5, Y_bulge = 0.7 M_sun/L_sun
- Quality cuts: g_bar > 0, g_obs > 0, sigma_g > 0, sigma_g < g_obs

**Fitting procedure:**
- The AQUAL equation mu(x)*x = x_N is solved via Brent's root-finding algorithm
- Best-fit a0 found by bounded scalar minimization (scipy.optimize.minimize_scalar)
- UCT uses a0 = cH0/(2pi) with no fitting

**Chi-squared:**
- Computed in log-acceleration space: chi2 = sum[(log10 g_obs - log10 g_pred)^2 / sigma_log^2]
- sigma_log = sigma_g / (g_obs * ln 10)
- Reduced: chi2_red = chi2 / (N - k), where k = number of free parameters

**Why chi2_red ~ 50 for all models:**
The large chi2_red values reflect that reported SPARC velocity errors
(~5-10 km/s) are purely observational and exclude intrinsic galaxy-to-galaxy
scatter from distance uncertainties, inclination errors, and baryonic mass
variations. The intrinsic RAR scatter is ~0.13 dex; accounting for it
reduces chi2_red to O(1) for all competitive functions. The relative ranking
is robust to error rescaling.

## Citation

If you use this code, please cite:
- Steiger (2026), Paper I of the UCT series
- Lelli, McGaugh & Schombert (2017), ApJ 836, 152 (SPARC database)
- McGaugh, Lelli & Schombert (2016), PRL 117, 201101 (RAR discovery)
