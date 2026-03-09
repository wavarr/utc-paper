Reproducibility Package — Steiger (2026)
=========================================

Requirements: Python 3.8+, numpy, scipy, pandas, matplotlib
Data: SPARC Rotmod_LTG from http://astroweb.cwru.edu/SPARC/

To reproduce Table IV (per-galaxy rotation curve comparison):

  1. Download SPARC data:
     mkdir sparc_data && cd sparc_data
     wget http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip
     unzip Rotmod_LTG.zip && cd ..

  2. Run the analysis:
     python individual_galaxy_fits.py

  3. Output: fit_results/wilcoxon_stats.csv
     Expected: UCT wins 104/171, Wilcoxon p = 2.35e-06

Files included:
  individual_galaxy_fits.py   — Main analysis script (Table IV)
  reproduce_sparc.py          — Zero-parameter RAR comparison
  uct_fit_results.csv         — Per-galaxy UCT fit results
  mond_fit_results.csv        — Per-galaxy MOND fit results
  wilcoxon_stats.csv          — Summary statistics
  bootstrap_a0.csv            — Bootstrap a0 distribution
