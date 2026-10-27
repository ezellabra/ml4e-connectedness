# Financial Connectedness among Emerging Equity Markets
Final project for *Machine Learning for Economists* (Prof. Matteo Barigozzi), University of Bologna, 2026.
**Authors:** Marco Iacuzzi, Agustín Lancelotti, Bautista Alonso, Ezequiel Llabra
## Overview
We measure how shocks propagate across emerging equity markets using weekly USD returns of US-listed country ETFs (plus SPY as a global reference), from March 2011 onward. We compare four ways of estimating the network: (1) Diebold-Yilmaz (2014): VAR + generalized forecast error variance decomposition; (2) sparse VAR (lasso) with time-series cross-validation; (3) NETS (Barigozzi & Brownlees): Granger, contemporaneous and long-run partial-correlation networks; (4) FNETS (Barigozzi, Cho & Owens): factor-adjusted network estimation, which separates the global common factor from idiosyncratic connectedness. Methods are compared on network density, agreement between methods, and stability over time.

## Installation

Requirements: Python 3.11+ (developed with 3.13), R 4.x, and (for the paper) a LaTeX distribution with `latexmk`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/setup_r.py      # checks/installs R and the R packages nets, fnets
.venv/bin/python scripts/setup_latex.py  # checks/installs LaTeX and the packages used by paper/
```

`scripts/setup_r.py` never reinstalls R; it installs R only if missing (Homebrew on macOS, apt/dnf on Linux; on Windows it prints instructions). `nets` is no longer on CRAN and is installed from the CRAN archive (version 0.9.1). `scripts/setup_latex.py` behaves the same way for LaTeX; on macOS the BasicTeX cask needs administrator rights, in which case it prints the command to run by hand; `--user-fallback` installs TinyTeX in `~/Library/TinyTeX` instead (no administrator rights; this is what was used to compile the paper).

## Running the pipeline

```bash
make all            # or: .venv/bin/python run_all.py
make test           # unit tests (pytest)
make paper          # compile paper/main.tex with latexmk
```

`make all` runs, in order: data validation and descriptive statistics, estimation of all methods on all scenarios with the comparison, figures, and the (optional, disabled by default) forecasting module. A full run takes about 30 minutes on an 8-core laptop, almost all of it in R (NETS and FNETS). Estimated models are cached in `data/processed/` (ignored by git); delete the cache with `make clean` to force re-estimation. Every parameter is in `config.yaml`.

## Modules

| File | Content |
|---|---|
| `config.yaml` | All parameters: paths, ticker metadata, windows, horizons, tuning, robustness scenarios, events, figure settings |
| `src/data.py` | Load and validate returns, flag extreme returns, descriptive statistics, verification of the data-builder claims |
| `src/dy.py` | OLS VAR with BIC lag selection, generalized FEVD (Pesaran-Shin), Diebold-Yilmaz connectedness measures, rolling DY |
| `src/var_lasso.py` | Equation-wise lasso VAR, lambda by `TimeSeriesSplit` cross-validation or extended BIC |
| `r/run_nets.R`, `r/run_fnets.R`, `r/common.R` | NETS and FNETS estimation on windows of returns (CSV in, CSV out) |
| `r/run_fnets_forecast.R` | One-step-ahead FNETS forecasts (optional module) |
| `src/r_bridge.py` | Calls Rscript through `subprocess`, parallel R processes, error handling, returns arrays/DataFrames |
| `src/estimate.py` | Maps every method to a common representation (VAR + Granger, contemporaneous and long-run networks); caching |
| `src/networks.py` | Density, degrees, strengths, Spearman/Jaccard similarity, positive-definite projection |
| `src/compare.py` | The three comparison criteria and the FNETS-vs-DY common-factor decomposition |
| `src/pipeline.py` | Baseline comparison and robustness scenarios; writes all tables |
| `src/figures.py` | Figures and the LaTeX descriptive table |
| `src/forecast.py` | Optional out-of-sample forecasting (AR, lasso VAR, FNETS; relative RMSE/MAE, Diebold-Mariano) |
| `run_all.py`, `Makefile` | One-command entry points |
| `tests/` | Unit tests (GFEVD properties, permutation invariance, recovery on simulated VARs, lasso, R bridge, comparison, forecasting) |
| `docs/` | `decisions.md` (all modeling decisions with origin tags), `data_notes.md`, `paper_structure.md`, `summary.md` |
| `paper/` | LaTeX skeleton (`main.tex`, `sections/`, `references.bib`) |

## Results structure

```
results/
  tables/
    descriptive_stats.csv / .tex       descriptive statistics by ticker
    extreme_returns.csv                 |weekly log return| > 25% (flagged, not modified)
    series_dy_step1_H4.csv              DY rolling connectedness, step 1
    series_<scenario>_H<h>.csv          per-window measures for every method (long format)
    density_*.csv, agreement_*.csv, stability_*.csv   the three comparison criteria
    common_factor_*.csv                 share of DY connectedness lost once the common factor is removed
    numerical_diagnostics_*.csv         failed windows, PD projections, clipped partial correlations
    robustness_summary.csv / .tex       all robustness scenarios
    static_connectedness.csv / .tex     full-sample and 2011-2019 / 2021-2026 static estimates
    comparison_criteria.tex             main comparison table for the paper
    static/                             full-sample matrices per method (GFEVD, Granger, pcor, lrpc)
  figures/                              PNG and PDF (300 dpi)
```

Scenarios: `base` (w=104, step 4), `w78`, `w130`, `noSPY`, `ex2020`; horizons H=2, 4, 8 for the baseline.
