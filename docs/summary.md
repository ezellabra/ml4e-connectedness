# Summary

## What was implemented

- Data loading and validation of the 816 x 18 weekly log-return panel; descriptive statistics; extreme-return flags; verification of the two data-builder claims (`src/data.py`, `docs/data_notes.md`).
- Setup scripts for R (+ `nets`, `fnets`) and LaTeX (`scripts/`).
- Diebold-Yilmaz connectedness: OLS VAR with BIC (p <= 2), generalized FEVD, total/directional/net/pairwise measures, rolling windows (`src/dy.py`).
- Sparse VAR by lasso, lambda by `TimeSeriesSplit` CV or eBIC (`src/var_lasso.py`).
- NETS and FNETS in R through a Python bridge, run in parallel (`r/`, `src/r_bridge.py`).
- A common representation of all methods (VAR + Granger, contemporaneous and long-run networks) and the three comparison criteria plus the common-factor decomposition (`src/estimate.py`, `src/compare.py`).
- Robustness: w = 78/130, H = 2/8, without SPY, excluding 2020 (gap + 2011-2019 / 2021-2026 subsamples) (`src/pipeline.py`).
- Optional forecasting module (AR, lasso VAR, FNETS; relative RMSE/MAE, Diebold-Mariano), disabled by default and not run (`src/forecast.py`).
- Figures and LaTeX tables (`src/figures.py`), LaTeX paper skeleton (`paper/`).
- 29 unit tests, all passing.

## Main results (baseline: w = 104, step 4, H = 4, 179 windows; all numbers from `results/tables/`)

1. Mean total connectedness: DY 81.0%, lasso VAR 80.0%, NETS 61.5%, FNETS (idiosyncratic) 47.2%.
2. The lasso shrinks almost all lag coefficients to zero (Granger density 2.5%; NETS 1.7%; FNETS 0.3%): weekly return connectedness is overwhelmingly contemporaneous.
3. DY and the lasso VAR agree almost perfectly (correlation of total connectedness 0.98); NETS tracks DY closely (0.89) but at a lower level and with larger window-to-window jumps.
4. Removing the common factor (FNETS vs DY) eliminates on average 41% of DY connectedness (median 42%, range 26-61% across windows); FNETS attributes on average 58% of the variance of each series to one common dynamic factor (q = 1 in every window). The FNETS common-variance share and DY total connectedness are highly correlated over time (0.95), while FNETS idiosyncratic connectedness is uncorrelated with DY (-0.00).
5. Net transmitters in DY/NETS: South Africa (EZA), Korea (EWY), Taiwan (EWT); net receivers: Turkey, Vietnam and Argentina (ARGT about -10 pp in DY, -4.6 in NETS). After removing the common factor, ARGT is close to neutral (+1.6 pp).
6. The share of DY connectedness lost without the common factor is stable across robustness scenarios: 38% (w = 78), 43% (w = 130), 42% (without SPY), 41% (excluding windows touching 2020); H does not matter for the sparse methods. Static estimates: DY total 83.0% in 2011-2019 vs 73.5% in 2021-2026 (`robustness_summary.csv`, `static_connectedness.csv`).

## Known limitations

- Returns, not volatilities (group decision); realized-volatility connectedness left as an extension.
- ETF returns embed USD exchange-rate moves and non-synchronous trading.
- FNETS with N = 18 and T = 104 is numerically fragile: the idiosyncratic covariance needed a positive-definite projection in 178/179 windows, partial correlations were clipped in about half of the windows, and the idiosyncratic connectedness series has tuning-driven drops (D4.8, D4.10).
- NETS lambda is chosen by our BIC on a grid (the package does not select it); the BIC-chosen lambda switches between adjacent grid values, which makes NETS series jumpy.
- The FNETS-vs-DY comparison mixes the factor effect with estimator differences (lasso, standardization, projection).
- `nets` is archived on CRAN (installed from the archive).

## Pending items (keep listing these until resolved)

- **Summary of the base paper:** Agustín Lancelotti wrote the one-page summary; it must replace the stopgap summary in `paper/sections/00_base_paper.tex` (written from the base paper PDF) once he sends it.
- **The professor's papers on the idiosyncratic component:** to be requested/sent; they go into the base-paper summary and the bibliography.
- **Argentina's role (decision D2): TO BE DEFINED.** For now it is one case among the 17 markets (Motivation, Results, figures).
- **SPY as reference node (decision D3):** kept in the baseline; confirmation from the group pending.
- **Research question, hypotheses and title (D1, A8):** drafted in Introduction and on the cover; the group must confirm the final wording and title.
- **Repository** (github.com/ezellabra/ML4E-code): make it public after force-pushing the cleaned history; the URL is already in Section 9.
- **Discussion and Conclusion:** the group's own work (course policy).

## What is missing

- The paper skeleton compiles (`make paper`, TinyTeX) to 12 pages, mostly figures and tables; Results, Discussion, Conclusion, Motivation and Introduction are still placeholders, so the final length is open (budget 20).
- Results, discussion, conclusion, motivation and introduction text (authors).
- Exclusion reasons per ticker in `docs/data_notes.md` (authors).
- Forecasting module not run (pending professor's confirmation).
- Not implemented: local-index robustness (no data), Latin-American subset (suggested in paper structure v2), the professor's idiosyncratic-component papers in the bibliography.
