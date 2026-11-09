# Modeling decisions

Each entry states what was decided, why, and its origin tag:
**[Base paper]** = Diebold and Yilmaz (2014); **[Course]** = Machine Learning for Economists course material / professor's request; **[Adaptation]** = our choice to adapt the method to this data set.

## 1. Data

**D1.1 Weekly frequency.** Weekly Friday-to-Friday returns. Why (group decision): weekly returns retain some cross-market predictability (see D1.5); weekly data also mitigates non-synchronous trading between Asian/EMEA local markets and US-listed ETFs. [Course]

**D1.2 US-listed country ETFs in USD.** 17 emerging-market country ETFs plus SPY. Why: a homogeneous, tradable, USD-denominated proxy for each market with a long common history. Returns therefore include currency moves against the USD. [Adaptation]

**D1.3 Universe selection.** History since 2011-03-03 (7-day tolerance), at most 5% missing weeks, average daily volume of at least USD 3M. Colombia (ICOL, GXG), INDA and MCHI excluded; India and China represented by EPI and FXI. See `docs/data_notes.md`. [Adaptation]

**D1.4 Sample.** 2011-03-11 to 2026-10-23, 816 weeks, no missing values. The start date is set by the youngest ETF in the universe. [Adaptation]

**D1.5 Returns, not volatilities.** The base paper measures connectedness of (realized) volatilities. We choose log returns (group decision): (i) the question is about the transmission of return shocks across markets; (ii) weekly returns retain some cross-market predictability, which is what the VAR-based networks capture; realized-volatility connectedness from daily prices is left as an extension; (iii) returns are not negligible as a source of cross-market predictive dependence: in a full-sample VAR(1) on the 18 series, 18.2% of the 324 slope coefficients are significant at 5% (59/324), against 5% expected by chance (reproduced, see `docs/data_notes.md`, claim (b)). Interpretation consequence: we measure transmission of return (level) shocks, not of risk. [Adaptation]

**D1.6 Data not cleaned of extremes.** Weekly |log returns| above 25% (5 cases: ARGT Aug-2019, four COVID-crash observations) are flagged for review but kept. They are genuine market moves. [Adaptation]

**D1.7 Known caveats.** FXI tracks the FTSE China 50 Index (50 large-cap Chinese companies listed in Hong Kong; verified on the iShares fund page); EPI tracks the earnings-weighted WisdomTree India Earnings Index (verified on StockAnalysis; the issuer page could not be accessed); Colombia is not in the sample; volatility connectedness is not studied. An earlier claim that ARGT changed its underlying index in August 2014 could not be verified on the fund's page and was removed from the paper. [Adaptation]

**D1.8 SPY as global reference node.** SPY is part of the baseline system, as the global reference market. Robustness: rerun excluding SPY. [Course] (paper structure v2, decision 4)

## 2. Diebold-Yilmaz connectedness (`src/dy.py`)

**D2.1 VAR estimation.** Unrestricted VAR(p) with intercept, equation-by-equation OLS; residual covariance with degrees-of-freedom correction (T - Np - 1). [Base paper]

**D2.2 Lag order.** p chosen by BIC over p in {1, 2}, on a common estimation sample within each window. Cap p <= 2 because with N = 18 and a 104-week window, p = 3 means 55 regressors per equation for about 100 observations, which over-parametrizes OLS. The base paper uses VAR(3) with far fewer variables. In practice BIC selects p = 1 in all 713 rolling windows (and in the full sample). [Adaptation]

**D2.3 Generalized FEVD.** Pesaran and Shin (1998) generalized variance decomposition, invariant to variable ordering, rows normalized to sum to one. [Base paper]

**D2.4 Connectedness measures.** Total (average off-diagonal row sum, in %), directional "to" (column sums of off-diagonal elements), "from" (row sums), net (to - from), and pairwise net (theta_ji - theta_ij). [Base paper]

**D2.5 Horizon and window.** Baseline H = 4 weeks and a rolling window of w = 104 weeks (two years), step 1 (713 windows, first window ends 2013-03-01). Robustness: w in {78, 130}, H in {2, 8}. The base paper uses daily data with w = 100 days and H = 12 days; we keep the logic (window long enough for a stable VAR, horizon of about one month) on the weekly scale. With N = 18 the window must be much longer than 100 observations in calendar terms anyway. [Adaptation]

**D2.6 Window dating.** Each rolling estimate is dated at the last week of its window. [Base paper]

**D2.7 Common evaluation grid.** For the comparison across methods, DY is also run with step 4 weeks (179 windows), the same grid used by NETS and FNETS (see D4.3). [Adaptation]

## 3. Sparse VAR (`src/var_lasso.py`)

**D3.1 Role.** A lasso-penalized VAR is the bridge between the dense DY VAR and the sparse NETS/FNETS estimators: same GFEVD and same connectedness measures as DY, only the estimator changes, so differences isolate the effect of regularization. [Course: lasso]

**D3.2 Estimator.** Equation-by-equation lasso (scikit-learn coordinate descent) on all lags 1..p_max with p_max = 4; regressors standardized within each training sample, intercept unpenalized. The lasso itself performs lag selection, so no separate lag criterion is used. Cap p <= 4 (72 candidate regressors per equation for about 100 observations). [Course: lasso] [Adaptation]

**D3.3 Choice of lambda.** Baseline: time-series cross-validation with `TimeSeriesSplit` (5 expanding folds, test blocks always after the training block), minimizing out-of-sample MSE over a log-spaced grid of 40 values from lambda_max to 0.01 lambda_max, chosen separately per equation. Random K-fold is never used because it leaks future information into the training set. Alternative: extended BIC (Chen and Chen, 2008) with gamma = 0.5. [Course]

**D3.4 Residual covariance.** Sigma estimated from the lasso residuals, divided by the effective sample size. [Adaptation]

**D3.5 Finding to keep in mind.** With 104-week windows, cross-validation shrinks almost all lag coefficients to zero (full sample: 0.6% non-zero with CV, 0% with eBIC). The lasso-VAR connectedness is then driven almost entirely by the contemporaneous covariance (for H steps with Psi_h close to zero, GFEVD reduces to normalized squared correlations). This is a substantive result, not a bug: weekly return predictability is weak relative to the estimation noise in short windows. [Adaptation]

## 4. NETS and FNETS (`r/`, `src/r_bridge.py`, `src/estimate.py`)

**D4.1 Language bridge.** NETS and FNETS are only available in R. Python writes each window to CSV, calls `Rscript r/run_<method>.R <manifest> <params>` through `subprocess`, and reads CSV outputs back. All tuning parameters are passed from `config.yaml` (no constants in the R code); each window gets seed `seed + window id`. [Adaptation]

**D4.2 Package versions.** `fnets` 0.1.6 from CRAN. `nets` has been removed from CRAN; version 0.9.1 is installed from the CRAN archive by `scripts/setup_r.py`. [Adaptation]

**D4.3 Common evaluation grid.** NETS and FNETS (and, for the comparison, DY and the lasso VAR) are estimated on the same rolling windows as DY (w = 104) but with step 4 weeks: 179 windows, the first ending 2013-03-01 and the last 2026-10-23 (window ends are rows 103, 107, ..., 815). Step 4 keeps the R runtime manageable (FNETS takes about 3-4 seconds per window, 90 seconds on the full sample). Windows are run in parallel across 8 R processes. [Adaptation]

**D4.4 Standardization.** Each window is demeaned and scaled to unit variance before NETS and FNETS. Without scaling, FNETS's data-driven tuning of the partial-correlation estimator produced partial correlations of order 10^6 (the raw weekly return variances are about 10^-3). Standardization also puts the lasso penalties of the different series on a common scale. GFEVD shares and partial correlations are scale invariant, so this does not affect comparability with DY. [Adaptation]

**D4.5 NETS specification.** VAR(1) (Granger network) and innovation concentration matrix (contemporaneous network) estimated jointly by the NETS algorithm with adaptive lasso weights (`activeshooting`). The `nets` package requires lambda to be supplied; we choose it by BIC (Gaussian likelihood with the estimated concentration matrix, degrees of freedom = non-zero autoregressive coefficients + non-zero off-diagonal partial correlations) over 15 log-spaced values from 100 down to 0.05. The long-run partial-correlation network is computed from the estimates as in Barigozzi and Brownlees (2019): K_LR = (I - A(1))' K (I - A(1)), partial correlations -K_LR,ij / sqrt(K_LR,ii K_LR,jj). [Base: NETS paper] [Adaptation: lambda by BIC]

**D4.6 FNETS specification.** Unrestricted (generalized dynamic) factor model; number of factors chosen by the Hallin and Liska (2007) information criterion (`q = "ic"`, criterion IC5, the package default, set explicitly as `ic_op: 5` in config); idiosyncratic VAR(1) estimated by lasso with lambda chosen by the package's BIC; contemporaneous and long-run partial-correlation networks of the idiosyncratic component estimated by the package's CLIME-type estimator with adaptive thresholding. [Course: FNETS] [Adaptation: tuning by BIC for speed]

**D4.7 Share of variance explained by the common component.** For each series, var(common)/var(series), from the lag-0 autocovariances of the common and observed series as estimated by FNETS. [Course: FNETS]

**D4.8 Numerical safeguards for FNETS.** With N = 18 and T = 104, the FNETS estimates of the idiosyncratic innovation covariance are often not positive definite, and the partial-correlation estimates can fall outside [-1, 1]. (i) For the GFEVD, the covariance is projected onto the positive-definite cone by clipping eigenvalues at 1% of the mean absolute eigenvalue. (ii) Partial correlations are clipped to [-1, 1] before computing network strengths; the support (which links are non-zero) is unchanged. In the baseline the projection was needed in 178 of 179 windows; partial correlations were clipped in 96 windows (contemporaneous) and 89 (long-run). Counts per scenario are in `results/tables/numerical_diagnostics_<scenario>.csv`. This is a limitation: FNETS is designed for large N and T. [Adaptation]

**D4.10 Known instability of FNETS idiosyncratic connectedness.** The FNETS GFEVD series shows sharp one-window drops (to about 30-35%). They coincide with windows in which the BIC-tuned idiosyncratic VAR has several non-zero coefficients (correlation between total connectedness and the number of non-zero coefficients: -0.72). The drops therefore reflect tuning noise in short windows rather than economic events, and should be read with care (median-based summaries are also reported). [Adaptation]

**D4.11 Horizon insensitivity of sparse methods.** Because the estimated lag coefficients are almost all zero in NETS, FNETS and the lasso VAR, their GFEVD is nearly the same for H = 2, 4 and 8 (it is essentially a normalized function of the innovation covariance). [Adaptation]

**D4.9 Comparable measures.** Every method is mapped to (i) a VAR (coefficients + innovation covariance) on which the same GFEVD connectedness as DY is computed, and (ii) three networks: Granger (sum over lags of |coefficients|, directed), contemporaneous partial correlations, and long-run partial correlations (undirected). For DY and the lasso VAR, the partial-correlation networks are computed from the estimated VAR in the same way as for NETS. For FNETS all objects refer to the idiosyncratic component, so FNETS GFEVD connectedness is "idiosyncratic connectedness". Directed networks give in-, out- and net degree; undirected give degree and strength. [Adaptation]

## 5. Comparison between methods (`src/compare.py`)

**D5.1 Three criteria**, from the paper structure v2: network density, agreement between methods, stability over time. All are computed on the common grid of 179 windows (w = 104, step 4, H = 4). [Course]

**D5.2 Density.** (a) Share of non-zero off-diagonal links in the Granger, contemporaneous and long-run partial-correlation networks (|w| > 1e-8); (b) for GFEVD tables, which are dense by construction, the share of pairwise shares above the equal-share benchmark 1/N; (c) total GFEVD connectedness. [Adaptation]

**D5.3 Agreement.** For every pair of methods: Pearson and Spearman correlation of the total-connectedness series; average over countries of the time-series correlation of net connectedness; Spearman correlation of the countries' average net connectedness (ranking of transmitters); per-window Spearman correlation between the off-diagonal entries of the two methods' matrices (absolute values), averaged over windows; Jaccard similarity of the strongest 10% of links, averaged over windows. Done for the GFEVD, Granger, contemporaneous and long-run networks. [Adaptation]

**D5.4 Stability.** Per method: standard deviation and mean absolute change of total connectedness across consecutive windows; Spearman correlation and top-10% Jaccard similarity between consecutive windows' matrices (higher = more stable). [Adaptation]

**D5.5 Common factor vs idiosyncratic.** The share of DY connectedness that disappears once the common factor is removed is 1 - C_FNETS / C_DY, where C_FNETS is the GFEVD total connectedness of the FNETS idiosyncratic VAR, computed per window and averaged. It is reported together with the FNETS share of variance explained by the common component. Caveat: the two systems differ in more than the factor (lasso vs OLS, standardization, PD projection); the lasso VAR and NETS rows help separate the regularization effect from the factor effect. [Adaptation]

## 6. Robustness (`src/pipeline.py`)

**D6.1 Scenarios.** Windows w in {78, 130} (H = 4); horizons H in {2, 8} (w = 104, reusing the same estimated VARs); excluding SPY (17 emerging markets only); excluding 2020. Each scenario reruns all four methods on the step-4 grid. Summary in `results/tables/robustness_summary.csv` / `.tex`. [Course] [Adaptation]

**D6.2 Excluding 2020.** (a) Every rolling window that contains any week of 2020 is dropped (series set to missing; plots leave a visible gap and never connect across it); (b) static full-period connectedness estimated separately on 2011-2019 and 2021-2026. [Adaptation]

**D6.3 Not done.** Realized-volatility connectedness (left as an extension) and local-index robustness (no local index data in the repository). [Adaptation]

## 7. Out-of-sample forecasting (`src/forecast.py`, optional)

**D7.1 Status.** Implemented but off by default (`forecast.enabled: false`), pending confirmation from the professor. [Course]

**D7.2 Design.** One-week-ahead forecasts, rolling 104-week window (expanding available). Benchmark: univariate AR(p) per series with p in 0..4 by BIC. Competitors: lasso VAR (same settings as Phase 3) and FNETS (`predict.fnets`, restricted representation of the common component; fitted on standardized data, forecasts transformed back). Metrics: RMSE and MAE relative to AR; Diebold-Mariano test with the Harvey-Leybourne-Newbold correction. [Course] [Adaptation]

## 8. Figures and tables (`src/figures.py`)

**D8.1 Readability in grayscale.** Methods are distinguished by line style (solid, dashed, dash-dot, dotted) as well as color; regions in bar charts by hatching as well as color; Argentina (ARGT) is outlined in black. PNG and PDF at 300 dpi. [Course]

**D8.2 Events shaded** in time-series plots (taper tantrum 2013, China devaluation 2015-16, Argentine PASO Aug-2019, COVID-19 2020, Fed tightening 2022); dates in `config.yaml`. Because estimates are dated at the window end, an event affects the series for the following 104 weeks. [Adaptation]

**D8.3 Network plot.** DY panel shows the strongest 10% of GFEVD links (the GFEVD table is dense); NETS and FNETS panels show all non-zero long-run partial correlations. [Adaptation]

## 9. Reproducibility

**D9.1** Fixed seed (`seed` in `config.yaml`; R windows use seed + window id); all parameters in `config.yaml`; estimated models cached in `data/processed/` keyed by a hash of data, windows and parameters; one command (`make all` / `python run_all.py`). [Course]
