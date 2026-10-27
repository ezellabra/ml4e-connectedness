# Modeling decisions

Each entry states what was decided, why, and its origin tag:
**[Base paper]** = Diebold and Yilmaz (2014); **[Course]** = Machine Learning for Economists course material / professor's request; **[Adaptation]** = our choice to adapt the method to this data set.

## 1. Data

**D1.1 Weekly frequency.** Weekly Friday-to-Friday returns. Why: the professor indicated that at weekly frequency there is some forecastability that is absent at daily frequency; weekly data also mitigates non-synchronous trading between Asian/EMEA local markets and US-listed ETFs. [Course]

**D1.2 US-listed country ETFs in USD.** 17 emerging-market country ETFs plus SPY. Why: a homogeneous, tradable, USD-denominated proxy for each market with a long common history. Returns therefore include currency moves against the USD. [Adaptation]

**D1.3 Universe selection.** History since 2011-03-03 (7-day tolerance), at most 5% missing weeks, average daily volume of at least USD 3M. Colombia (ICOL, GXG), INDA and MCHI excluded; India and China represented by EPI and FXI. See `docs/data_notes.md`. [Adaptation]

**D1.4 Sample.** 2011-03-11 to 2026-10-23, 816 weeks, no missing values. The start date is set by the youngest ETF in the universe. [Adaptation]

**D1.5 Returns, not volatilities.** The base paper measures connectedness of (realized) volatilities. We use log returns because (i) the repository has no daily data, so realized weekly volatility cannot be built; (ii) the professor's emphasis is on forecastability at weekly frequency, which concerns returns; (iii) returns are not negligible as a source of cross-market predictive dependence: in a full-sample VAR(1) on the 18 series, 18.2% of the 324 slope coefficients are significant at 5% (59/324), against 5% expected by chance (reproduced, see `docs/data_notes.md`, claim (b)). Interpretation consequence: we measure transmission of return (level) shocks, not of risk. [Adaptation]

**D1.6 Data not cleaned of extremes.** Weekly |log returns| above 25% (5 cases: ARGT Aug-2019, four COVID-crash observations) are flagged for review but kept. They are genuine market moves. [Adaptation]

**D1.7 Known caveats.** ARGT changed underlying index in Aug-2014; FXI covers only large-cap Chinese firms (FTSE China 50); EPI is the WisdomTree India (earnings-weighted) fund; Colombia is not in the sample; no daily data, so realized-volatility measures are not available. [Adaptation]

**D1.8 SPY as global reference node.** SPY is part of the baseline system, as the global reference market. Robustness: rerun excluding SPY. [Course] (paper structure v2, decision 4)

## 2. Diebold-Yilmaz connectedness (`src/dy.py`)

**D2.1 VAR estimation.** Unrestricted VAR(p) with intercept, equation-by-equation OLS; residual covariance with degrees-of-freedom correction (T - Np - 1). [Base paper]

**D2.2 Lag order.** p chosen by BIC over p in {1, 2}, on a common estimation sample within each window. Cap p <= 2 because with N = 18 and a 104-week window, p = 3 means 55 regressors per equation for about 100 observations, which over-parametrizes OLS. The base paper uses VAR(3) with far fewer variables. In practice BIC selects p = 1 in every window. [Adaptation]

**D2.3 Generalized FEVD.** Pesaran and Shin (1998) generalized variance decomposition, invariant to variable ordering, rows normalized to sum to one. [Base paper]

**D2.4 Connectedness measures.** Total (average off-diagonal row sum, in %), directional "to" (column sums of off-diagonal elements), "from" (row sums), net (to - from), and pairwise net (theta_ji - theta_ij). [Base paper]

**D2.5 Horizon and window.** Baseline H = 4 weeks and a rolling window of w = 104 weeks (two years), step 1 (713 windows, first window ends 2013-03-01). Robustness: w in {78, 130}, H in {2, 8}. The base paper uses daily data with w = 100 days and H = 12 days; we keep the logic (window long enough for a stable VAR, horizon of about one month) on the weekly scale. With N = 18 the window must be much longer than 100 observations in calendar terms anyway. [Adaptation]

**D2.6 Window dating.** Each rolling estimate is dated at the last week of its window. [Base paper]

**D2.7 Common evaluation grid.** For the comparison across methods, DY is also run with step 4 weeks (179 windows), the same grid used by NETS and FNETS (see D4.x). [Adaptation]

## 3. Sparse VAR (`src/var_lasso.py`)

**D3.1 Role.** A lasso-penalized VAR is the bridge between the dense DY VAR and the sparse NETS/FNETS estimators: same GFEVD and same connectedness measures as DY, only the estimator changes, so differences isolate the effect of regularization. [Course: lasso]

**D3.2 Estimator.** Equation-by-equation lasso (scikit-learn coordinate descent) on all lags 1..p_max with p_max = 4; regressors standardized within each training sample, intercept unpenalized. The lasso itself performs lag selection, so no separate lag criterion is used. Cap p <= 4 (72 candidate regressors per equation for about 100 observations). [Course: lasso] [Adaptation]

**D3.3 Choice of lambda.** Baseline: time-series cross-validation with `TimeSeriesSplit` (5 expanding folds, test blocks always after the training block), minimizing out-of-sample MSE over a log-spaced grid of 40 values from lambda_max to 0.01 lambda_max, chosen separately per equation. Random K-fold is never used because it leaks future information into the training set. Alternative: extended BIC (Chen and Chen, 2008) with gamma = 0.5. [Course]

**D3.4 Residual covariance.** Sigma estimated from the lasso residuals, divided by the effective sample size. [Adaptation]

**D3.5 Finding to keep in mind.** With 104-week windows, cross-validation shrinks almost all lag coefficients to zero (full sample: 0.6% non-zero with CV, 0% with eBIC). The lasso-VAR connectedness is then driven almost entirely by the contemporaneous covariance (for H steps with Psi_h close to zero, GFEVD reduces to normalized squared correlations). This is a substantive result, not a bug: weekly return predictability is weak relative to the estimation noise in short windows. [Adaptation]
