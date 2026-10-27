# Data notes

## Source and construction

- File: `data/raw/retornos_semanales.csv` (committed; never rebuilt by the pipeline).
- 816 weekly Friday-to-Friday **log returns** in USD, from 2011-03-11 to 2026-10-23, 18 columns, no missing values.
- Built by the data builder from yfinance prices with `auto_adjust=True` (dividend- and split-adjusted), so returns are total returns.
- Columns: ARGT, ECH, EWZ, EWW, EPU (Latin America); EWY, EWT, EPI, FXI, THD, EIDO, EWM, VNM, EPHE (Asia); EZA, TUR, EPOL (EMEA); SPY (global reference).
- Ticker metadata (country, region, index tracked) is in `config.yaml` under `metadata`.

## Universe selection criteria (applied by the data builder)

1. Price history available since 2011-03-03 (7-day tolerance).
2. At most 5% missing weeks.
3. Average daily traded volume of at least USD 3M.

Excluded candidates:

| Ticker | Market | Reason |
|---|---|---|
| ICOL | Colombia | TODO: exact reason per ticker (authors to complete) |
| GXG | Colombia | TODO: exact reason per ticker (authors to complete) |
| INDA | India | TODO: exact reason per ticker (authors to complete) |
| MCHI | China | TODO: exact reason per ticker (authors to complete) |

India and China are represented by EPI and FXI respectively.

## Validation (`src/data.py`)

Checks run on every load: dates sorted, exact 7-day spacing, no duplicate dates or columns, no NaN, every ticker has metadata. All checks pass on the current file.

## Extreme returns flagged for review (|log return| > 0.25; data NOT modified)

| Date | Ticker | Log return | Likely event (to be confirmed by authors) |
|---|---|---|---|
| 2019-08-16 | ARGT | -0.2845 | Argentine primary election (PASO) shock |
| 2020-03-13 | EPOL | -0.2507 | COVID-19 crash |
| 2020-03-20 | EIDO | -0.3365 | COVID-19 crash |
| 2020-03-20 | EWZ | -0.2648 | COVID-19 crash |
| 2020-03-20 | EZA | -0.2693 | COVID-19 crash |

Machine-readable copy: `results/tables/extreme_returns.csv`. Descriptive statistics (mean, std, min, max, skewness, kurtosis, ADF p-value) are in `results/tables/descriptive_stats.csv`; the ADF test rejects a unit root for every series (p < 0.001).

## Caveats

- **ARGT** changed its underlying index in August 2014 (from the Solactive Argentina Index to the MSCI All Argentina 25/50 Index). Its pre- and post-2014 returns track slightly different baskets.
- **FXI** tracks the FTSE China 50, i.e. only large-cap Chinese companies (heavy in financials and state-owned enterprises); it is not a broad China index.
- **EPI** is the WisdomTree India Earnings fund (earnings-weighted), not an MSCI India tracker (INDA was excluded).
- **Colombia** is not in the sample (no Colombia ETF passed the selection criteria).
- **No daily data** is available in the repository, so realized-volatility measures (as used by Diebold and Yilmaz, 2014) cannot be built. The analysis uses returns.
- ETF returns are US-market USD returns: they embed exchange-rate moves and are subject to non-synchronous trading with the local market (Asian markets close before the US open). Weekly frequency mitigates, but does not remove, this.

## Verification of data-builder claims

### (a) "The week of March 2020 concentrates 7 of the 10 largest weekly drops across the 17 emerging markets"

**Result: NOT CONFIRMED as stated** (the qualitative point, that the COVID crash dominates the largest drops, holds).

Definitions tested (17 emerging markets, SPY excluded):

1. *Pooled* (primary definition): the 10 most negative country-week returns among the 17 x 816 observations.
   - 8 of 10 fall in March 2020 (weeks ending 2020-03-13 and 2020-03-20).
   - 5 of 10 fall in the single week ending 2020-03-20; 3 in the week ending 2020-03-13.
   - The other two are ARGT on 2019-08-16 (PASO) and TUR on 2018-08-10 (lira crisis).
2. *Per market*: each market's single worst week. 14 of 17 markets have their worst week in March 2020 (exceptions: ARGT 2019-08-16, TUR 2018-08-10, FXI 2011-09-23).
3. *Equal-weighted EM average*: the 10 worst weeks of the cross-sectional average return. 2 of 10 are in March 2020 (they are the two worst weeks: 2020-03-20 at -17.3% and 2020-03-13 at -16.3%).

No definition yields 7 of 10. The closest is the pooled count (8 for March 2020, 5 for a single week).

### (b) "In a preliminary VAR(1) on the full sample, about 18% of coefficients are significant at 5%"

**Result: CONFIRMED.** VAR(1) with constant on all 18 series (816 observations, OLS, statsmodels): 59 of 324 slope coefficients (intercepts excluded) have p < 0.05, i.e. **18.2%**, against 5% expected under the null of no predictability. Excluding the diagonal (own-lag) coefficients gives 17.6%.
