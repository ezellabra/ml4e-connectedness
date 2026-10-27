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
