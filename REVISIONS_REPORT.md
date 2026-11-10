# Paper revisions: report

## A1: covariance used by the FNETS GFEVD
**Case A.** The GFEVD uses the estimated idiosyncratic innovation covariance (`fn$idio.var$Gamma`, `r/run_fnets.R:28`), clipped to positive definite by `nearest_pd` (`src/estimate.py:69`) and passed to `gfevd` (`src/estimate.py:124`). It is not derived from the thresholded/CLIME precision matrix, so C1 was applied as written.

## Placeholder values
| Placeholder | Value | Placeholder | Value |
|---|---|---|---|
| {IS} / {IS_round} | 19.6 / 20 | {N60} | 61.4 |
| {IS_min} / {IS_max} | 10.0 / 28.1 | {N60_jump} | 4.5 (corr. with DY 0.90) |
| {IS_full} | 14.6 | {cs2} | 68.9 |
| {IS_full_lo}, {IS_full_hi} | 13.3, 18.3 | {F2} | 38.6 |
| {CB} / {CB_round} | 81.1 / 81 | {lost2} | 52.2 |
| {rB} | 0.98 | {cs_ARG} / {rank_ARG} / {ord_ARG} | 48.6 / 16 / sixteenth |
| {RB} | 100.2 | {spy_dy} / {spy_f} | +2.7 / +2.4 |
| {CB_full} | 83.4 | IS, C^B under q = 2 | 11.7, 81.0 |

## Conditional changes
- **C4:** q-case 2 (lost2 = 52.2 > 46), so the "although it changes when two factors are imposed" ending was used.
- **C11a:** rank 16 >= 10, so the "does not reflect a stronger exposure" version was used.
- **C16a:** N60_jump = 4.5 > 3.4, so the "does not remove this" version was used.
- **C16b:** q-case 2, so the "Consistently, the loss changes" sentence was used.
- **C18a:** q-case 2, so the "41% should be read as a lower bound" sentence was used.

## Notes
- All pre-existing numbers are unchanged: the tables, the series CSVs and the static matrices were compared against the pre-change snapshot, and the only differences are added columns/rows.
- {RB} = 100.2% follows the stated formula (100 x C^B / 81.0), so the C14b sentence reads that a world without contagion would produce 100.2% of the DY connectedness. Please check the wording.
- Under q = 2 the loss of connectedness (52%) differs from the baseline (41%), so the paper now says so.
- The existing Table 1 has a 6 pt overfull box. It predates this work and was not touched.
- The paper is 20 pages.

## Bootstrap
B = 200, block length 8, seed 42. Runtime 3.9 minutes (well under the 3-hour limit). 2 of the 200 FNETS fits failed and are dropped from the percentiles. Other new runtimes: NETS with the 60-value grid about 29 minutes; FNETS refits about 3 minutes per scenario.

## Files changed
`config.yaml`, `Makefile`, `run_all.py`, `r/run_fnets.R`, `src/{compare,estimate,figures,pipeline,r_bridge}.py`, new `src/extras.py`, `tests/test_compare.py`, `paper/sections/*.tex` (00, 02, 04, 05, 06, 07, 08, appendix), `paper/main.pdf`, and regenerated `results/tables` and `results/figures` (new: `robustness_extended`, `country_table`, `bootstrap_full_sample`, `placeholders.json`).
