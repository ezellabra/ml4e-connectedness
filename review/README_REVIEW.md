# Reviewer guide

This folder contains everything needed to read and review the paper.

| File | What it is |
|---|---|
| `00_full_paper.pdf` | The whole paper as it stands now |
| `01_base_paper_summary_motivation_introduction.pdf` | Summary of the base paper, Motivation, Introduction |
| `02_data.pdf` | Section 3, Data |
| `03_methods.pdf` | Section 4, Methods |
| `04_results.pdf` | Section 5, Results (text, figures and the comparison table) |
| `05_discussion_conclusion.pdf` | Sections 6 and 7 |
| `06_references_data_and_code_availability.pdf` | References and Data/Code availability |
| `07_appendix_robustness.pdf` | Appendix: robustness tables and figure |

A section PDF can start or end on a page shared with its neighbour, so a few lines of the adjacent section may appear at its edges. The page limit for the final version is 20; the paper currently has 14 pages.

## What the paper is about, in five lines

We measure how shocks propagate across 17 emerging equity markets (plus the S&P 500 as a global reference) using weekly USD returns of US-listed country ETFs, March 2011 to October 2026. We compare four ways of estimating the network: Diebold-Yilmaz (a VAR plus a forecast-error variance decomposition, the base paper), a sparse VAR estimated with the lasso, NETS (sparse Granger and partial-correlation networks) and FNETS (the same after removing a global common factor). Methods are compared on network density, agreement between methods and stability over time. A key question is how much of the measured connectedness disappears once the common factor is removed.

Short glossary: *total connectedness* is the average share (in %) of a market's forecast-error variance explained by shocks to the other markets; *net connectedness* is what a market transmits minus what it receives; a *partial correlation* is the correlation between two markets after controlling for all the others; the *common component* is the part of each return explained by a few factors shared by all markets, and the *idiosyncratic component* is the rest.

## What to check in each section

1. **Summary of the base paper, Motivation, Introduction.** The summary is a stand-in that will be replaced by the one-page summary already written; Motivation and Introduction are still open. Nothing to verify yet.
2. **Data.** Check the numbers against `results/tables/descriptive_stats.csv`, `results/tables/extreme_returns.csv` and `docs/data_notes.md`: 816 weeks, five returns above 25% in absolute value, 8 of the 10 largest pooled weekly drops in March 2020, 18.2% significant VAR coefficients (59 of 324). Check that the limitations listed are fair.
3. **Methods.** Check the formulas and the parameters against `docs/decisions.md` (the reasoning behind every choice) and `config.yaml` (the values: window 104 weeks, horizon 4 weeks, lag cap 2 for DY and 4 for the lasso, five CV folds, and so on). Check that every method is described clearly enough to be reproduced.
4. **Results.** Every number in the text should match a table in `results/tables/`:
   - full-sample and subsample totals: `static_connectedness.csv`
   - densities: `density_base_H4.csv`
   - agreement between methods: `agreement_base_H4.csv`
   - stability: `stability_base_H4.csv`
   - share of connectedness lost without the common factor, common-variance share: `common_factor_base_H4.csv`
   - robustness scenarios: `robustness_summary.csv`
   - net transmitters and receivers (averages over windows): `series_base_H4.csv`, columns `net_*`

   Also check that each claim is worded no more strongly than the table supports, and that the figure captions describe what the figures show.
5. **Discussion, Conclusion.** Still open; the evidence they should draw on is listed in `docs/summary.md`.
6. **References, Data and code availability.** Check the bibliography entries (`paper/references.bib`). The professor's papers on the idiosyncratic component are still to be added; the repository link is still to be filled in.
7. **Appendix.** Check that the robustness tables match `results/tables/robustness_summary.csv` and `results/tables/static_connectedness.csv`.

Passages in small gray italics ("To be completed: ...") are placeholders for work still in progress.

## How to send comments

One line per comment: *section, page, the sentence or number concerned, the problem or suggestion*. For numbers, please say which table you compared against.
