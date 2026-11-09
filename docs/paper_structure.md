# Paper structure (summary of "Estructura del paper v2", 27 Oct 2026)

Source: the internal planning document "Estructura del paper v2" (Spanish, by Marco; kept locally, not in the repository). This is an English summary used as the reference for every later phase.

## Constraints that do not change

- Deadline: 12 November, 18:00. Maximum 20 pages, in English.
- Each member must register in AlmaEsami before 5 November.
- Data section separate from Methods.
- Explicit comparison criteria between methods: **network density**, **agreement between methods**, **stability over time**.
- Robustness, plus a link to the code and data in the final submission.
- Before submission someone checks the text against the numbers (to avoid repeating the errors of the reference paper).
- Discussion and conclusions are the group's own work (course policy allows AI for code, not for that component).
- Weekly frequency is fixed. Justification (from the professor): with weekly data something can be forecast, with daily data not; used in Data and Methods.

## Sections and page budget (19.5 of 20 pages)

| Section | Content | Pages |
|---|---|---|
| Title page | New title without "Latin America", about emerging markets | 1 |
| Summary of base paper | Diebold-Yilmaz (2014) and a fuller description of NETS and FNETS, plus the professor's papers on the idiosyncratic component (to request) | 1 |
| 1. Motivation | Why connectedness among emerging markets matters. Argentina as a case within the universe (whether it remains the focus is open) | 1 |
| 2. Introduction | Question: how measured connectedness changes with the method (DY, NETS, FNETS) in emerging markets. Final wording and hypotheses: group | 1 |
| 3. Data | Which/how many ETFs, period, weekly frequency and justification, returns vs volatility, limitations (composition, liquidity, young ETFs) | 2 |
| 4. Methods | Three methods with equal weight: DY (VAR + generalized FEVD), NETS, FNETS (common and idiosyncratic). Lasso VAR as optional intermediate step. Window, lags, horizon, lambda choice and comparison criteria | 5 |
| 5. Results | Full-sample network per method, connectedness over time with shaded events, net transmitters/receivers, comparison table. Out-of-sample weekly forecasting as extra criterion (to confirm with professor) | 4 |
| 6. Discussion | FNETS separates global-factor vs idiosyncratic connectedness. Interpretation, limitations, critical analysis: written by the group | 2 |
| 7. Conclusion | Answer the question, state what can be claimed safely | 0.5 |
| 8. References | NETS, FNETS and the professor's papers | 0.5 |
| 9. Data and code availability | Repository link and language | 0.5 |
| Appendix | Robustness: other window, other horizon, local indices instead of ETFs. Suggestion: Latin American subset of the universe | 1 |

Page-budget open issue: a corrections document suggested moving one page from the Appendix to Methods; that would remove the robustness appendix (still required). Alternative under discussion: Methods 5.5 pages using the free half page, Appendix stays at 1.

## Open decisions in the document (all marked "to be defined") and how this repository resolves them

| # | Decision | Options in the document | Status in this repo |
|---|---|---|---|
| 1 | Emerging-market ETFs | More than the current 5, with enough liquidity | Fixed by the brief: 17 EM ETFs (see `docs/data_notes.md`) |
| 2 | Sample window | From the youngest ETF, in weeks | Fixed by the brief: 2011-03-11 to 2026-10-23 (816 weeks) |
| 3 | Returns or volatility | Weekly returns or weekly volatility | Returns (brief; no daily data for realized volatility). See `docs/decisions.md` |
| 4 | US as reference node | Keep SPY or drop it | Kept in baseline; "excluding SPY" run as robustness |
| 5 | Lasso VAR | Intermediate step within FNETS, or dropped | Implemented as a separate method (sparse VAR, Phase 3); FNETS also uses a lasso VAR internally for the idiosyncratic part |
| 6 | Argentina | Focus or one case within the universe | Treated as a highlighted case within the universe (highlighted in figures); final framing left to the authors |
| 7 | Research question | Group's wording | Working version from the brief; final wording left to the authors |

## Conflicts / differences with the project brief

1. **Methods weighting.** The document asks for three methods with equal weight (DY, NETS, FNETS) and the lasso VAR as an *optional* intermediate step. The brief makes the sparse VAR a full fourth method. Resolution: the code implements four methods; the paper skeleton presents DY, NETS and FNETS as the core and the lasso VAR as a bridge subsection, so the authors can drop it without restructuring.
2. **Robustness content.** The document's appendix lists "local indices instead of ETFs" and a "Latin American subset". The brief's robustness list is windows 78/130, H = 2/8, excluding SPY, excluding 2020. Local indices are not available in the repository (not implemented; listed as missing). The Latin-American subset is not in the brief; not implemented, listed as a possible extension in `docs/summary.md`.
3. **Section numbering.** The document numbers References (8) and Data and code availability (9) as sections and puts a "summary of the base paper" before Motivation. The LaTeX skeleton follows the document's order.
4. **Out-of-sample forecasting** is "to confirm with the professor" in both; it is implemented as an optional module, off by default.
5. **Professor's papers on the idiosyncratic component** were "to request" in the document; the group decided not to add them (9 Nov). The bibliography covers DY, NETS, FNETS and the methods used.
