# Plan of the exploratory analyses

**Status.** Written on 2026-10-25, after the single run of the frozen analysis script and after the regime key was opened (`desvios.md`). It is committed before any of the analyses below is computed. Nothing here is confirmatory: no category of section 1 of `preregistro.md` is applied, no p-value is used to claim an effect, and the results go to a section of the dissertation labeled exploratory.

**Rules.**

- Every analysis in this plan is reported, whatever it shows.
- The frozen scripts (`analysis/analise_congelada.py`, `nucleo.py`, `reamostragem.py`, `categorias.py`) are not changed. The new code imports them read-only, checks their SHA-256 against `analysis/README.md` at the start, and lives in `analysis/exploratoria/`.
- The tables of the confirmatory run are not replaced. New outputs go to a separate folder.
- MSP, τ (per run, from `execucoes.csv`), error, flagged pixel (empty set) and AUROC_e are computed as in `nucleo.py`. Regime values are the mean over the 5 seeds, as in the confirmatory tables.
- Any change to this plan after the first computation is added at the end, with its date and reason.

**Inputs.** The margin and label files of the target inference (`run_<code>_s<seed>_<set>.npz`, `labels_<set>.npz`), the regime key, and the confirmatory output tables.

## E1. Per-event plot

A forest plot of Δ_shift (R3) and Δ_reg (R3 − R1) per event with their 90% intervals (`eventos.csv`), next to the pooled value and the share of target pixels of each event. No new estimate: the plot only presents numbers that already exist.

## E2. Unflagged errors by class

Listed as exploratory in section 9 of `preregistro.md`. Errors are split into false water (predicted water, label land) and false land (predicted land, label water). For each regime and for the validation set, the source test, the pooled target and each target event, report the share of each class among all errors and among unflagged errors (MSP ≥ τ), and P(MSP ≥ τ | error, class).

## E3. Distance to the label boundary

Question: how much of the relation between the empty-set rate and the water fraction (RQ3) goes with boundary density instead.

- **Distance.** d is the Chebyshev distance from a scored pixel to the nearest pixel of the other class in the label. Pixels labeled −1 are not counted as either class. For Sen1Floods11 and Bolivia, d is computed within the chip, so chip edges are not boundaries. For T2, d is computed on the whole scene label and then cut into windows. A pixel with no pixel of the other class in its chip or scene goes to the last band. The labels are re-read with the functions of the inference notebook, and the order and values of the scored pixels must match `labels_<set>.npz` exactly before d is used.
- **Bands.** d = 1 or 2 (the `borda` flag), 3 to 5, 6 to 10, 11 to 20, more than 20.
- **Per band**, for each regime and set: share of pixels, error rate, empty-set rate, unflagged error rate and AUROC_e.
- **Per unit** (chip or window), with b = share of pixels with d ≤ 2: the Spearman correlation between empty-set rate and water fraction, and the partial Spearman correlation given b (Pearson correlation of the residuals of the ranks after regressing each on the rank of b). Also the Spearman correlation between water fraction and the empty-set rate of the pixels with d > 10, over units with at least 1,000 such pixels.

## E4. Ensemble of seeds and disagreement between seeds

The pooled ensemble values are already in `resultados.json`. Added here:

- AUROC_e of the ensemble (MSP of the mean probability over the 5 seeds, as in section 9) per event and per regime, and the ensemble minus single-model difference per event.
- Ensemble R3 minus ensemble R1 on the pooled target and per event, with a 90% interval from a bootstrap of units within events (strata as in the primary analysis, B = 10,000, `numpy.random.default_rng(1)`; seeds are not resampled, because the ensemble uses all 5).
- AUROC_e of the variance between seeds of the water probability, used as the error score, per event and per regime.

## E5. Released checkpoint through the inference pipeline of week 6

Question: is the precondition-2 miss on AUROC_e (R0 on Bolivia 0.9432 against an R3 maximum of 0.9398) partly a difference between the week-1 pipeline that produced `analysis/r0_referencia.json` and the week-6 pipeline that produced the R1 to R3 margins?

The released checkpoint (commit `918b9f1`, see `modelo.md`) is run with the reading, preprocessing and 32-bit inference of `notebooks/semana6_inferencia_alvos.ipynb` on the source validation set, the source test and Bolivia. τ is computed on the validation set. Report AUROC_e, unflagged error rate, error rate and mIoU on the source test and on Bolivia next to the values in `r0_referencia.json`, and the difference. The precondition result recorded by the confirmatory run does not change. This needs about one hour of GPU.

## E6. Δ_reg and water fraction within events

Suggested by the confirmatory results themselves: across the 5 target events, the per-event Δ_reg is in the same order as the event water fraction (n = 5). This analysis only generates a hypothesis for a later study and is not a test.

For each chip or window with at least 50 errors and 50 correct pixels in every run of R1 and R3 (the rule of the macro sensitivity analysis), Δ_reg per unit is the mean AUROC_e of the 5 R3 seeds minus the mean of the 5 R1 seeds. Within each event with at least 10 such units, report the Spearman correlation between Δ_reg per unit and the water fraction of the unit, the number of units, and a scatter plot. All events that pass the threshold are reported.

## Not in this plan

- Test-time augmentation is a secondary analysis, not an exploratory one; see `desvios.md`.
- The other exploratory analyses listed in section 9 of `preregistro.md` (mixed logistic model, CKA, fragmentation index, IsoMax arm in R1) are not part of this plan. If any of them is run later, its own plan is committed first.

## Changes to this plan

None yet.
