# Plan of the exploratory analyses, part 2

**Status.** Written on 2026-11-19, written after the single run of the frozen analysis script, after the opening of the regime key and after the exploratory analyses E1 to E6 (`plano_exploratorio.md`). It is committed before any of the analyses below is computed. These are three of the exploratory analyses listed in section 9 of `preregistro.md` and left out of `plano_exploratorio.md`. Nothing here is confirmatory: no category of section 1 of `preregistro.md` is applied, no p-value is used to claim an effect, and the results go to a section of the dissertation labeled exploratory. The IsoMax arm, also listed in section 9, is not part of this plan.

**Rules.** The rules of `plano_exploratorio.md` apply unchanged:

- Every analysis in this plan is reported, whatever it shows.
- The frozen scripts (`analysis/analise_congelada.py`, `nucleo.py`, `reamostragem.py`, `categorias.py`) are not changed. The new code imports them read-only, checks their SHA-256 against `analysis/README.md` at the start, and lives in `analysis/exploratoria/`.
- The tables of the confirmatory run and of E1 to E6 are not replaced. New outputs go to a separate folder (`resultados/semana11/`).
- MSP, τ (per run, from `execucoes.csv`), error, flagged pixel (empty set) and AUROC_e are computed as in `nucleo.py`. Regime values are the mean over the 5 seeds.
- Any change to this plan after the first computation is added at the end, with its date and reason.

**Inputs.** The margin and label files of the target inference (`run_<code>_s<seed>_<set>.npz`, `labels_<set>.npz`), the regime key, the confirmatory output tables, the trained weights of the 15 confirmatory runs and the pinned pretrained backbone (`Prithvi_EO_V2_300M_TL.pt`, SHA-256 in `configs/ambiente_regimes_2026-10-10.json`). Sets: source test (`teste`), Bolivia and the T2 core windows. The cloud 0.50 maps and the full set are not used.

## E7. CKA between the pretrained and the fine-tuned encoder

Question: how far does each regime move the encoder representations away from the pretrained ones, and does that distance go with AUROC_e on the target?

- **Representations.** For each model, the token outputs of the encoder blocks at indices 5, 11, 17 and 23 (the four blocks that feed the neck), on whole 512 × 512 inputs, with the preprocessing and 32-bit inference of `notebooks/semana6_inferencia_alvos.ipynb`. Only patch tokens enter; the class token, if the encoder has one, is dropped (time and location are added to the tokens in the TL variants, not carried as separate tokens). Each patch token is one row.
- **Inputs.** All chips of the source test (89 with scored pixels) and all target units (15 Bolivia chips and the 242 T2 core windows). CKA is computed separately on the source test and on the pooled target, and per target event.
- **Measure.** Linear CKA (Kornblith et al., 2019) between the pretrained encoder and the fine-tuned encoder of the same run, on the same inputs, with features centered over all tokens of the set. Computed from the feature cross-covariance, so no token subsampling is needed.
- **Runs.** The 5 R2 and 5 R3 runs. In R1 the encoder is frozen, so CKA is 1 by construction; it is computed for one R1 run only as a check of the code, and the run passes if CKA is at least 0.9999 at every layer.
- **Report.** CKA per run, layer and set; mean and range over the 5 seeds of each regime. Per run, the AUROC_e on the pooled target (recomputed from the margins with `nucleo.py`; the regime mean must match `regimes.csv` to the 4th decimal). The Spearman correlation between CKA at layer 23 on the target and AUROC_e on the target over the 10 runs of R2 and R3, and a scatter plot with the 15 runs (R1 at CKA = 1). With 10 runs this is descriptive only.
- **Cost.** About 1 hour of T4 GPU (11 encoders, 346 inputs each).

## E8. Mixed logistic model of error pixels

Question: among pixels the model gets wrong, does the chance that the error goes unflagged (MSP ≥ τ) change with the regime and with the shift, once the error class, the distance to the label boundary and the clustering of pixels in units and events are taken into account?

- **Rows.** Error pixels of the source test and of the target, for R1, R2 and R3, seeds 1 to 5. In each run and unit (chip or window), at most 200 error pixels are drawn without replacement with `numpy.random.default_rng(2)`, in the order run, set, unit; units with fewer errors enter whole. Each row carries a weight equal to the number of errors of its run and unit divided by the number drawn, used in a sensitivity fit.
- **Outcome.** Unflagged error: 1 if MSP ≥ τ of the run, 0 otherwise.
- **Fixed effects.** Regime (R1 reference, R2, R3); domain (source test reference, target); regime × domain; error class (false water, false land; from E2); distance band to the label boundary (d ≤ 2, 3 to 10, more than 10; d as in E3); water fraction of the unit, standardized.
- **Random effects.** Intercept per unit nested in event; intercept per run.
- **Fit.** `statsmodels` `BinomialBayesMixedGLM` (variational Bayes), with the default priors. Report the posterior mean and the 95% interval of each fixed effect as an odds ratio, and the standard deviation of each random effect.
- **Sensitivity.** (a) A GEE logistic model with the same fixed effects and exchangeable correlation within unit; (b) the main model with the row weights; (c) the main model on the target only, with event as a fixed effect in place of domain.
- **Report.** All fits, with the number of rows, units and events. No coefficient is called significant.
- **Cost.** CPU only; a few hours on a 12 GB machine (estimate).

## E9. Fragmentation index per unit

Question: does the fragmentation index of Kolluru et al. (2026) flag bad units (chips or windows with high error rate) as well as the mean MSP of the unit does?

- **Definition.** Kolluru et al. (2026, arXiv 2606.07780, Results, "Patch-Level Spatial Analysis", and Methods, "Stratified Analysis and Accuracy Metrics") define the fragmentation index as the number of connected flood patches of the predicted map per km² of predicted flood area ("patches per km² of flood area"), computed per event. The paper gives no formula beyond that phrase and does not state the connectivity rule, a minimum patch size, or how no-data pixels enter the area. Choices fixed here: patches are the 8-connected components of predicted water; there is no minimum patch size; area is the number of predicted water pixels times 100 m² (10 m pixels), in km². The index is computed per unit (chip or window) and, as in the paper, per event (all units of the event pooled, patches counted within each unit). A unit with less than 0.1 km² of predicted water (1,000 pixels) gets no index and is counted and reported. Sensitivity: 4-connected components.
- **Predicted mask.** Water where the margin is positive. The scored pixels are placed back on the unit grid by re-reading the labels with the functions of the inference notebook, as in E3, and the order and values must match `labels_<set>.npz` exactly. Unscored pixels (label −1, cloud, no data) are not water in the mask and do not join patches; the paper does not say how it treats them, and this choice is reported.
- **Bad unit.** A unit whose error rate is in the top quarter of its set (source test or target), per run.
- **Report.** Per regime (mean over seeds) and set: AUROC of the fragmentation index and AUROC of 1 − mean MSP of the unit for detecting bad units; their difference with a 90% interval from a bootstrap of units within events (B = 10,000, `numpy.random.default_rng(3)`, seeds not resampled); the Spearman correlation between the two scores.
- **Cost.** CPU only; under an hour.

## Not in this plan

- The IsoMax arm in R1 (section 9 of `preregistro.md`).
- Any new event, sensor or model.

## Changes to this plan

None yet.
