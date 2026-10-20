# Pre-registration: error detection by a fine-tuned geospatial foundation model under real geographic shift, and the effect of the fine-tuning regime

**Author:** Rosemeri Janiski Bida de Oliveira Borges (Centro de Informática, Universidade Federal de Pernambuco). **Advisor:** Cleber Zanchettin.
**Status:** final text for registration. It is registered with a dated commit of this repository and on OSF before any confirmatory training run. The files it relies on are listed with commit and SHA-256 in section 12.

---

## 1. Research questions and hypotheses

A geospatial foundation model fine-tuned on a few flood events is applied to regions it has not seen, where it makes more errors. The question is whether its confidence still signals where those errors are, and whether the fine-tuning regime changes that.

- **RQ1.** Under real geographic shift, measured with pixel labels, does the ability of the model's confidence to discriminate its errors hold relative to the source, or does it drop?
- **RQ2.** Does the fine-tuning regime (frozen encoder, LoRA, full fine-tuning), with an identical recipe, change that ability in the shifted domain?
- **RQ3 (descriptive).** What does the binary conformal layer add and not add: how much does coverage drop under shift, and how much of the unlabeled empty-set rate is explained by scene composition?
- **RQ4 (descriptive).** In the Rio Grande do Sul flood of 2024-05-06, does the model's confidence agree with imperfect independent references in a way compatible with the labeled events?

RQ3 and RQ4 have no confirmatory hypothesis.

**Margin.** δ = 0.02 in the error-detection AUROC (AUROC_e). Under an equal-variance binormal model at the operating point observed in Bolivia (AUROC_e 0.943, d′ ≈ 2.24), a drop of 0.02 means about 5 percentage points more errors without a signal at a fixed threshold.

**H1 against H2 (RQ1).** Outcome: Δ_shift = AUROC_e(target) − AUROC_e(source test), in the full fine-tuning regime (R3), mean of 5 seeds. Categories, applied in this order to the 90% confidence interval of Δ_shift; the first that applies decides:

1. **H2, preservation:** the interval lies entirely inside (−δ, +δ).
2. **H1, relevant degradation:** upper limit below 0 and point estimate ≤ −δ.
3. **Degradation smaller than the margin:** upper limit below 0 and point estimate between −δ and 0.
4. **Improvement:** lower limit above 0.
5. **Inconclusive:** any other result. The minimum detectable effect obtained is reported.

The source validation and test chips are random draws from the same events as the training chips (Bonafilia et al., 2020), so the source is optimistic. This favors H1. Section 7 shows that, with 89 source test chips, the H2 category is not expected to be reachable.

**H3 (RQ2).** Primary outcome: Δ_reg = AUROC_e(R3) − AUROC_e(R1) on the target, paired by chip, mean of 5 seeds per regime. Three competing predictions: distortion (Δ_reg < 0; Kumar et al., 2022), rigidity (Δ_reg > 0; Lehmann et al., 2026) and null (|Δ_reg| < δ; Hekler et al., 2025). S1, used below, is Δ_reg on the target minus Δ_reg on the source test (section 6). Categories, applied in this order to the 90% confidence interval of Δ_reg; the first that applies decides:

| Order | Category | Condition |
|---|---|---|
| 1 | Null | Interval entirely inside (−δ, +δ) |
| 2 | Distortion, strong | Upper limit < 0, point estimate ≤ −δ, point estimate of S1 < 0, and per-event sign negative in at least ⌈2n/3⌉ of the n target events (4 of 5) |
| 3 | Distortion, weak | Upper limit < 0, without all conditions of the strong category |
| 4 | Rigidity, strong | Lower limit > 0, point estimate ≥ +δ, point estimate of S1 > 0, and per-event sign positive in at least ⌈2n/3⌉ of the n target events (4 of 5) |
| 5 | Rigidity, weak | Lower limit > 0, without all conditions of the strong category |
| 6 | Inconclusive | None of the above: the interval contains 0 and leaves (−δ, +δ). The minimum detectable effect obtained is reported |

The **per-event sign** is the sign of the exact per-event Δ_reg: pooled AUROC_e over the scored pixels of that event, computed per seed, averaged over the 5 seeds of each regime, R3 minus R1. An event where AUROC_e is undefined in any run (no error or no correct pixel) does not meet the sign condition. ⌈2n/3⌉ is two thirds of the events, rounded up: 4 of the 5 primary target events, and the same formula in any sensitivity analysis with another number of events. The S1 condition applies to both strong categories, because both predictions are about what happens under shift. Inconclusive is not a refutation of any prediction.

---

## 2. What is known before registration

- **Released checkpoint (R0).** The fine-tuned checkpoint published by IBM and NASA (`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`, commit `918b9f1`, see `modelo.md`) was evaluated on the source validation and test splits and on Bolivia: AUROC_e 0.937 on the source test, 0.943 on Bolivia, difference +0.006 with 95% interval [−0.039; 0.055]. R0 is not part of any confirmatory test. It is a reproduction anchor and the first input of the analysis script.
- **New regimes.** R1, R2 and R3 have been trained only for a timing check (three epochs per regime), for the learning-rate grid (seed 0, `configs/grade_lr_2026-10-11.md`) and for the seed pilot (R1, seeds 101 and 102, `configs/piloto_sementes_2026-10-13.md`). All of them read only the training and validation splits. No new regime has produced any output on the source test, Bolivia, WorldFloods v2 or Rio Grande do Sul.
- **WorldFloods v2.** Metadata, labels and cloud masks were read to apply the selection criteria (`prereg/t2_criterios.md`) and the window rule (`prereg/t2_janelas.md`), which count labeled pixels. The Sentinel-2 input was read to check the band order and the fraction of pixels without input data. No model has been run on it, so no error or confidence value is known.

---

## 3. Data

| Set | Role | Units |
|---|---|---|
| Sen1Floods11, hand-labeled train | Fine-tuning | 252 chips of 512 × 512 |
| Sen1Floods11, validation | Learning rate, temperature, conformal threshold τ | 89 chips (86 with scored pixels) |
| Sen1Floods11, test | Source evaluation | 90 chips (89 with scored pixels), 10 source events |
| Sen1Floods11, Bolivia hold-out (T1) | Target, event 1 | 15 chips, all with scored pixels |
| WorldFloods v2 test, core set (T2) | Target, events 2 to 5 | 242 windows of 512 × 512: EMSR333 Italy 2, EMSR342 Australia 92, EMSR347 Malawi 141, EMSR438 Tanzania 7 |
| Rio Grande do Sul, 2024-05-06 | Application (RQ4) | Sentinel-2 L1C tiles 22JDM and 22JCN |

Sentinel-2 L1C, six bands (B2, B3, B4, B8A, B11, B12), reflectance × 10,000 scaled by 1e-4 and normalized with the Sen1Floods11 means and standard deviations of TerraTorch, exactly as in training. A pixel is scored if its label is water (1) or land (0); pixels labeled −1 (no data, cloud, unlabeled) are never scored. Chips and windows without any scored pixel are left out.

The **target** is the five events: Bolivia and the four core events of T2.

---

## 4. Models and training

| Regime | Frozen | Trained | Trainable parameters | Learning rate | Role |
|---|---|---|---|---|---|
| R1 | Whole encoder `prithvi_eo_v2_300_tl` | Neck (layers 5, 11, 17, 23), UperNet with 256 channels, head | 15.1 M | 5e-4 | Confirmatory |
| R2 | Encoder base weights | LoRA r = 16, α = 16 on q, v, fc1, fc2, plus neck, UperNet and head | 20.6 M | 5e-4 | Secondary, descriptive |
| R3 | Nothing | Everything | 319.0 M | 5e-5 | Confirmatory |

- Starting point: pretrained backbone `Prithvi_EO_V2_300M_TL.pt`, Hugging Face revision `63adbd39c271da4c42f447e69b1a7c91a338cdc9`, SHA-256 `3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb` (`modelo.md`).
- Recipe of the released checkpoint, identical across regimes except the learning rate: random crop of 224 pixels with horizontal and vertical flips, batch 16, cross-entropy with `ignore_index` −1, AdamW with weight decay 0.05, cosine schedule with `T_max` 50, `head_dropout` 0.1, 16-bit mixed precision. 50 fixed epochs, no early stopping; the model at epoch 50 is evaluated. TerraTorch 1.2.11, torchgeo 0.7.1, peft 0.21.2 (`configs/ambiente_regimes_2026-10-10.json`).
- Learning rates chosen on the validation split by validation mIoU only, with the rule in `prereg/piloto_lr.md`.
- **Seeds 1 to 5** in every regime (`seed_everything(seed, workers=True)`). The pilot seeds (0, 101, 102) are never reused. No seed is added after any target output exists.
- **Failed runs.** A run fails if its loss becomes non-finite or if its validation mIoU at epoch 50 is below 0.80 (the grid and the pilot gave 0.886 to 0.900). A failed run is repeated once with the same seed; if it fails again, seed 6 replaces it, and so on. Failures are decided before any target inference and are reported.
- Validation mIoU is logged every 5 epochs, for precondition 3 (section 8).

---

## 5. Inference and measured variables

Inference runs in 32-bit precision on whole 512 × 512 chips and windows, for every run, on validation, source test, Bolivia, the T2 core and full sets (primary and sensitivity windows), the T2 maps added by the cloud sensitivity (section 9) and Rio Grande do Sul. The windows of the maps added by the cloud sensitivity are cut with the rule in `prereg/t2_janelas.md` and committed before any inference. For each scored pixel, the logit margin m = z_water − z_land is saved.

- **Prediction:** water if m > 0. **Error:** prediction different from the label.
- **MSP:** σ(|m|). **Score:** s = 1 − p(true class).
- **AUROC_e (primary metric):** area under the ROC curve for error detection, errors as the positive class and 1 − MSP as the score, pooled over all scored pixels of the set. Ranked by |m|; ties count one half.
- **Conformal threshold:** for each run, q̂ is the ⌈(n + 1)(1 − α)⌉/n quantile (method `higher`) of s over the scored validation pixels, α = 0.10; τ = 1 − q̂. In the binary case with τ > 0.5 the conformal set is a singleton or empty, and the empty set is equivalent to MSP < τ.
- **Unflagged error rate:** P(MSP ≥ τ | error).
- **Silent failure at τ:** P(error and MSP ≥ τ). **AUGRC:** area under the generalized risk-coverage curve (Traub et al., 2024), that is, the integral over coverage of the joint probability of an error among the accepted pixels, with pixels accepted in decreasing order of |m|. At each coverage, the generalized risk is the number of accepted errors divided by the number of all scored pixels.
- **Coverage, empty-set rate, E-AURC.**
- **Calibration:** signed ECE with 15 equal-mass bins, NLL and Brier, before and after temperature scaling. One temperature per run, fitted on the validation split by NLL. The temperature enters only these calibration metrics: in the binary case, with τ recalibrated after temperature, the conformal set, the unflagged error rate and AUROC_e are invariant to it.
- **Accuracy:** error rate and mIoU (mean over the two classes of the intersection over union, pooled over pixels).

For each regime and set, a metric is first computed for each seed and then averaged over the seeds. The ensemble of the seeds is a separate baseline (section 9).

---

## 6. Analysis of the primary and key secondary outcomes

**Point estimates.** Exact values from all scored pixels, per seed, then averaged over seeds.

**Interval estimates.** Bootstrap with B = 10,000 replicates and random generator seed 0. In each replicate:

1. Within each target event, the chips (Bolivia) or windows (T2) are resampled with replacement. The events are strata and are not resampled. The source test chips are resampled within their source event (the location prefix of the chip name).
2. Within each regime, the 5 seeds are resampled with replacement, independently between regimes. The seed draw of a regime is the same for every set.
3. Per resampled seed, AUROC_e and AUGRC are computed over the resampled chips from per-chip histograms of |m| (bins of width 0.005 from 0 to 40, plus one bin above), separately for errors and correct pixels. The unflagged error rate is computed from per-chip counts, with each run's τ kept fixed, because the validation split is not resampled.
4. The regime value is the mean over the resampled seeds. The same chip draw is used for all regimes, so Δ_reg is paired by chip.

**Histogram check.** If, in any run and set, the histogram AUROC_e differs from the exact value by more than 1e-4, the upper limit of the bins is raised to the maximum |m| over all runs, rounded up to a multiple of 5, for every computation. If the difference still exceeds 1e-4, the bin width is halved until it does not.

**Intervals and p-values.** Percentile intervals: the 90% interval decides, the 95% interval is reported. The p-value of a difference is p = 2 · min(share of replicates ≤ 0, share ≥ 0), with each share computed as (count + 1)/(B + 1) and p capped at 1, so intervals and p-values come from the same replicates. With 5 seeds, resampling seeds understates the variance between seeds by a factor of 4/5; this is accepted and stated.

**Aggregation across events.** The primary estimand pools all scored target pixels: each pixel weighs the same, so EMSR347 Malawi (about two thirds of the T2 pixels) and EMSR342 Australia weigh most, and Bolivia, Italy and Tanzania weigh little. Consistency across events enters the strong categories of H3 (section 1), and the per-event estimates are always reported (section 9).

**Why the events are strata and not resampled.** Five events of very unequal size (one with most of the pixels, another with two windows) give unstable intervals when the events themselves are resampled. The conclusions are therefore conditional on these five events. Because neighbouring windows of one scene are not independent, a bootstrap of spatial blocks and a bootstrap that also resamples events are run as sensitivity analyses (section 9). **If either of them changes the category of H1 against H2 or of H3, both categories are reported in the abstract and in the conclusions.** This replaces the sentence on the bootstrap in `prereg/t2_criterios.md` ("The bootstrap resamples events first and chips within events"), which was written before the aggregation was decided; `prereg/t2_janelas.md` left this decision to this document.

**Key secondary outcomes, with Holm correction across the three at a family-wise level of 0.05:**

- **S1, specificity to shift:** Δ_reg on the target minus Δ_reg on the source test.
- **S2, unflagged error rate:** difference R3 − R1 on the target, margin 5 percentage points.
- **S3, AUGRC:** difference R3 − R1 on the target.

For each, the estimate and the 90% and 95% intervals are reported. A claim that the difference is not zero requires a Holm-adjusted p-value below 0.05. For S2, a claim of equivalence within ±5 percentage points requires the 90% interval to lie inside (−5, +5) percentage points; this equivalence claim is not adjusted for multiplicity and is reported as such.

---

## 7. Sample size and minimum detectable effect

Fixed: 5 seeds per regime, 257 target units in 5 events, 89 source test chips. There is no stopping rule and no data are added after registration.

**Paired contrasts between regimes (H3, S2).** The minimum detectable effect (MDE) with 80% power is

MDE = (z₁₋α/₂ + z₀.₈₀) · √(2 · SE² · (1 − ρ) + 2σ² / k),

with SE the chip-level standard error of one model on the target, ρ the correlation between two models under chip resampling, σ the standard deviation between seeds and k = 5 seeds. Inputs: SE from Bolivia under R0 (0.0115 in AUROC_e; 3.9 percentage points in unflagged error rate); ρ and σ from the seed pilot on the validation split (`configs/piloto_sementes_2026-10-13.md`): ρ for AUROC_e between 0.73 and 0.81 and σ = 0.0041; ρ for the unflagged error rate between 0.76 and 0.86 and σ = 2.0 percentage points.

| Outcome | MDE, two-sided α of 5% | MDE with the 90% interval of the decision rule |
|---|---|---|
| Δ_reg in AUROC_e (ρ 0.73 to 0.81) | 0.021 to 0.025 | 0.019 to 0.022 |
| S2, unflagged error rate (ρ 0.76 to 0.86) | 6.8 to 8.3 pp | 6.0 to 7.4 pp |

Consequences for H3, stated before any result (ρ = 0.77 unless a range is given):

- If the true Δ_reg is zero: null in 46% to 69% of repetitions (57%), a directional category (almost always weak) in 10%, inconclusive in the rest (21% to 44%; 33%).
- If the true Δ_reg is −0.02: the interval excludes zero in 73% to 84% of repetitions (79%), but the point estimate reaches −δ only half the time, so a strong category is reached at most about half the time. At −0.025, the point estimate reaches −δ in about 73% of repetitions. The same holds, with the signs reversed, for rigidity.
- S2: the first Holm step needs an effect of 7.8 to 9.6 percentage points for 80% power. If the true difference is zero, the 90% interval falls inside ±5 percentage points in 3% to 33% of repetitions. The margin stays at 5 percentage points because it is the translation of δ into unflagged errors; smaller differences will tend to be inconclusive, and the result is reported in any case.

**H1 against H2 (unpaired, target against source).** From the R0 intervals, the SE of AUROC_e is about 0.020 on the source test and 0.011 on Bolivia, and the interval of their difference corresponds to an SE of about 0.023 to 0.024. With the five target events, the SE of Δ_shift is expected to be about 0.021 to 0.023, dominated by the 89 source chips. The half-width of the 90% interval is then 0.035 to 0.038, larger than δ, so **the H2 category is not expected to be reachable**: in practice this test can show a degradation, not a preservation. A degradation is detectable with 80% power only if it is about 0.052 to 0.057 or larger (90% interval). The analysis script computes the stratified SE on R0 before any target output exists, and that value is reported. The categories are kept as stated in section 1.

The inputs carry uncertainty: ρ was measured on the source, with R0 in place of a retrained R3; SE comes from Bolivia alone; σ comes from two seeds. The effective MDE is reported with the results.

---

## 8. Preconditions, reported without invalidating the tests

1. α = 0.10 is larger than the validation error rate in every run.
2. On Bolivia, the R0 values of AUROC_e, unflagged error rate, error rate and mIoU fall within the range (minimum to maximum) of the 5 R3 seeds. Each metric is reported separately.
3. The source test mIoU of R1 and R3 (seed means) differ by at most 1.5 percentage points. If not, a sensitivity analysis re-trains each seed of the stronger regime with the same seed, recipe and validation schedule, stopped at the logged epoch (a multiple of 5) at which that seed's validation mIoU is closest to the seed mean of the final validation mIoU of the weaker regime; if two epochs are equally close, the later one. The primary analysis is repeated with those models, and the validation mIoU actually reached by each retrained run is reported; there is no second retraining. The primary result remains the one at epoch 50.

---

## 9. Secondary, sensitivity and exploratory analyses

**Secondary, descriptive:** Δ_shift in R1 and R2 as well; R2 against R1 and against R3; E-AURC, coverage and empty-set rate; ECE, NLL and Brier before and after temperature; mIoU and error rate; ensemble of the 5 seeds (MSP of the mean probability, and variance between seeds) and test-time augmentation with the 8 dihedral transforms, against single-model MSP; constant confidence (AUROC_e = 0.5) and "always land" as sanity baselines.

**Sensitivity analyses, announced now.** For each, the estimate, its 90% interval and the category it would give are reported, except for the per-event estimates, which get estimate and interval only (Italy has two windows, so its bootstrap is nearly degenerate):

- Per-event estimates of Δ_reg and Δ_shift, with 90% intervals from a chip bootstrap within the event.
- Equal weight per event: mean of the five per-event estimates.
- AUROC_e from within-event pairs only: per-event AUROC_e combined with weights proportional to the number of error × correct pixel pairs of each event, which removes pairs across events.
- Spatial block bootstrap: the windows of each T2 scene are grouped into blocks of 2 × 2 by their row and column rank in the scene's window grid (`splits/t2_janelas.csv`), counted from the top-left; incomplete blocks are kept; blocks are resampled within the event. Bolivia chips stay as units.
- Bootstrap that also resamples events (two stages: events, then chips within events).
- Leave one event out: the range of Δ_reg and whether the category changes. Leave one chip out: the range of Δ_reg.
- AUROC_e as the mean over chips (macro), over the chips with at least 50 errors and 50 correct pixels in every run.
- Exclusion of water and land pixels within 2 pixels of a water–land boundary of the label; pixels labeled −1 do not form boundaries.
- T2 sensitivity windows (at least 1 scored pixel instead of 10%).
- T2 with the cloud criterion at 0.50 instead of 0.30: adds EMSR286 (two maps) and EMSR347 Zomba on 2019-03-12.
- T2 full set (criteria C1 and C3, with or without C2: 12 maps in 8 events), descriptive.

**Exploratory, outside any correction:** a mixed logistic model of error pixels; CKA between the pretrained and the fine-tuned encoder against AUROC_e; unflagged errors by class (false water against false land); the fragmentation index of Kolluru et al. per chip; an IsoMax-loss arm in R1 with 3 seeds, run only after the confirmatory analysis.

**RQ3 and RQ4:** coverage on each set against 1 − α; per-chip relation between the empty-set rate and the scene composition (water fraction); in Rio Grande do Sul, agreement conditioned on confidence against a consensus mask of the available references, with the selection bias stated. No statement about the model's errors in Rio Grande do Sul is made.

---

## 10. Analysis script, blinding and deviations

- **Before any target output exists**, the analysis script is written and tested in two ways: on synthetic margins with known AUROC_e, covering the multi-seed path and the T2 and Bolivia strata (histogram against exact AUROC_e within 1e-4); and on R0, where it must reproduce the source test and Bolivia values of AUROC_e and unflagged error rate to the third decimal. Then it is frozen, and its SHA-256 is recorded in `analysis/README.md`.
- **Blinding.** The inference notebook stores target outputs under regime codes drawn at random, and the code key is written to a separate file. The author holds the key and does not open it before the single run of the frozen script, which reads the key only to label its final tables. The blinding is procedural, and this is stated.
- **Errors found after the frozen script has run on target outputs.** Only a fix that makes the script do what this document defines is allowed (for example, a crash or a metric computed differently from section 5). Each fix goes to `prereg/desvios.md` with date, the change and the SHA-256 of the corrected script; the results before and after the fix are both reported, and they stay confirmatory only if the fix restores a definition given here. Any other change is exploratory.
- Any other change after registration goes to `prereg/desvios.md`, with date, what changed and why. An analysis changed after target outputs exist is labeled exploratory.
- If R2 cannot be trained in the environment, it is dropped; the primary outcome does not depend on it.

---

## 11. Data and code availability

Code, configurations, chip and window lists, seeds, the analysis script and this document are public in this repository, archived with a DOI on Zenodo. The logit margins of every run are shared. Fine-tuned weights are published on the Hugging Face Hub. Sen1Floods11 is not redistributed: the lists point to its public bucket. WorldFloods v2 is licensed CC BY-NC 4.0, so derived windows are shared only under that licence, with attribution.

---

## 12. Files fixed before registration

| File | Commit | SHA-256 (first 12) |
|---|---|---|
| `splits/sen1floods11_train.txt` | `4228134` | `4d77d923fced` |
| `splits/sen1floods11_val.txt` | `f8e76b6` | `14a372859fb1` |
| `splits/sen1floods11_test.txt` | `f8e76b6` | `cfcf04cc3688` |
| `splits/sen1floods11_bolivia.txt` | `2197355` | `d3c9e2d98e9f` |
| `splits/t2_eventos.csv` | `3b4e931` | `d0cd80324207` |
| `splits/t2_janelas.csv` | `57294b9` | `3c21970835f2` |
| `prereg/t2_criterios.md` | `703c7a3` | `4284f944a5b9` |
| `prereg/t2_janelas.md` | `7ab6806` | `3932c414e3de` |
| `prereg/piloto_lr.md` | `8c68bee` | `2bf64c72fa1e` |
| `prereg/piloto_sementes.md` | `8e90879` | `3b006f56565a` |
| `configs/ambiente_regimes_2026-10-10.json` | `41ac168` | `8eb2ba2bb02c` |
| `configs/grade_lr_2026-10-11.md` | `e12469b` | `3c39e1b94b19` |
| `configs/piloto_sementes_2026-10-13.md` | `a1633d4` | `ee841114b275` |
| `modelo.md` | `173aea5` | `d9f88a269617` |

The commit of this file and its SHA-256 are recorded on OSF.
