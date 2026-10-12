# Pilot: two seeds of the frozen encoder against the released checkpoint, validation set only

Fixed on 2026-10-12, before any run of this pilot. Follows section 14.4 of the project plan.

## Purpose

Estimate two inputs of the minimum detectable effect in section 14.4: how strongly the error-detection AUROC of two models moves together when the same chips are resampled, and how much it varies between seeds of one regime. Nothing in this pilot tests a hypothesis, and no choice of recipe or learning rate depends on it. Every value it produces is reported.

## What runs

- **R1 (frozen encoder), seeds 101 and 102.** These seeds are used only in the pilot and are never reused as confirmatory runs; the confirmatory seeds stay 1 to 5. Learning rate 5e-4, the value chosen by the grid (`configs/grade_lr_2026-10-11.md`). Everything else as in `prereg/piloto_lr.md`: same backbone file, recipe, 50 fixed epochs, batch 16, 16-bit mixed precision in training, `seed_everything(seed, workers=True)`, the checkpoint at epoch 50, the same environment.
- **R0 (released checkpoint).** Not rerun. Its validation scores are the ones saved on 2026-06-07 (`cal_scores.npy`, `cal_labels.npy`, `cal_chip_sizes.npy`, `cal_chip_names.npy`), the same files used in week 1.

## Data

Only the 89 chips of `splits/sen1floods11_val.txt`, at 512 × 512 pixels. A pixel is scored if its label is 0 or 1. Chips without any scored pixel are left out. The test set, Bolivia and WorldFloods v2 are not read.

## Inference for R1

The model is evaluated in 32-bit precision on whole chips, with the same scaling and normalization as in training. For every scored pixel the logit margin m = z_water − z_land is saved.

## Definitions

- Prediction: water if m > 0. Error: prediction different from the label.
- Maximum softmax probability (MSP): σ(|m|). For R0, which has only the saved score s = 1 − p(true class), MSP = max(s, 1 − s) and error means s > 0.5.
- AUROC_e: area under the ROC curve for error detection, errors as the positive class and 1 − MSP as the score, pooled over all scored pixels. Ties count one half. R1 is ranked by |m|, which keeps the order where MSP rounds to 1; R0 is ranked by its saved score.
- Conformal threshold: q̂ is the ⌈(n + 1)(1 − α)⌉/n quantile (method `higher`) of s over the n scored validation pixels, α = 0.10, and τ = 1 − q̂. In this pilot τ is computed and applied on the same validation pixels, so it is in-sample.
- Unflagged error rate: the share of errors with MSP ≥ τ.
- mIoU: mean over the two classes of the intersection over union, pooled over all scored pixels.

## Estimates

1. **Correlation between models (primary).** Chip bootstrap with B = 2,000 draws and random generator seed 0. Each draw resamples the chips with replacement and uses the same draw for every model. In each draw the pooled AUROC_e of each model is computed from per-chip histograms of the confidence (bins of width 0.005 in |logit margin|, from 0 to 40, plus one bin above; for R0 the margin is recovered from its saved score), and the unflagged error rate from per-chip counts, with τ kept at its value on all chips. The estimate is the Pearson correlation of the 2,000 replicates of two models. Pairs: R1 seed 101 with R0, R1 seed 102 with R0, the mean of the two R1 seeds with R0, and R1 seed 101 with R1 seed 102. The standard deviation of the replicates is reported as the bootstrap standard error of each model.
2. **Correlation of per-chip values (secondary).** Pearson and Spearman correlation of the per-chip AUROC_e, over the chips with at least 50 errors and at least 50 correct pixels in every model.
3. **Spread between seeds.** For AUROC_e, the unflagged error rate and mIoU: |x₁ − x₂| / √2 from seeds 101 and 102. With two seeds this has one degree of freedom and is a rough figure.

## Checks reported with the results

- The pooled AUROC_e from the histograms against the exact value from all pixels.
- The mIoU of the 32-bit inference against the `val/mIoU` logged by the trainer at epoch 50.
- The scored pixels of each validation chip in the current label files against those in the saved R0 labels.
- Precondition 1 of section 14.8: α larger than the validation error rate, for every model.

## Records

`resultados/semana4/` on Google Drive: `piloto_R1_s101.json` and `piloto_R1_s102.json` (run record), `margens_R1_s101.npz` and `margens_R1_s102.npz` (chip keys, scored pixels per chip, margins and labels), the trainable weights of each run, and `piloto_sementes.json` (all estimates and checks).
