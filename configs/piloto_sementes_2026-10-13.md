# Pilot: two seeds of the frozen encoder against the released checkpoint (2026-10-13)

Rule fixed before the runs: `prereg/piloto_sementes.md` (commit `8e90879`). Notebook: `notebooks/semana4_piloto_sementes.ipynb` (commit `ca17c37`), Colab T4. Validation set only: 89 chips, 86 with scored pixels, 20,294,725 scored pixels. The test set, Bolivia and WorldFloods v2 were not read.

## Runs

| Model | Seed | val/mIoU logged in training (16-bit) | mIoU, 32-bit inference | Minutes | Peak GPU memory |
|---|---|---|---|---|---|
| R1 (frozen encoder, lr 5e-4) | 101 | 0.8908 | 0.8908 | 9.4 | 4.7 GB |
| R1 (frozen encoder, lr 5e-4) | 102 | 0.8925 | 0.8925 | 9.3 | 4.7 GB |
| R0 (released checkpoint) | n/a | n/a | 0.8964 (saved scores) | n/a | n/a |

## Validation metrics

τ is computed and applied on the validation pixels themselves (α = 0.10), so the unflagged error rate is in-sample. Bootstrap standard errors resample chips (B = 2,000, generator seed 0).

| Model | Error rate | mIoU | AUROC_e | Bootstrap SE of AUROC_e | τ | Unflagged error rate |
|---|---|---|---|---|---|---|
| R1, seed 101 | 0.0237 | 0.8908 | 0.9602 | 0.0044 | 0.9101 | 0.1293 |
| R1, seed 102 | 0.0229 | 0.8925 | 0.9661 | 0.0036 | 0.9342 | 0.1015 |
| R0 | 0.0219 | 0.8964 | 0.9601 | 0.0046 | 0.9458 | 0.0992 |

## Correlation between models (primary)

Pearson correlation of the 2,000 bootstrap replicates, same chip draws for every model.

| Pair | AUROC_e | Unflagged error rate |
|---|---|---|
| R1 seed 101 × R0 | 0.805 | 0.863 |
| R1 seed 102 × R0 | 0.730 | 0.759 |
| Mean of the two R1 seeds × R0 | 0.792 | 0.884 |
| R1 seed 101 × R1 seed 102 | 0.894 | 0.717 |

## Correlation of per-chip AUROC_e (secondary)

76 chips with at least 50 errors and 50 correct pixels in every model.

| Pair | Pearson | Spearman |
|---|---|---|
| R1 seed 101 × R0 | 0.756 | 0.925 |
| R1 seed 102 × R0 | 0.739 | 0.929 |
| R1 seed 101 × R1 seed 102 | 0.951 | 0.959 |

## Spread between the two R1 seeds

|x₁ − x₂| / √2, one degree of freedom: AUROC_e 0.0041, unflagged error rate 0.0197, mIoU 0.0012.

## Checks

- Scored pixels per chip in the current label files equal those in the saved R0 labels, in all 89 chips.
- Pooled AUROC_e from the histograms equals the exact value to six decimals for all three models.
- The 32-bit mIoU equals the 16-bit `val/mIoU` logged at epoch 50, for both seeds.
- α = 0.10 is larger than the validation error rate of every model (precondition 1).
- R0 reproduces the week 1 validation values: AUROC_e 0.9601 and unflagged error rate 9.9%.

The full record, with per-chip values, is `resultados/semana4/piloto_sementes.json` on Google Drive.
