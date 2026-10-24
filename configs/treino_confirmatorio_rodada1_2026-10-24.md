# Confirmatory training, round 1: R1 and R3, seeds 1 and 2 (2026-10-24)

Rule fixed before the runs: `prereg/preregistro.md` (commit `26cc8b7`, section 4). Notebook: `notebooks/semana5_treino_confirmatorio.ipynb` (commit `7bee93f`), Colab T4, environment of `configs/ambiente_regimes_2026-10-10.json`. Validation set only: 89 chips, 86 with scored pixels, 20,294,725 scored pixels. The test set, Bolivia and WorldFloods v2 were not read, and no target output was produced.

## Runs

All four runs finished on the first attempt. No run failed the rule (non-finite loss, or validation mIoU below 0.80 at epoch 50). Failures recorded: 0.

| Regime | Seed | Attempt | val/mIoU at epoch 50 (16-bit) | mIoU, 32-bit inference | Validation error rate | Minutes | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| R1 (frozen encoder, lr 5e-4) | 1 | 1 | 0.8917 | 0.8917 | 2.34% | 10.4 | 4.7 GB |
| R1 (frozen encoder, lr 5e-4) | 2 | 1 | 0.8898 | 0.8897 | 2.38% | 10.8 | 6.0 GB |
| R3 (full fine-tuning, lr 5e-5) | 1 | 1 | 0.8939 | 0.8939 | 2.23% | 12.6 | 10.8 GB |
| R3 (full fine-tuning, lr 5e-5) | 2 | 1 | 0.8956 | 0.8956 | 2.21% | 13.0 | 10.8 GB |

## Validation mIoU every 5 epochs

| Run | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 seed 1 | 0.814 | 0.833 | 0.863 | 0.861 | 0.878 | 0.888 | 0.885 | 0.890 | 0.890 | 0.892 |
| R1 seed 2 | 0.818 | 0.860 | 0.857 | 0.864 | 0.881 | 0.884 | 0.884 | 0.888 | 0.888 | 0.890 |
| R3 seed 1 | 0.795 | 0.865 | 0.873 | 0.868 | 0.893 | 0.886 | 0.887 | 0.896 | 0.893 | 0.894 |
| R3 seed 2 | 0.836 | 0.867 | 0.872 | 0.871 | 0.871 | 0.892 | 0.892 | 0.896 | 0.894 | 0.896 |

## Checks

- The 32-bit mIoU equals the 16-bit `val/mIoU` logged at epoch 50 to the fourth decimal in every run (one difference of 0.0001, R1 seed 2).
- α = 0.10 is larger than the validation error rate of every run (precondition 1).
- No seed of the pilot (0, 101, 102) was used.

## Notes

- Spread between the two seeds, |x₁ − x₂| / √2, one degree of freedom: val mIoU 0.0013 (R1) and 0.0012 (R3). The pilot gave 0.0012 for two R1 seeds.
- The grid (seed 0, one run per setting) gave 0.8941 for R1 and 0.8996 for R3. The round 1 scores are 0.002 to 0.006 lower. The grid picked the best of nine runs, so its scores are optimistic by construction. This was not tested.
- Peak GPU memory differs from the grid (R1 4.7 GB, R3 9.5 GB) in R1 seed 2 and in both R3 runs. The cause was not investigated. It does not affect the scores.

The full record of each run is `resultados/semana5/exec_*.json` on Google Drive. Margins and trained weights are `margens_*.npz` and `pesos_*.pt` in the same folder.
