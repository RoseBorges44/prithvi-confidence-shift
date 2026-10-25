# Confirmatory training, R1, R2 and R3, seeds 1 to 5 (2026-10-24)

Rule fixed before the runs: `prereg/preregistro.md` (commit `26cc8b7`, section 4). Notebook: `notebooks/semana5_treino_confirmatorio.ipynb`, Colab T4, environment of `configs/ambiente_regimes_2026-10-10.json`. Validation set only: 89 chips, 86 with scored pixels, 20,294,725 scored pixels. The test set, Bolivia and WorldFloods v2 were not read, and no target output was produced.

Notebook commits used: `7bee93f` (round 1, seeds 1 and 2), `d2da4d6` (round 2 until the interruption below: R1 seed 3, R3 seed 3, R1 seed 4), `c5dcec1` (R3 seed 4, R1 seed 5, R3 seed 5), `6632363` (R2, seeds 1 to 5). The four commits differ only in the run list, the introduction text, the RAM check between runs and, for R2, the LoRA configuration with its two checks (96 LoRA modules, 20.6 M trainable parameters). The training, inference and failure-rule code is the same.

## Runs

All fifteen runs finished and passed the rule (finite loss, validation mIoU at epoch 50 not below 0.80). Failures recorded: 0. Every run is attempt 1.

| Regime | Seed | val/mIoU at epoch 50 (16-bit) | mIoU, 32-bit inference | Validation error rate | Minutes | Peak GPU memory |
|---|---|---|---|---|---|---|
| R1 (frozen encoder, lr 5e-4) | 1 | 0.8917 | 0.8917 | 2.34% | 10.4 | 4.7 GB |
| R1 | 2 | 0.8898 | 0.8897 | 2.38% | 10.8 | 6.0 GB |
| R1 | 3 | 0.8954 | 0.8954 | 2.21% | 10.8 | 4.7 GB |
| R1 | 4 | 0.8915 | 0.8915 | 2.37% | 11.3 | 6.0 GB |
| R1 | 5 | 0.8894 | 0.8894 | 2.41% | 11.4 | 6.0 GB |
| R2 (LoRA r = 16, lr 5e-4, secondary) | 1 | 0.8957 | 0.8957 | 2.17% | 12.4 | 6.1 GB |
| R2 | 2 | 0.8971 | 0.8971 | 2.15% | 12.5 | 7.4 GB |
| R2 | 3 | 0.8999 | 0.8999 | 2.10% | 12.9 | 7.4 GB |
| R2 | 4 | 0.8909 | 0.8909 | 2.26% | 12.9 | 7.4 GB |
| R2 | 5 | 0.9016 | 0.9016 | 2.08% | 13.0 | 7.4 GB |
| R3 (full fine-tuning, lr 5e-5) | 1 | 0.8939 | 0.8939 | 2.23% | 12.6 | 10.8 GB |
| R3 | 2 | 0.8956 | 0.8956 | 2.21% | 13.0 | 10.8 GB |
| R3 | 3 | 0.8977 | 0.8977 | 2.19% | 13.4 | 10.8 GB |
| R3 | 4 | 0.8925 | 0.8925 | 2.28% | 13.4 | 9.5 GB |
| R3 | 5 | 0.8950 | 0.8950 | 2.23% | 12.8 | 9.5 GB |

| Regime | Mean val/mIoU | SD between seeds (n − 1) | Range | Mean validation error rate |
|---|---|---|---|---|
| R1 | 0.8916 | 0.0024 | 0.8894 to 0.8954 | 2.34% |
| R2 | 0.8970 | 0.0041 | 0.8909 to 0.9016 | 2.15% |
| R3 | 0.8949 | 0.0019 | 0.8925 to 0.8977 | 2.23% |

## Validation mIoU every 5 epochs

| Run | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 seed 1 | 0.814 | 0.833 | 0.863 | 0.861 | 0.878 | 0.888 | 0.885 | 0.890 | 0.890 | 0.892 |
| R1 seed 2 | 0.818 | 0.860 | 0.857 | 0.864 | 0.881 | 0.884 | 0.884 | 0.888 | 0.888 | 0.890 |
| R1 seed 3 | 0.836 | 0.857 | 0.869 | 0.869 | 0.875 | 0.889 | 0.892 | 0.890 | 0.893 | 0.895 |
| R1 seed 4 | 0.829 | 0.824 | 0.831 | 0.880 | 0.884 | 0.882 | 0.888 | 0.891 | 0.893 | 0.891 |
| R1 seed 5 | 0.846 | 0.838 | 0.849 | 0.874 | 0.869 | 0.879 | 0.881 | 0.890 | 0.893 | 0.889 |
| R2 seed 1 | 0.823 | 0.845 | 0.848 | 0.876 | 0.881 | 0.884 | 0.895 | 0.892 | 0.895 | 0.896 |
| R2 seed 2 | 0.849 | 0.849 | 0.871 | 0.891 | 0.867 | 0.893 | 0.891 | 0.894 | 0.896 | 0.897 |
| R2 seed 3 | 0.833 | 0.848 | 0.863 | 0.874 | 0.894 | 0.898 | 0.901 | 0.897 | 0.899 | 0.900 |
| R2 seed 4 | 0.851 | 0.822 | 0.861 | 0.887 | 0.871 | 0.895 | 0.893 | 0.891 | 0.894 | 0.891 |
| R2 seed 5 | 0.808 | 0.854 | 0.862 | 0.884 | 0.883 | 0.882 | 0.894 | 0.902 | 0.901 | 0.902 |
| R3 seed 1 | 0.795 | 0.865 | 0.873 | 0.868 | 0.893 | 0.886 | 0.887 | 0.896 | 0.893 | 0.894 |
| R3 seed 2 | 0.836 | 0.867 | 0.872 | 0.871 | 0.871 | 0.892 | 0.892 | 0.896 | 0.894 | 0.896 |
| R3 seed 3 | 0.847 | 0.860 | 0.861 | 0.889 | 0.876 | 0.897 | 0.895 | 0.896 | 0.896 | 0.898 |
| R3 seed 4 | 0.818 | 0.841 | 0.865 | 0.874 | 0.888 | 0.885 | 0.880 | 0.891 | 0.893 | 0.892 |
| R3 seed 5 | 0.842 | 0.868 | 0.876 | 0.883 | 0.882 | 0.891 | 0.894 | 0.892 | 0.895 | 0.895 |

## Checks

- The 32-bit mIoU equals the 16-bit `val/mIoU` logged at epoch 50 to the fourth decimal in every run (one difference of 0.0001, R1 seed 2). For R2 the five runs agree exactly.
- α = 0.10 is larger than the validation error rate of every run (precondition 1).
- No seed of the pilot (0, 101, 102) was used.

## Interruption

The first execution of R3 seed 4 was interrupted: the Colab session ran out of RAM after the training had finished and before any score was printed or saved, so no result of that execution was seen. No record was written. The run was executed again from the start with the same seed (attempt 1 in the record). This is a crash of the session, not a failure under the rule of section 4, and it is reported here. After it, the notebook was given a check that stops between runs when RAM use passes 60% (commit `c5dcec1`). RAM use rose by about 3 to 5 GB per run in the same session (20%, 58%, 77% before the next runs), so the session was restarted after the interruption and again when the check stopped the notebook before R3 seed 5. The five R2 runs were done in one session starting from 20%: RAM use was 42% to 44% before runs 2 to 5 and the check never stopped the notebook.

## Notes

- The grid (seed 0, one run per setting) gave 0.8941 for R1 and 0.8996 for R3. The grid picked the best of nine runs, so its scores are optimistic by construction; the five-seed means are 0.8916 (R1) and 0.8949 (R3). The grid gave 0.9000 for R2; its five-seed mean is 0.8970. This was not tested.
- Peak GPU memory differs between runs of the same regime (R1 4.7 to 6.0 GB, R2 6.1 to 7.4 GB, R3 9.5 to 10.8 GB), possibly because of what the session still held from earlier runs. The cause was not investigated. It does not affect the scores.
- On the validation set the means rank R2 (0.8970), R3 (0.8949) and R1 (0.8916), with differences of 0.002 to 0.005 and a spread between seeds of 0.002 to 0.004. This is the source validation only; it says nothing about the hypotheses.
- The R2 weights files hold the full state dict (base and LoRA tensors).

The full record of each run is `resultados/semana5/exec_*.json` on Google Drive. Margins and trained weights are `margens_*.npz` and `pesos_*.pt` in the same folder.
