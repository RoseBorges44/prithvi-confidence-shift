# Pilot: learning-rate choice on the validation set

Fixed on 2026-10-11, before any run of the grid. Follows section 14.5 of the project plan.

## What runs

- Regimes R1 (frozen encoder), R2 (LoRA r = 16, alpha = 16) and R3 (full fine-tuning), built as in `notebooks/semana3_regimes.ipynb`, from the pinned backbone (Hugging Face revision `63adbd39c271da4c42f447e69b1a7c91a338cdc9`, SHA-256 `3629cedf…6ddb`).
- Recipe of the public checkpoint (`config.yaml`, commit `918b9f1`): 6 bands, `constant_scale` 1e-4, random crop of 224 pixels with horizontal and vertical flips, batch 16, cross-entropy with `ignore_index` -1, AdamW with weight decay 0.05, cosine schedule with `T_max` 50, UperNet with 256 channels and `head_dropout` 0.1, 16-bit mixed precision.
- One deliberate change: 50 fixed epochs, no early stopping. The checkpoint at epoch 50 is the one evaluated.
- Training chips: the 252 chips of `splits/sen1floods11_train.txt`. Validation: the 89 chips of `splits/sen1floods11_val.txt`, at 512 × 512 pixels, validated once, at the end of epoch 50.
- Seed 0 in every run (`seed_everything(0, workers=True)`). GPU kernels are not bit-for-bit deterministic, so a repeated run can differ slightly.
- Same environment as `configs/ambiente_regimes_2026-10-10.json`.

## Grid

| Regime | Learning rates |
|---|---|
| R1 | 1e-4, 5e-4, 1e-3 |
| R2 | 1e-4, 5e-4, 1e-3 |
| R3 | 2e-5, 5e-5, 1e-4 |

Nine runs in total.

## Choice rule

1. The score of a run is `val/mIoU` after epoch 50: the mean intersection over union of the two classes on the 89 validation chips, with `ignore_index` -1 pixels left out.
2. Per regime, the learning rate with the highest score is chosen.
3. Tie: among the learning rates whose score is less than 0.001 below the highest score, the smallest learning rate is chosen.
4. A run whose loss becomes non-finite is excluded from the choice and reported.
5. If the chosen value is at an end of the grid, it is kept. The grid is not extended after seeing the results, and this is stated as a limitation.

## What is not used

No confidence measure (maximum softmax probability, AUROC for error detection, calibration error, conformal threshold) enters the choice. The test set, Bolivia and WorldFloods v2 are not read in the pilot. Only the validation set is.

## Records

Each run writes its learning rate, regime, seed, `val/mIoU`, the other validation metrics logged by TerraTorch, the training time and the peak GPU memory to `resultados/semana3/grade_lr.json` on Google Drive. All nine runs are reported, not only the chosen ones.
