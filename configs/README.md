# configs

TerraTorch configuration files: one per fine-tuning regime (frozen encoder, LoRA, full fine-tuning), plus the configuration of the released checkpoint. All regimes use the same TerraTorch version, recorded in `../modelo.md`.

## Environment check

`ambiente_regimes_2026-10-10.json` is the output of `notebooks/semana3_regimes.ipynb` on Google Colab (Tesla T4): library versions, the pinned pretrained backbone, and the trainable parameter counts of the three regimes built with the recipe of the released checkpoint. R1 has 15.08 M trainable parameters, R2 (LoRA) 20.59 M and R3 (full fine-tuning) 318.97 M. In R1 and R3, 295 encoder tensors equal the pinned backbone file.

Colab ships `torchao` 0.10.0, which `peft` 0.21.2 rejects when it builds LoRA modules. The notebook uninstalls it. After the install cell the session has to be restarted, because the install replaces the NumPy that is already loaded.

## Timing of one run per regime (2026-10-11)

Three epochs of each regime on a Colab T4, batch 16, same recipe as the pilot, validation after every epoch. Epoch 1 includes warm-up and is left out of the estimate. The data loader alone (one pass over the 252 training chips, 2 worker processes) takes 12 to 13 seconds, which is most of an epoch for all three regimes.

| Regime | Training, s per epoch (epochs 1, 2, 3) | Validation, s per epoch | Peak GPU memory | 50 epochs, validation at the end | 50 epochs, validation every 2 |
|---|---|---|---|---|---|
| R1 | 17.0, 13.2, 13.7 | 10.5, 7.2, 7.0 | 4.7 GB | 11.3 min | 14.2 min |
| R2 | 14.5, 14.7, 15.1 | 8.8, 9.9, 9.2 | 6.1 GB | 12.6 min | 16.4 min |
| R3 | 15.0, 16.2, 16.0 | 8.2, 9.1, 8.5 | 9.5 GB | 13.6 min | 17.1 min |

Validation mIoU after three epochs: R1 0.808, R2 0.815, R3 0.819 (lr values chosen only for timing). A first measurement of R1 on the same GPU gave 12.8 minutes, so the estimates carry an uncertainty of about a minute.
