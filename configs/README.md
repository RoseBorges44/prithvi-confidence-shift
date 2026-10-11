# configs

TerraTorch configuration files: one per fine-tuning regime (frozen encoder, LoRA, full fine-tuning), plus the configuration of the released checkpoint. All regimes use the same TerraTorch version, recorded in `../modelo.md`.

## Environment check

`ambiente_regimes_2026-10-10.json` is the output of `notebooks/semana3_regimes.ipynb` on Google Colab (Tesla T4): library versions, the pinned pretrained backbone, and the trainable parameter counts of the three regimes built with the recipe of the released checkpoint. R1 has 15.08 M trainable parameters, R2 (LoRA) 20.59 M and R3 (full fine-tuning) 318.97 M. In R1 and R3, 295 encoder tensors equal the pinned backbone file.

Colab ships `torchao` 0.10.0, which `peft` 0.21.2 rejects when it builds LoRA modules. The notebook uninstalls it. After the install cell the session has to be restarted, because the install replaces the NumPy that is already loaded.
