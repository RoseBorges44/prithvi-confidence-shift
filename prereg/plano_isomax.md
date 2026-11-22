# Plan of the exploratory IsoMax+ arm (E10)

**Status.** Written on 2026-11-22, after the single run of the frozen analysis script, after the opening of the regime key and after the exploratory analyses E1 to E9 (`plano_exploratorio.md`, `plano_exploratorio_2.md`). It is committed before any code for this arm is written and before any model is trained with it. This is the IsoMax arm listed in section 9 of `preregistro.md` ("an IsoMax-loss arm in R1 with 3 seeds, run only after the confirmatory analysis"). Nothing here is confirmatory: no category of section 1 of `preregistro.md` is applied, no p-value is used to claim an effect, and the results go to a section of the dissertation labeled exploratory.

**Rules.** The rules of `plano_exploratorio.md` apply unchanged:

- Every analysis in this plan is reported, whatever it shows.
- The frozen scripts (`analysis/analise_congelada.py`, `nucleo.py`, `reamostragem.py`, `categorias.py`) are not changed. The new code imports them read-only, checks their SHA-256 against `analysis/README.md` at the start, and lives in `analysis/exploratoria/`.
- The tables of the confirmatory run and of E1 to E9 are not replaced. New outputs go to a separate folder (`resultados/semana12/`).
- MSP, τ, error, flagged pixel (empty set) and AUROC_e are computed as in `nucleo.py`. Arm values are the mean over the 3 seeds.
- Any change to this plan after the first computation is added at the end, with its date and reason.

## Choices and why they were made

The pre-registration fixes the arm (R1, IsoMax loss in place of cross-entropy, same recipe, 3 seeds, after the confirmatory analysis) but not the variant of the loss, its scales or the score. These were fixed as follows, before any result.

1. **Variant: IsoMax+ (enhanced IsoMax).** Macêdo and Ludermir (arXiv 2105.14399, v10, 2022) present IsoMax+ as the improved version of the IsoMax loss of Macêdo et al. (IEEE TNNLS, 2022): the features and the class prototypes are normalized ("isometrization"), and a learnable distance scale is added. It is the version the authors recommend and the one used in the most recent work of the same group (Macêdo, arXiv 2208.03566, 2022). Like IsoMax, it is presented as a drop-in replacement of the output layer, the SoftMax and the cross-entropy, with no change in architecture or training hyperparameters. That property is what the pre-registration asks for ("same recipe"): the learning rate, the number of epochs, the optimizer and the data stay those of R1, and no new grid is run.
2. **Scales: those of the method, not tuned.** The entropic scale is fixed at 10, as in the authors' code, and is applied only in the loss during training. The distance scale is a learnable parameter, initialized at 1, as in the authors' code. Tuning either scale on the data of this work would be a choice made with the results in view, and is not done.
3. **Two scores with different roles.** The primary score is the MSP of the IsoMax+ model, computed exactly as for the other regimes, so that τ, the unflagged error rate and AUROC_e are comparable with R1 and any difference comes from training. In the binary case, the entropic score is a monotone function of the MSP, orders pixels in the same way and adds nothing. The secondary score is the minimum distance score proposed by the authors for IsoMax+, which does not order pixels as the MSP does.
4. **Comparison within R1, same seeds.** The arm is compared with the confirmatory R1 runs of seeds 1, 2 and 3, which share the frozen encoder, the decoder, the recipe and the seeds. The difference is the loss.

## Model and training

- **Model.** The R1 model of `notebooks/semana5_treino_confirmatorio.ipynb`: pinned backbone `Prithvi_EO_V2_300M_TL.pt` (SHA-256 in `configs/ambiente_regimes_2026-10-10.json`), frozen; UperNet decoder with 256 channels; segmentation head with dropout 0.1. The only change is the last layer of the head: the 1 × 1 convolution to 2 classes is replaced by an IsoMax+ layer applied per pixel. For each pixel, with feature vector f (the input of the replaced convolution) and prototypes p_land and p_water, the logit of class j is −|d_s| · ‖f/‖f‖ − p_j/‖p_j‖‖ (Euclidean distance). Prototypes are initialized from a normal distribution with mean 0 and standard deviation 1, and d_s at 1, as in the authors' code.
- **Loss.** TerraTorch cross-entropy with `ignore_index=-1`, applied to the logits multiplied by the entropic scale 10 during training. In evaluation mode the layer returns the logits without the entropic scale. Validation loss during training is therefore computed without the scale; validation mIoU is not affected, because the scale does not change the argmax.
- **Recipe.** As R1: learning rate 5e-4, AdamW with weight decay 0.05 (applied to all trainable parameters, prototypes and d_s included), cosine schedule over 50 epochs, the batch size, bands, augmentation and data of the confirmatory training, `seed_everything(seed, workers=True)`.
- **Seeds.** 1, 2 and 3.
- **Failure rule.** As in section 4 of `preregistro.md`: a run fails if the loss stops being finite or if validation mIoU at epoch 50 is below 0.80. A failed run is repeated once with the same seed; if it fails again, seed 4 replaces it. The decision is made before any inference on the source test or the target.

## Inference

- With the reading, preprocessing and 32-bit inference code of `notebooks/semana6_inferencia_alvos.ipynb`, on the validation set, the source test, Bolivia and the T2 windows, with the labels checked against the week-6 label files.
- Margin = logit of water minus logit of land, both without the entropic scale. Minimum distance score = the largest of the two logits (minus the scaled minimum distance). Both are stored per scored pixel, in files with the same layout as the week-6 margin files.
- Code check before the arm: the same notebook runs the confirmatory R1 seed 1 with its stored weights, and its Bolivia margins must match the week-6 margins within 1e-3. If they do not, nothing is written for the arm.
- The weights of the 3 runs are stored, with SHA-256 in a manifest.

## Measures

On the validation set, the source test, the pooled target (primary units, as in the confirmatory run) and each target event, per run and as the mean over the 3 seeds:

- τ on validation (as in `nucleo.py`), coverage, empty-set rate, error rate, mIoU, unflagged error rate and AUROC_e of the MSP;
- AUROC_e of the minimum distance score;
- Δ_desl of the arm (AUROC_e of the MSP, target minus source test).

Comparisons, each with a 90% interval from a bootstrap of units within events (B = 10,000, `numpy.random.default_rng(5)`, seeds not resampled, the same draws for both arms so the comparison is paired):

- IsoMax+ minus R1 (seeds 1 to 3) in AUROC_e of the MSP on the target and on the source test;
- IsoMax+ minus R1 in the unflagged error rate on the target, in percentage points;
- within the arm, AUROC_e of the minimum distance score minus AUROC_e of the MSP, on the target and on the source test.

The margin δ = 0.02 of the pre-registration is shown as a reference only. With 3 seeds, the intervals cover the variation of the units and not the choice of seeds.

## Cost

About 35 minutes of T4 GPU for training (about 11 minutes per run in R1), plus inference on four sets for 4 models (the 3 runs and the code check), and a CPU analysis.

## Not in this plan

- IsoMax+ in R2 or R3, more seeds, a learning-rate grid or any tuning of the scales.
- The original IsoMax, without normalization.
- Any new event, sensor or model.

## Changes to this plan

None yet.
