# Deviations from the pre-registration

Changes and differences from `preregistro.md`, each with its date, what the text says, what happened and what follows (section 10 of the pre-registration). The text of `preregistro.md` is not edited.

## 2026-10-24: no OSF registration before the confirmatory training

**What the text says.** The status line of `preregistro.md` says the document is registered with a dated commit of this repository and on OSF before any confirmatory training run. Section 12 says the commit of the file and its SHA-256 are recorded on OSF.

**What happened.** Only the dated commit `26cc8b7` (2026-10-20) was made. The confirmatory training runs (R1, R2 and R3, seeds 1 to 5, `configs/treino_confirmatorio_2026-10-24.md`) were run on 2026-10-24 without an OSF record.

**Effect.** The public repository with the dated commit `26cc8b7` is the only time stamp of the pre-registration. The training runs read only the Sen1Floods11 training and validation splits. No output on the target (Bolivia, WorldFloods v2, Rio Grande do Sul) or on the source test split exists.

**Follow-up.** Any other registration or archive made later is added here with its date and what it records.

## 2026-10-25: distance used for the boundary exclusion

**What the text says.** Section 9 lists, as a sensitivity analysis, the exclusion of water and land pixels within 2 pixels of a water–land boundary of the label, and says that pixels labeled −1 do not form boundaries. It does not say which distance is used.

**What happened.** The definition was fixed in the inference notebook before any target inference (commit `665182a`) and documented in `analysis/README.md`: a water or land pixel is flagged when a pixel of the other class lies within the 5 × 5 window around it (Chebyshev distance 2 or less). Chip edges are not boundaries. For T2 the flag is computed on the whole scene label and then cut into windows.

**Effect.** Only the boundary-exclusion sensitivity analysis (`sem_fronteira`) uses the flag. No primary result and no other sensitivity analysis depends on it.

## 2026-10-25: test-time augmentation not run

**What the text says.** Section 9 lists, among the secondary descriptive analyses, test-time augmentation with the 8 dihedral transforms, against single-model MSP.

**What happened.** In the inference notebook (`notebooks/semana6_inferencia_alvos.ipynb`) this is an optional cell (`FAZER_TTA`), which takes about 8 times the plain inference time. It was not attempted. The single run of the frozen script therefore used `--sem-tta` (`"sem_tta": true` in `resultados.json`), and no TTA result exists.

**Effect.** No statement can be made about test-time augmentation against single-model MSP. The TTA variant is only reported, so no primary outcome, no outcome of the Holm family and no sensitivity analysis depends on it.

**Follow-up.** Whether TTA is run is still open. If it is, it is run after the regime key was opened (next entry) and is labeled so. The frozen script already reads the `margem_tta` field, so no new analysis choice is needed. The tables of the first run remain the record, and the primary tables of a second run must be identical to them.

## 2026-10-25: single run of the frozen script and opening of the regime key

This is not a deviation. It is recorded here because the pre-registration has no OSF time stamp (first entry).

**What happened.** The target inference of the 15 confirmatory runs and the single run of the frozen analysis script (commit `f9de2f5`, SHA-256 in `analysis/README.md`, run with `notebooks/semana7_analise_final.ipynb`) were done on 2026-10-25, Brasília time. The script read the regime key only after processing all runs, and the key was opened at that point: R1 = `8tjg`, R2 = `4m9f`, R3 = `6wke`. The outputs (`resultados.json`, `analise.log`, `rodada.json` and the tables) are stored with the margins, outside this repository.

**Effect.** Any analysis done after this point is done with the regime key known.

## 2026-11-05: test-time augmentation run after the opening of the regime key

This follows the entry of 2026-10-25 on test-time augmentation. It is written before any TTA margin exists.

**Decision.** The test-time augmentation of section 9 is run. It is labeled "secondary pre-registered analysis, run after the opening of the regime key".

**How.**

- The definition is the one already written in `notebooks/semana6_inferencia_alvos.ipynb`: the 8 dihedral transforms (rotations by 0, 90, 180 and 270 degrees, without and with a horizontal flip), and the margin is the logit of the mean water probability over the 8 (mean in float64, stored as float32). Model assembly, preprocessing, 32-bit inference and the batch of 8 are those of week 6.
- `margem_tta` is computed for the source test, Bolivia and the T2 windows, the sets where the frozen script reads it. It is not computed for the validation set, because τ comes from the plain margins.
- The optional TTA cell of the week-6 notebook writes into the week-6 margin files. It is not used, so that the files read by the first run stay as they are. `notebooks/semana10_tta_margens.ipynb` writes `margem_tta` to a new folder (`resultados/semana10/tta_margens`). In the same pass it computes the untransformed margin and compares it with the stored week-6 margin of each run and set; a maximum difference above 1e-3 stops the run before anything is written.
- The frozen script (`analysis/analise_congelada.py` and its modules, SHA-256 in `analysis/README.md`, unchanged) is run once more, without `--sem-tta`, on files that join the stored week-6 `margem` with the new `margem_tta`, built on local disk after the week-6 files are checked against their manifest. The output goes to a new folder.

**Check made before the run.** On synthetic margins (`analysis/gerador_sintetico.py` with `tta=True`, B = 300), the script was run with and without `--sem-tta`. The seven tables were byte-identical, and `resultados.json` differed only in `meta.sem_tta` and `variantes.<regime>.tta`. The bootstrap draws are made before the runs are read, so the TTA margins do not change them.

**What must hold.** The seven tables of the second run must be byte-identical to those of the first run, and every value of `resultados.json` other than `meta.sem_tta` and `variantes.<regime>.tta` must be equal. If so, the TTA values are reported as a secondary pre-registered analysis run after the opening of the regime key. If not, the difference is recorded here and the first run remains the record.

**Follow-up, 2026-11-07.** The TTA margins were computed (`analysis/tta/margens/`). The untransformed margin matched the stored week-6 margin within 2.3e-5 (limit 1e-3), and 3 pixels changed sign out of 137,160,388 per run, one in each of three runs. The analysis uses the stored week-6 margin. Colab's numpy changed from 2.1.3, the version recorded by the first run, to 2.4.6. The second run of the frozen script therefore uses numpy 2.1.3 and scipy 1.16.3, the versions of the first run, installed apart from the notebook's own packages, so that a library change cannot alter the tables. Colab's Python is now 3.13.18 (first run 3.13.16) and cannot be pinned there; `meta.versoes.python` is the only other value of `resultados.json` allowed to differ.
