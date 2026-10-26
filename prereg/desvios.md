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
