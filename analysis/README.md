# analysis

Analysis scripts. Each script is frozen, and its SHA-256 recorded here, before it is run on target data.

| Script | SHA-256 | Frozen on |
|---|---|---|
| `analise_congelada.py` | `ef0a1736d8c195eaf0e9cf21ee3e63694be0bf9ed7edcaca6545e0bb544993fb` | 2026-10-25 |
| `nucleo.py` | `19d0db55d4b8c1cc290eed2e3cd696b9aab9139b82b654d7521ee338ffce3a6b` | 2026-10-25 |
| `reamostragem.py` | `58ab295f001cf9f8d20b19817d54ccea5c20e00a4a72be1a081eccc1040973ed` | 2026-10-25 |
| `categorias.py` | `fb6b362125514d826be170be23b52b4fa78e4d74943df4d6e6b341ffacb7df3f` | 2026-10-25 |

## Files

| File | Role |
|---|---|
| `analise_congelada.py` | The analysis of `prereg/preregistro.md`: reads the margins, computes the metrics, the bootstrap, the categories, S1 to S3 with Holm correction, the preconditions and the sensitivity analyses. |
| `nucleo.py` | Metrics: AUROC_e, AUGRC, E-AURC, conformal threshold, temperature, calibration, counts per unit. |
| `reamostragem.py` | Draws, histogram metrics of the replicates, intervals, p-values, Holm. |
| `categorias.py` | Decision categories of H1 against H2 and of H3. |
| `r0_referencia.json` | R0 values from week 1, used by precondition 2 and by the R0 test. |
| `gerador_sintetico.py`, `testes_sinteticos.py` | Synthetic data with known effects and the tests of section 10. |
| `teste_r0.py` | Reproduction of the R0 values from the week 1 histograms. |

## Input

Directory given to `--margens`, written by the inference notebook.

- `labels_<set>.npz` with `chaves` (unit names), `tamanhos` (scored pixels per unit), `rotulo` (uint8, 0 land, 1 water) and, for `teste`, `bolivia` and `t2`, `borda` (bool, True for a pixel within 2 pixels of a water-land boundary of the label; pixels labeled -1 do not form boundaries).
- `run_<code>_s<seed>_<set>.npz` with `chaves`, `tamanhos`, `margem` (float32, z_water - z_land from 32-bit inference) and, optionally, `margem_tta` (float32, logit of the mean probability over the 8 dihedral transforms).
- Sets: `val`, `teste` (source test), `bolivia`, `t2` (every window with at least one scored pixel of the core, full-set and cloud 0.50 maps).
- Windows of the cloud 0.50 maps are read from `splits/t2_janelas_nuvem050.csv` (same columns as `splits/t2_janelas.csv`, no window in both files). Without that file the analysis stops, unless `--sem-nuvem050` is given.
- Pixels of a unit are the scored pixels in row-major order. The same margins hold for all sets in the same file order as the labels file.
- `<code>` is the random regime code (letters and digits). Each code has seeds 1 to 5 (a replaced seed 6 takes the place of the failed one) and all four sets.
- `chave_codigos.json`, a JSON object code to `"R1"`, `"R2"` or `"R3"`, is read after all codes have been processed.

## Run

```
python analysis/analise_congelada.py --repo . --margens DIR --chave DIR/chave_codigos.json --saida OUT
```

It needs about 8 GB of memory at the real scale (a machine with 12 GB is recommended) and about 4 hours. `--B` sets the replicates (registered: 10000). `--sem-nuvem050` skips the cloud 0.50 analysis and is recorded in the output; use it only if the windows of those maps were not cut.

The output has `resultados.json` and the tables in `tabelas/`: `principal.csv` (H1 to H3, S1 to S3), `sensibilidades.csv`, `eventos.csv`, `regimes.csv`, `precondicoes.csv`, `execucoes.csv`, `rq3_por_unidade.csv`.

## Checks on the input

The analysis stops, with a message, when: a file name starting with `run_` or `labels_` is not recognized; a code has fewer or more than 5 seeds, a pilot seed (0, 101, 102) or a missing set; two runs have identical margins in a set; `rotulo` is not integer or boolean, `borda` is not boolean or flags no pixel; labels stored in a run file differ from the labels file; margins are not float32, not finite or constant; the validation error rate of a run is 0.5 or more (margins look inverted); the source test does not have 89 units with scored pixels in 10 events, or the validation set does not have 86; the key file is not a one-to-one map of the codes to R1, R2, R3; the histogram check cannot be met after 3 refinements.

## Definitions fixed by the script where the registration is silent

- Signed ECE: sum over 15 equal-mass bins of the pixel share times (mean confidence minus accuracy); positive means overconfident. Bins are made on the rank of |m| with a stable sort.
- Brier: mean of (p_water - y)^2.
- Percentile intervals use the linear rule of `numpy.quantile`. The MDE is (1.645 + 0.842) times the standard deviation of the replicates.
- Conformal threshold: tau = 1 - q, with q the empirical quantile of the validation score s = 1 - (probability of the true class) at level ceil((n + 1)(1 - alpha))/n, alpha = 1/10 computed as a fraction (`numpy.quantile`, method `higher`). A pixel is flagged when MSP >= tau.
- Seed draws and unit draws come from `numpy.random.default_rng(0)`: seed draws of R1, then R3, then the strata of the source test, then the strata of the target, each in sorted order, then the seed draw of R2. R1 and R3 therefore do not depend on R2 or on the order of the codes. The seeds of a regime are drawn independently of the other regimes; the same seed draw serves the source test and the target.
- Equivalence in S2 (90% interval inside +-5 pp) is not adjusted for multiplicity.
- Two-stage bootstrap: events are drawn with replacement, then the units of each drawn event with replacement; applied to the target events and to the source test events.
- Bins that are empty in every unit are dropped from the stored histograms; this does not change any AUROC_e, AUGRC or E-AURC.
- The histogram check of section 6 covers the pooled sets (source test; Bolivia and T2 together) of the 15 runs, with the threshold 1e-4. The histograms of events, leave-one-event-out, boundary subsets, the ensemble and the augmentation variants report their own difference and do not trigger refinements. If the check fails, vmax is raised to the largest |m| rounded up to a multiple of 5; otherwise the bin width is halved; at most 3 refinements.
- Leave-one-chip-out values and per-unit AUROC_e come from the histograms of each unit.
- Ambiguities of the registration resolved by the script: an event with an undefined AUROC_e in any run counts in neither sign count of H3; Holm stops on an undefined p-value; categories with an undefined point or limit are reported as Undefined.

## Tests

```
python analysis/testes_sinteticos.py
python analysis/teste_r0.py . fonte_registros.pkl bolivia_registros.pkl
```
