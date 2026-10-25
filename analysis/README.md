# analysis

Analysis scripts. Each script is frozen, and its SHA-256 recorded here, before it is run on target data.

| Script | SHA-256 | Frozen on |
|---|---|---|

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
- Pixels of a unit are the scored pixels in row-major order. The same margins hold for all sets in the same file order as the labels file.
- `<code>` is the random regime code (letters and digits). Each code has seeds 1 to 5 (a replaced seed 6 takes the place of the failed one) and all four sets.
- `chave_codigos.json`, a JSON object code to `"R1"`, `"R2"` or `"R3"`, is read after all codes have been processed.

## Run

```
python analysis/analise_congelada.py --repo . --margens DIR --chave DIR/chave_codigos.json --saida OUT
```

It needs about 10 GB of memory with all windows. `--B` sets the replicates (registered: 10000). `--sem-nuvem050` skips the cloud 0.50 analysis and is recorded in the output; use it only if the windows of those maps were not cut.

The output has `resultados.json` and the tables in `tabelas/`: `principal.csv` (H1 to H3, S1 to S3), `sensibilidades.csv`, `eventos.csv`, `regimes.csv`, `precondicoes.csv`, `execucoes.csv`, `rq3_por_unidade.csv`.

## Definitions fixed by the script where the registration is silent

- Signed ECE: sum over 15 equal-mass bins of the pixel share times (mean confidence minus accuracy); positive means overconfident. Bins are made on the rank of |m| with a stable sort.
- Brier: mean of (p_water - y)^2.
- Percentile intervals use the linear rule of `numpy.quantile`. The MDE is (1.645 + 0.842) times the standard deviation of the replicates.
- Seed draws and unit draws come from `numpy.random.default_rng(0)`, seeds first (codes in sorted order), then the strata of the source test, then the strata of the target, each in sorted order.
- Two-stage bootstrap: events are drawn with replacement, then the units of each drawn event with replacement; applied to the target events and to the source test events.
- Bins that are empty in every unit are dropped from the stored histograms; this does not change any AUROC_e, AUGRC or E-AURC.
- The histogram check of section 6 covers the 15 runs. The ensemble and the augmentation variants report their own difference.

## Tests

```
python analysis/testes_sinteticos.py
python analysis/teste_r0.py . fonte_registros.pkl bolivia_registros.pkl
```
