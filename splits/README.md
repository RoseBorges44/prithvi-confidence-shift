# splits

Lists of chips and events: the official Sen1Floods11 splits, the Bolivia hold-out, and the second target set. Each list is fixed, with date and hash, before any inference on it.

| File | Chips | Role | Source | SHA-256 | Fixed on |
|---|---|---|---|---|---|
| `sen1floods11_val.txt` | 89 | Calibration (q-hat and tau) and learning-rate choice | Official `flood_valid_data.csv`, hand-labeled split, `gs://sen1floods11/v1.1/splits/flood_handlabeled/` | `14a372859fb18b1f7f9791810b7a233c2cd53aaab84f74721f9ac3476ba50504` | 2026-10-04 |
| `sen1floods11_test.txt` | 90 | Source evaluation | Official `flood_test_data.csv`, same location | `cfcf04cc36880d0b4156681c9c2efaeb26d62e7ecb6987002e437436cdf98627` | 2026-10-04 |
| `t2_eventos.csv` | 18 maps; core 6 maps in 4 events | Second shifted target (T2), WorldFloods v2 test split. Primary analysis: rows with `core` true. Secondary, descriptive: `full_set` true (12 maps, 8 events) | Criteria in `prereg/t2_criterios.md` (commit `703c7a3`), applied by `notebooks/semana2_t2_worldfloods.ipynb`, part B | `d0cd80324207e78fbc8fe02b9626610e914cc3b470bb8168d5be790235c5b892` | 2026-10-10 |
| `t2_janelas.csv` | 819 windows of the 12 full-set maps; core: 242 in the primary set, 258 in the sensitivity set | 512 × 512 windows of T2, one row per window with the inclusion result | Rule in `prereg/t2_janelas.md` (commit `7ab6806`), applied by `notebooks/semana2_t2_janelas.ipynb` | `3c21970835f2d43cd436be9616c451ebbff5fcea138b7401d3867f0901643b2c` | 2026-10-10 |

Each line is the label file name of one chip. The lists were matched against the official CSV files by the chip key (country and number).

### T2 files

- `t2_eventos.csv`: one row per map of the WorldFloods v2 test split, with the inputs of each criterion and the result. The `country_ems` column is empty because the `country` field of the dataset metadata files is empty.
- `t2_eventos_info.json`: dataset revision, criteria, list of training countries, Natural Earth file and hash, core and full sets, and library versions.
- `t2_arquivos_sha256.csv`: SHA-256 of the 92 downloaded files, each checked against the hash published by Hugging Face for revision `1f3faa2989e69930ac31d5c0a79fd224461123f8`.

Core set: EMSR333 (Italy: Rattaloro, Portopalo), EMSR342 (Australia: North Normanton), EMSR347 (Malawi: Mwanza, Zomba on 2019-03-10) and EMSR438 (Tanzania: Ukerewe, AOI06). EMSR466 left by C1, with 62.8% of its labeled pixels in Nigeria. EMSR286 (Colombia, both maps) and Zomba on 2019-03-12 left by C3.

### T2 windows

- `t2_janelas.csv`: one row per window, with offsets, fraction of pixels without input data, scored, water and land pixels, and the inclusion result for the primary set (at least 10% scored) and the sensitivity set (at least 1 pixel). Every labeled cloud-free pixel of the 12 maps is scored in exactly one window.
- `t2_janelas_info.json`: the rule, the hash of the T2 list, band indices, label values, counts and library versions.
- The input rule (at most 1% of a window without input data) removed 272 windows. The blank input lies inside the scenes, in irregular patches, and not in a frame along the scene edges: no blank border was found in any core map, and the two EMSR333 scenes have 5.2% and 4.0% of their area without input data. In EMSR333 only the two central windows remain. No labeled pixel lacks input data.
