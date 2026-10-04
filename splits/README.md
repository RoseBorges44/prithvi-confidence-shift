# splits

Lists of chips and events: the official Sen1Floods11 splits, the Bolivia hold-out, and the second target set. Each list is fixed, with date and hash, before any inference on it.

| File | Chips | Role | Source | SHA-256 | Fixed on |
|---|---|---|---|---|---|
| `sen1floods11_val.txt` | 89 | Calibration (q-hat and tau) and learning-rate choice | Official `flood_valid_data.csv`, hand-labeled split, `gs://sen1floods11/v1.1/splits/flood_handlabeled/` | `14a372859fb18b1f7f9791810b7a233c2cd53aaab84f74721f9ac3476ba50504` | 2026-10-04 |
| `sen1floods11_test.txt` | 90 | Source evaluation | Official `flood_test_data.csv`, same location | `cfcf04cc36880d0b4156681c9c2efaeb26d62e7ecb6987002e437436cdf98627` | 2026-10-04 |

Each line is the label file name of one chip. The lists were matched against the official CSV files by the chip key (country and number).
