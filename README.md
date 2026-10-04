# Prithvi confidence under geographic shift

Code, configurations, data splits, pre-registration and analysis for a master's dissertation at the Centro de Informática, Universidade Federal de Pernambuco (CIn/UFPE).

**Status:** work in progress. Nothing in this repository is a result yet.

## Question

When a geospatial foundation model fine-tuned for flood segmentation is applied to a region it was not trained on, does its confidence still indicate where it is wrong? Does the answer change between a frozen encoder, LoRA and full fine-tuning?

Model under study: Prithvi-EO-2.0-300M-TL (IBM and NASA), fine-tuned on Sen1Floods11.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | TerraTorch configuration files, one per fine-tuning regime |
| `notebooks/` | Colab notebooks for training and inference |
| `analysis/` | Analysis scripts, frozen and hashed before use on target data |
| `splits/` | Lists of chips and events used for calibration and evaluation |
| `prereg/` | Pre-registration, with date and hash |
| `modelo.md` | Identification of the checkpoint used: source commit and file hashes |

## Data and model weights

Large files (checkpoints, imagery, saved probabilities) are not stored here. Checkpoints will be archived on Zenodo and the Hugging Face Hub when the work is published.

## Citation

Each release receives a DOI through Zenodo. Until the first release, see `CITATION.cff`.

## Author

Rosemeri Borges, master's student, CIn/UFPE. Advisor: Prof. Dr. Cleber Zanchettin.

## License

Code under the MIT License (see `LICENSE`).
