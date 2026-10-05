# notebooks

Colab notebooks for training and inference. Google Drive paths are set in the first cell of each notebook.

| Notebook | What it does | Needs a GPU |
|---|---|---|
| `semana1_higiene.ipynb` | Checks the model hashes against the pinned Hugging Face commit, recomputes the source reference with the conformal threshold calibrated on the validation split and evaluated on the test split, checks 16 vs 32 bit precision, and compares Bolivia against the source test split with chip-level bootstrap confidence intervals for each difference. Reads saved scores only; it does not run the model. | No |
