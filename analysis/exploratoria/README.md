# analysis/exploratoria

Exploratory analyses of `prereg/plano_exploratorio.md`, written after the single confirmatory run. Nothing here is confirmatory, and nothing here changes the frozen scripts in `analysis/`: they are imported read-only, and their SHA-256 is checked against `analysis/README.md` before anything runs.

| File | What it does |
|---|---|
| `exploratorias.py` | E1 (per-event plot), E2 (unflagged errors by class), E4 (ensemble and seed variance per event; ensemble R3 minus ensemble R1 with a 90% interval) and E6 (per-unit Δ_reg against water fraction within events), on the primary target units. Before writing anything it checks that tau of every run, the single-model AUROC_e per event, the unflagged error rate and the pooled ensemble values equal those of the confirmatory run (tolerance 1e-9), and stops if they do not. Refuses an output folder inside the confirmatory one or one that is not empty. |
| `teste_exploratorias.py` | Runs the frozen analysis and then `exploratorias.py` on synthetic data, checks the internal consistency of the tables, and checks that a changed tau or an output folder inside the confirmatory one stops the script. |

E3 (distance to the label boundary) and E5 (released checkpoint through the week-6 inference pipeline) need the label rasters or a GPU and come in their own notebooks.

Run on Colab with `notebooks/semana8_exploratorias.ipynb`.
