# analysis/exploratoria

Exploratory analyses of `prereg/plano_exploratorio.md`, written after the single confirmatory run. Nothing here is confirmatory, and nothing here changes the frozen scripts in `analysis/`: they are imported read-only, and their SHA-256 is checked against `analysis/README.md` before anything runs.

| File | What it does |
|---|---|
| `exploratorias.py` | E1 (per-event plot), E2 (unflagged errors by class), E4 (ensemble and seed variance per event; ensemble R3 minus ensemble R1 with a 90% interval) and E6 (per-unit Δ_reg against water fraction within events), on the primary target units. Before writing anything it checks that tau of every run, the single-model AUROC_e per event, the unflagged error rate and the pooled ensemble values equal those of the confirmatory run (tolerance 1e-9), and stops if they do not. Refuses an output folder inside the confirmatory one or one that is not empty. |
| `teste_exploratorias.py` | Runs the frozen analysis and then `exploratorias.py` on synthetic data, checks the internal consistency of the tables, and checks that a changed tau or an output folder inside the confirmatory one stops the script. |
| `e3_borda.py` | E3 in two steps. `distancias`: re-reads the original labels with the code of the inference notebook, computes for every scored pixel the Chebyshev distance to the nearest pixel of the other class (per chip for Sen1Floods11, on the whole scene for T2), and stops unless units, sizes and labels equal `labels_<set>.npz` and the pixels with distance 1 or 2 are exactly the `borda` pixels. `analise`: error rate, empty-set rate, unflagged error rate and AUROC_e per distance band, per regime and set; per unit, the Spearman correlation between empty-set rate and water fraction, the partial correlation given the boundary share, and the correlation over the pixels farther than 10 from the boundary. Checks tau and the RQ3 Spearman values against the confirmatory run. |
| `teste_e3.py` | Step 1 on small GeoTIFFs written by the test (including a chip without water and a T2 scene cut into four windows) and a changed boundary flag that must stop it; step 2 on the synthetic margins after the frozen analysis. |

E5 (released checkpoint through the week-6 inference pipeline) needs a GPU and comes in its own notebook.

Run on Colab with `notebooks/semana8_exploratorias.ipynb` (E1, E2, E4, E6) and `notebooks/semana8_e3_borda.ipynb` (E3).
