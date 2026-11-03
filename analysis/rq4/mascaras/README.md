# Reference masks of RQ4

Written by `notebooks/semana9_rq4_mascaras.ipynb` (first run, 2026-11-03, folder `rq4_mascaras`, repository at commit `31f21eb`) with `analysis/rq4/mascaras.py`, before any model output on Rio Grande do Sul. The rules are those of `prereg/plano_rq4.md`, sections 1 to 3, and of its dated changes, in particular the entry of 2026-11-03 on the masks. The rasters themselves are not in the repository: they stay in the Drive (`dados/rs2024/mascaras`) and are identified by the SHA-256 of the files and of their values (`rq4_mascaras_sha256.csv`).

| File | Content |
|---|---|
| `rq4_mascaras.json` | Inputs checked by SHA-256, references used (damage source of each EMSR720 product), and for each tile: area of the UFRGS extent, mapped area of each EMSR720 product, check of alignment, and for each distance of the UFRGS mapped area (1000 m primary, 500 m and 2000 m) the counts by code, by reason (overlapping), by number of covering references and by stratum, and the summary of the windows |
| `rq4_mascaras_sha256.csv` | Size, SHA-256 of the file and SHA-256 of the values (`mascaras.sha_conteudo`) of the 12 rasters in the Drive |
| `rq4_janelas.csv` | One row per 512 x 512 window, tile and distance: offsets, fraction without input data, scored, water and land pixels assigned to the window, and whether it enters |
| `emsr720_fontes.csv` | Source table of the five EMSR720 products |
| `<tile>_mascara.png` | Code of every tenth pixel, primary distance |

Rasters in `dados/rs2024/mascaras`: `<tile>_classe_<d>m.tif` (band 1 the code of `mascaras.CODIGOS`, band 2 the number of covering references) and `<tile>_estrato.tif` (1 built-up, 2 permanent water, 3 other, from ESA WorldCover 2021).
