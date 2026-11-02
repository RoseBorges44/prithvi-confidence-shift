# Inventory of the candidate references of RQ4

Written by `notebooks/semana9_rq4_referencias.ipynb` (second run, 2026-11-02, folder `rq4_referencias_2`) with `analysis/rq4/inventario_referencias.py`, before any model output on Rio Grande do Sul. The areas per tile cover 22JDM and 22JCN only; 22JDN was added afterwards (see the end of `prereg/plano_rq4.md`).

| File | Content |
|---|---|
| `emsr720_api.json` | Copernicus EMS Rapid Mapping metadata of activation EMSR720: areas of interest, products and source images with acquisition times |
| `inventario_referencias.csv` | One row per layer of the UFRGS GeoPackage (Zenodo 14617475, v1.4) and of the EMSN194 package: dates in the names, features, CRS, bounds, area in each tile |
| `valores_referencias.json` | Distinct values of the low-cardinality columns of each layer, and the whole data-source table `P00_data_source` of EMSN194 |
| `zenodo_14617475.json` | Zenodo metadata of the UFRGS database (files and MD5) |
| `rodada.json` | URLs, sizes and SHA-256 of the downloaded files, versions |

## EMSR720 grading products of 2024-05-06 (`emsr720/`)

Written by `notebooks/semana9_rq4_emsr720.ipynb` (2026-11-02, folder `rq4_emsr720`) with `analysis/rq4/inventario_referencias.py`, before any model output. The areas are given in each of the three tiles (22JDM, 22JCN, 22JDN). The vector layers themselves are not in the repository: they stay in the Drive (`dados/rs2024/referencias/emsr720`) and are identified by the SHA-256 of the packages (`rodada.json`) and of the `observedEvent` files (`observedEvent_sha256.csv`).

| File | Content |
|---|---|
| `inventario_referencias.csv` | One row per layer of the five products (AOI 1 to 5): features, geometry, CRS, bounds, total area and area in each tile |
| `valores_referencias.json` | Distinct values of the low-cardinality columns of each layer (classes of `observedEventA` in the column `notation`) |
| `emsr720_conteudo.tsv` | Whole content of the five zip packages, with the size of each item |
| `rodada.json` | The five products chosen by the rule of the plan, URLs, sizes and SHA-256 of the packages, library versions |
| `observedEvent_sha256.csv` | SHA-256 of the `observedEvent` files (shp, shx, dbf, prj, json) of each AOI |

