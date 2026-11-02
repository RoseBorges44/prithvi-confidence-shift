# Inventory of the candidate references of RQ4

Written by `notebooks/semana9_rq4_referencias.ipynb` (second run, 2026-11-02, folder `rq4_referencias_2`) with `analysis/rq4/inventario_referencias.py`, before any model output on Rio Grande do Sul. The areas per tile cover 22JDM and 22JCN only; 22JDN was added afterwards (see the end of `prereg/plano_rq4.md`).

| File | Content |
|---|---|
| `emsr720_api.json` | Copernicus EMS Rapid Mapping metadata of activation EMSR720: areas of interest, products and source images with acquisition times |
| `inventario_referencias.csv` | One row per layer of the UFRGS GeoPackage (Zenodo 14617475, v1.4) and of the EMSN194 package: dates in the names, features, CRS, bounds, area in each tile |
| `valores_referencias.json` | Distinct values of the low-cardinality columns of each layer, and the whole data-source table `P00_data_source` of EMSN194 |
| `zenodo_14617475.json` | Zenodo metadata of the UFRGS database (files and MD5) |
| `rodada.json` | URLs, sizes and SHA-256 of the downloaded files, versions |
