# Plan of RQ4: Rio Grande do Sul, 2024-05-06

**Status.** Written on 2026-10-31, after the confirmatory run and before any model output on Rio Grande do Sul exists. RQ4 is descriptive (`preregistro.md`, section 1): no hypothesis, no test, no category. Section 9 fixes its measure: agreement conditioned on confidence against a consensus mask of the available references, with the selection bias stated. **No statement about the model's errors in Rio Grande do Sul is made.** The word used is disagreement, never error.

**Rules.** Every measure below is reported, whatever it shows. Adjustments forced by the data (a reference that is not as described here, a missing layer) are allowed only before any model output on Rio Grande do Sul exists, and each is added at the end with its date and reason. After the first model output, changes are exploratory and labeled so.

## 1. Input

- **Scenes.** Sentinel-2 L1C granules of tiles 22JDM (Porto Alegre, Canoas, São Leopoldo) and 22JCN (Encantado, Roca Sales) sensed on 2024-05-06 (UTC). If a tile has more than one granule that day, the one with the lowest cloud cover. Product IDs and the SHA-256 of the exported files are recorded.
- **Source and offset.** Google Earth Engine `COPERNICUS/S2_HARMONIZED`, which removes the offset of 1,000 added to L1C digital numbers from processing baseline 04.00 (January 2022) on. Sen1Floods11 (2016 to 2019) and the WorldFloods v2 target events (2019 and 2020) predate that offset, so the values seen by the model stay on the same scale as in training.
- **Bands and grid.** B2, B3, B4, B8A, B11, B12, exported at 10 m in the tile's UTM grid; the 20 m bands by nearest neighbour (the Earth Engine default, as in the Sen1Floods11 export). Scale 1e-4 and the Sen1Floods11 normalization of TerraTorch, as in training.
- **Windows.** The grid rule of `prereg/t2_janelas.md` applied to the whole tile: 512 × 512 windows from the top-left corner, the last row and column against the tile edge, each pixel scored at most once; a window enters if at most 1% of its pixels lack input data and at least 10% of its pixels are scored (section 3 below).
- **Inference.** The code of `notebooks/semana6_inferencia_alvos.ipynb`, 32-bit, all 15 confirmatory runs (R1, R2, R3, seeds 1 to 5). τ of each run is the one of the confirmatory run (`execucoes.csv`). Nothing is fitted on Rio Grande do Sul.

## 2. References

- **Which enter.** A reference enters only if its source image was acquired on 2024-05-06, according to its own metadata. Expected, to be confirmed when downloaded: the UFRGS Planet/SkySat layer of Porto Alegre (Zenodo 14617475) for 22JDM; the Copernicus EMS EMSR720 delineation of Encantado from Pléiades Neo for 22JCN; EMSN194 (Porto Alegre) if its source date is 2024-05-06. A product from another date, or an aggregate of several dates such as CEMADEN Nota Técnica 469/2024, does not enter and is listed with the reason.
- **Rasterization.** Polygons to the 10 m grid of the tile, pixel-centre rule. Water = observed flood extent of the reference ∪ permanent water (JRC Global Surface Water 1.4, occurrence ≥ 90%). Land = inside the reference's mapped area and not water. Outside the mapped area the reference says nothing. A "mud" or "flood trace" class, if present, is left out (unscored).
- **Consensus.** A pixel is water (or land) in the consensus if every reference that covers it says water (or land). Pixels where covering references disagree form a separate group, counted and not scored. Each scored pixel keeps the number of references covering it (1, or 2 or more).
- **Not scored.** Pixels within 2 pixels of a consensus water–land boundary (the `borda` rule of the inference notebook); cloud (s2cloudless probability ≥ 40%, `COPERNICUS/S2_CLOUD_PROBABILITY`) or cloud shadow, cloud or cirrus in the scene classification of the same granule's L2A product (`SCL` 3, 8, 9, 10); pixels without input data; pixels outside every reference.

## 3. Measures

For each run, then the mean and range over the 5 seeds of each regime, for each tile and for both tiles pooled:

- disagreement rate, P(prediction ≠ consensus);
- AUROC_d: the AUROC_e of `nucleo.py` with disagreement in place of error;
- unflagged disagreement rate, P(MSP ≥ τ | disagreement), and empty-set rate, P(MSP < τ);
- disagreement rate among flagged and among unflagged pixels.

Each also by stratum (ESA WorldCover 2021 built-up, permanent water, other) and by the number of covering references. **Comparison with the labeled events:** each value is reported next to the value of the same regime on the pooled target and next to the range of the five target events (`regimes.csv`, `eventos.csv`). Compatible or not is described, not tested. Maps of MSP, flags and disagreement for both areas.

## 4. Selection bias, stated with the results

References exist where mapping was requested; same-day mapping is optical or very-high-resolution imagery, so cloud removes areas (22JDM had about 36% cloud); boundaries and disagreements between references are removed, which leaves easier pixels; flooded urban areas are hard for optical sensors and for reference maps alike. The agreement measured here is with imperfect references, not with ground truth.

## 5. If references are missing

If no same-day reference can be obtained for a tile, that tile is reported only as maps of confidence and flags, without any agreement measure, and this is recorded below.

## Changes to this plan

None yet.
