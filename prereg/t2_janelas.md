# T2 windows: cutting and inclusion rule

Fixed on 2026-10-10, before any window is cut or counted and before any model is run on WorldFloods v2. It applies to the maps of `splits/t2_eventos.csv`, both the core set and the full set, selected by the criteria in `prereg/t2_criterios.md`.

## Input

- Bands: indices [1, 2, 3, 8, 11, 12] of the 15-band Sentinel-2 files, that is B2, B3, B4, B8A, B11 and B12. The order was checked from the native resolution of each band (`notebooks/semana2_t2_worldfloods.ipynb`, cell 4b). The `bgriswirs` configuration of ml4floods is never used, because it takes B8.
- Values as stored: int16 digital numbers, reflectance × 10,000. The IBM recipe applies the 1e-4 scale.

## Label

From the two-channel label (`gt`):

| Condition | Value |
|---|---|
| Water channel 2 | 1 (water) |
| Water channel 1 | 0 (land) |
| Cloud channel 2, water channel 0, or no input data | −1 (ignored, the recipe's `ignore_index`) |

## Grid

- Windows of 512 × 512 pixels, the size of the Sen1Floods11 chips. Every T2 scene is at least 857 pixels on each side.
- Windows start at the top-left corner with a step of 512. The last window of each row and column is placed against the scene edge, so every window has 512 × 512 pixels of real input and no padding.
- A pixel covered by two windows is scored only in the first one, in row-major order. Every pixel is scored at most once.

## Input completeness

A window enters only if at most 1% of its pixels lack input data (all 13 Sentinel-2 bands equal to 0, the invalid-pixel definition of ml4floods). Pixels without input data are never scored.

Reason: the encoder mixes information across the whole window, so missing data can change the prediction at valid pixels. The 1% tolerance keeps a thin strip without data at a scene border from removing every window along that border.

## Minimum of scored pixels

A scored pixel has label 0 or 1 under the rules above and is assigned to the window.

- **Primary:** a window enters if at least 26,215 of its 262,144 pixels (10%) are scored. This removes slivers at the edge of the mapped area or next to clouds, where per-chip statistics are unstable.
- **Sensitivity, announced before running:** windows with at least 1 scored pixel. This is the rule used for the source chips, where the 4 Ghana chips with no valid pixel were left out of the bootstrap.

## Output

- `splits/t2_janelas.csv`: one row per window, with map, event, row and column offsets, fraction of pixels without input data, scored, water and land pixels, and whether the window enters the primary set and the sensitivity set. Committed with its hash before inference.
- Window files on Google Drive, with the same layout as the Sen1Floods11 chips: a 6-band image and a label with values 1, 0 and −1.

## Not decided here

The aggregation across events (pooled pixels or mean by event) and the cloud sensitivity at 0.50 are pre-registration decisions, made before inference.
