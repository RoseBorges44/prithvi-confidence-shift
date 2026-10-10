# Second target set (T2): selection criteria

Fixed on 2026-10-10. These criteria and thresholds are committed before the country and cloud fractions are computed, and before any model is run on WorldFloods v2. They are applied by cells 5 and 6 of `notebooks/semana2_t2_worldfloods.ipynb`, whose thresholds must match this file. The resulting list goes to `splits/t2_eventos.csv`.

## Source

- WorldFloods v2, test split (Portalés-Julià et al., *Scientific Reports* 13, 20316, 2023).
- Hugging Face `isp-uv-es/WorldFloodsv2`, revision `1f3faa2989e69930ac31d5c0a79fd224461123f8` (2025-07-31). Licence CC BY-NC 4.0.
- 18 flood maps from 11 Copernicus EMS activations. Each map has a Sentinel-2 L1C scene with 13 bands, a two-channel label (`gt`), the JRC permanent water layer and a metadata file.

## Units

- **Map:** one Sentinel-2 scene with its label.
- **Event:** one Copernicus EMS activation code (`ems_code`). Maps of the same activation belong to the same event. The bootstrap resamples events first and chips within events.

## Criteria

Applied only to the metadata table and to the label rasters. No model output is used.

| # | Criterion | Rule | Threshold |
|---|---|---|---|
| C1 | Country | Exclude a map if more than this fraction of its labeled pixels (label water channel not 0) falls inside a country with events in Sen1Floods11, using Natural Earth 1:10m admin-0 polygons | 0.05 |
| C2 | Alignment | Core: the reference map was made from a Sentinel-2 image taken on the same UTC date as the scene (`satellite`, `satellite date` and `s2_date` in `dataset_metadata.csv`) | none |
| C3 | Cloud | Exclude a map if cloud pixels (label cloud channel = 2) exceed this fraction of its valid pixels (cloud channel not 0) | 0.30 |

**Core set:** maps passing C1, C2 and C3. The primary T2 analysis uses only these.

**Full set:** maps passing C1 and C3, with or without C2. Reported separately, as a secondary descriptive analysis.

Countries with events in Sen1Floods11 (Natural Earth `ADM0_A3`): BOL, ESP, GHA, IND, LKA, NGA, PAK, PRY, SOM (with SOL, Somaliland, which Natural Earth separates), USA, and, for the Mekong event, the four countries of the lower basin, KHM, LAO, THA and VNM, as a precaution.

## Reasons for the thresholds

- **C1, 0.05.** A tolerance for border generalization in the Natural Earth polygons. A map with a small strip across a border stays; a map with a real part inside a training country goes. The rule is applied the same way to every map, so the case of EMSR466 (activation for Niger, scene reaching the border with Nigeria) is decided by it and not case by case.
- **C3, 0.30.** Cloud pixels are masked in the evaluation, but the WorldFloods v2 label has no cloud shadow class (`ml4floods/data/create_gt.py`), and shadow area grows with cloud cover. Shadow over land can look like water to the model and would count as a confident error caused by the label, which biases the comparison toward H1.

## Known before applying, from the metadata only

- EMSR422 (Spain) is expected to fail C1.
- By C2, nine maps have a same-day Sentinel-2 reference besides EMSR466: EMSR286 (two maps), EMSR333 (two), EMSR342 North Normanton, EMSR347 (three maps, with Zomba on two dates) and EMSR438 AOI06. EMSR422 has a same-day reference from Pléiades, not Sentinel-2.
- The two Zomba maps cover the same area on 2019-03-10 and 2019-03-12. Both stay, and their chips are not independent within the event.
- The inspection in part A of the notebook (cell 4) listed the values present in each label channel. It showed that three maps have no cloud pixels at all: EMSR273, EMSR347 Zomba on 2019-03-10 and EMSR419. No cloud or country fraction was computed before this commit.

## Left for a later commit, before inference

Window size and stride, the rule for keeping a window, and the binary label (water against land, with cloud and invalid pixels ignored).
