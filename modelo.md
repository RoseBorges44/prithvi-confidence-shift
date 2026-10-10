# Model identification

| Item | Value |
|---|---|
| Checkpoint | Prithvi-EO-2.0-300M-TL-Sen1Floods11 |
| Source | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 |
| Hugging Face commit | `918b9f140bb1783716664a2421ea3253d806017d` (2026-01-15, "remove always apply") |
| Weights file | `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` |
| SHA-256 of the weights | `3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60` |
| SHA-256 of `config.yaml` | `ecc9ba411c7e2ca0be70bb597e0aae45d866acaf1fdfd2f4b8ac652ee5eab0b0` |
| TerraTorch version used for inference | Source scores: 1.2.7. Bolivia run: 1.2.11. Both inferred, with the same inference code (see below) |
| Python, NumPy, torchgeo | Source scores: 3.12, 2.2.6, 0.9.0. Bolivia run: 3.13, 2.2.6, 0.7.1 |
| Recorded on | 2026-10-04 (TerraTorch versions added 2026-10-10) |

## Evidence

- The weights were downloaded on 2026-05-24 with `snapshot_download`. The download record that `huggingface_hub` keeps in the model folder (`.cache/huggingface/download/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt.metadata`) names commit `918b9f1` and the file hash `3675e9c2…`.
- The local `config.yaml` has the SHA-256 above and is identical to the `config.yaml` of commit `918b9f1` only.
- The same weights file (`3675e9c2…`) is served by every commit from `e54b97f` (2025-09-03) to `918b9f1` (2026-01-15). Commit `91ce9d3` (2026-08-25, "Migrate off deprecated decoder_scale_modules") replaced it with a different file (`76eed77d…`). Loading with `revision="main"` today returns the new file, not the one used in this work.
- The source calibration scores (`cal_scores.npy`, 2026-06-07) were produced before that change. The calibration notebook and the Bolivia notebook (2026-09-24) both pass the local weights file in the model folder to `LightningInferenceModel.from_config`, so neither downloaded the fine-tuned weights again.

## TerraTorch version

Neither notebook prints the installed TerraTorch version. It was determined as follows.

- **Source scores, 1.2.7.** The calibration notebook installs `terratorch` without a version, so pip installs the latest release. Its saved install log shows a 592.6 kB download, the size of the 1.2.7 wheel (the 1.2.8 wheel is 605.3 kB). The same log installs lightning 2.6.4, released on 2026-05-20 and replaced by 2.6.5 on 2026-05-27. TerraTorch 1.2.7 was the latest release in that window; 1.2.8 came out on 2026-05-29.
- **Bolivia run, 1.2.11.** The notebook installs `terratorch<1.2.12` and `torchgeo>=0.7.0,<0.7.2`. TerraTorch 1.2.11 is the latest release allowed by that pin, and releases 1.2.8 to 1.2.11 declare the same dependencies. The pin is needed because release 1.2.12 removed `scale_modules` from the UperNet decoder, and the `config.yaml` of commit `918b9f1` still uses it.
- Both notebooks log the warning `scale_modules is deprecated` from line 39 of `upernet_decoder.py`. That warning exists at that line in releases 1.2.7 to 1.2.11.
- **Same inference code.** From 1.2.8 to 1.2.11, every file on the inference path (Prithvi backbone, UperNet decoder, encoder-decoder factory, segmentation task, `LightningInferenceModel`, Sen1Floods11 datamodule and transforms) has identical content. From 1.2.7 to 1.2.8 that path has two changes, and neither affects the scores. In `prithvi_mae.py`, a `.view` becomes `.reshape` in the positional-embedding interpolation, which yields the same values. In `datamodules/utils.py`, the `Normalize` class gains an optional argument whose default keeps the previous behavior, and the Sen1Floods11 datamodule normalizes with `kornia.augmentation.Normalize` rather than with this class.
- If `cal_scores.npy` was rewritten on 2026-06-07, the date of the file on Drive, that run would have installed 1.2.8, which has the same inference code.
- Loading the model also downloads the pretrained backbone `Prithvi_EO_V2_300M_TL.pt` from the Hugging Face Hub without a fixed revision. This does not affect the scores, because `LightningInferenceModel` then loads the fine-tuned checkpoint with a strict `load_state_dict`, which replaces every weight.

## Loading this exact version

Install TerraTorch with `pip install terratorch==1.2.11`. Releases 1.2.12 and later cannot load this `config.yaml`.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
    revision="918b9f140bb1783716664a2421ea3253d806017d",
)
```

Check the SHA-256 of the downloaded file against the table before using it.
