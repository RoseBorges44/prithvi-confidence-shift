# Model identification

| Item | Value |
|---|---|
| Checkpoint | Prithvi-EO-2.0-300M-TL-Sen1Floods11 |
| Source | https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 |
| Hugging Face commit | `918b9f140bb1783716664a2421ea3253d806017d` (2026-01-15, "remove always apply") |
| Weights file | `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` |
| SHA-256 of the weights | `3675e9c2b52547de8ff8a19f4881c28573e6d4d2f0805d866f4fc48c1e517d60` |
| SHA-256 of `config.yaml` | `ecc9ba411c7e2ca0be70bb597e0aae45d866acaf1fdfd2f4b8ac652ee5eab0b0` |
| TerraTorch version used for inference | to fill |
| Recorded on | 2026-10-04 |

## Evidence

- The weights were downloaded on 2026-05-24 with `snapshot_download`. The download record that `huggingface_hub` keeps in the model folder (`.cache/huggingface/download/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt.metadata`) names commit `918b9f1` and the file hash `3675e9c2…`.
- The local `config.yaml` has the SHA-256 above and is identical to the `config.yaml` of commit `918b9f1` only.
- The same weights file (`3675e9c2…`) is served by every commit from `e54b97f` (2025-09-03) to `918b9f1` (2026-01-15). Commit `91ce9d3` (2026-08-25, "Migrate off deprecated decoder_scale_modules") replaced it with a different file (`76eed77d…`). Loading with `revision="main"` today returns the new file, not the one used in this work.
- The source calibration scores (`cal_scores.npy`, 2026-06-07) were produced before that change. The Bolivia run (2026-09-24) loaded the local file from the model folder without downloading it again (inferred from the notebook code).

## Loading this exact version

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
    revision="918b9f140bb1783716664a2421ea3253d806017d",
)
```

Check the SHA-256 of the downloaded file against the table before using it.
