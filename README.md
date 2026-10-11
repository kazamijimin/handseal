# handseal

## Included model

The `training` branch includes the selected retrained model (about 6.53 MB) and its metadata under `artifacts/exports/`. A normal Git pull downloads both files; the model files use ordinary Git and inference does not require the datasets.

After installing the project dependencies, run from the repository root:

```powershell
python -m app.run_hand_sign --mode model --metadata artifacts/exports/collected_mobilenetv3_v1.metadata.json
```

The metadata resolves the model path relative to itself, so the pair can be copied together to another folder. Training checkpoints and datasets remain excluded from Git.

This is an experimental model: test macro F1 is 0.364, most correct predictions are `no_seal`, and Dragon/Ram had zero recall on that test set. See [collected-data retraining](docs/training/COLLECTED_RETRAIN.md) for the data audit, results, and reproduction steps. The previous model remains available as `external_mobilenetv3_v1` for comparison; its [training results](docs/training/RESULTS.md) and [improvement experiments](docs/training/IMPROVEMENTS.md) are historical.
