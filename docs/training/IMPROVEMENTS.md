# Improvement experiments â€” 2026-10-11

No new run outperformed the previous validation macro-F1 of 0.3733. The selected artifact remains `artifacts/exports/external_mobilenetv3_v1.torchscript.pt`. No additional test evaluation was performed and no real-camera improvement is claimed.

| Run | Epochs | Best epoch | Validation macro-F1 |
|---|---:|---:|---:|
| external_frozen_v1 | 8 | 1 | 0.1449 |
| external_fixed_bn_v1 | 13 | 6 | 0.2839 |
| external_enriched_v1 | 8 | 1 | 0.2974 |
| external_enriched_low_lr_v1 | 9 | 2 | 0.3322 |
| external_enriched_augmented_v1 | 8 | 1 | 0.2605 |

## Changes and validation

- Optional `training.freeze_backbone` freezes MobileNet features and their running statistics, while training its classifier.
- Optional `training.freeze_batchnorm` retains pretrained running statistics while allowing weights to learn.
- Optional `data.robust_augmentation` adds train-only zoom out, translation, lighting/color variation and occasional grayscale. No horizontal flips were added. Validation preprocessing is unchanged.
- Added 400 photos from Lucas Fernando (100 per target class) to training only. All frames from the new source stay together in training; native source validation/test names are not used to split this author across our evaluation sets.
- Expanded dataset: 1,585 images; training counts dragon 217, ram 193, snake 215, tiger 232, no_seal 395.
- Original validation (169) and test (164) files were verified byte-identical. Full expanded-dataset audit found no corrupt files, metadata mismatches or detected cross-split exact/near/source/known-participant leakage. Unknown participant/session data remains unknown, and strict participant-level audit requirements still fail.
- Downloaded ZIP passed CRC verification; archive SHA-256 and source tree SHA are recorded in enrichment_source.json. No new source image or trained binary is published in Git.

## Reproduction

```powershell
.venv/Scripts/python.exe -m scripts.import_lucas --archive local_setup/datasets/lucas/source.zip --output handseal_dataset_external_v3
.venv/Scripts/python.exe -m training.train --config training/configs/external_enriched_low_lr.yaml --run-name external_enriched_low_lr_v1 --device cuda
```

Run the full dataset audit and check `baseline_issues` before training on a newly prepared output. Keep existing outputs for comparison; the importer refuses to overwrite. The other checked-in experiment configs reproduce the table. These are fresh runs. If using `--resume`, retain the same model and freezing options; the existing trainer starts a fresh history/patience segment on each invocation.

## Sources and remaining limitations

Lucas Fernando public source: https://github.com/lucasfernandoprojects/hand-sign-detection . Repository LICENSE is GPL-3.0; no separate image license was found. Original labels were checked against its code/config.yaml and per-image annotations. The archive is retained locally with its license. The new collection adds one author/capture setting, rather than many independent people.

The sampled original training positives showed one repeated person/scene; validation positives showed other people and lighting. Source appearance confounding is plausible, but these experiments do not isolate or prove it. The next data priority is more independently captured positive signs and matched negatives; a larger model or more epochs has not been justified by these results.

Roboflow candidate https://universe.roboflow.com/adityas-workshop-kr2u8/naruto-hand-sign-cyq8u/dataset/4 returned an API-key requirement (401); it was not downloaded, and its contents/license/independence have not been verified. The user requested account-free downloads. Other inspected repositories exposed demo assets or reused the existing Kaggle collection rather than an additional downloadable photo collection.
