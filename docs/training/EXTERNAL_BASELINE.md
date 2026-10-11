# External dataset training baseline

Work branch: training, tracking origin/training on https://github.com/kazamijimin/handseal.

This run trains the five configured classes using original Naruto photographs and HaGRIDv2 natural-hand negative samples. It is exploratory: the Naruto source does not provide participant/session IDs, and Kaggle lists its license as Unknown. No person-disjoint generalization or webcam performance claim is made. HaGRID uses a custom attribution/share-alike-style license; preserve its actual license document, available at https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf.

## Prepare the data

Download the source archives into local_setup/datasets/hagrid/{no_gesture.zip,annotations.zip} and local_setup/datasets/naruto_kaggle/naruto-hand-sign-dataset.zip:

- https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagrid_v2_zip/no_gesture.zip
- https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
- https://www.kaggle.com/api/v1/datasets/download/vikranthkanumuru/naruto-hand-sign-dataset

Use the repository root as the working directory:

```powershell
.\.venv\Scripts\python.exe scripts/import_hagrid.py
.\.venv\Scripts\python.exe scripts/import_naruto.py
.\.venv\Scripts\python.exe -m data_tools.prepare_external_dataset
```

The importer keeps the downloaded source splits and full HaGRID subject annotations. The preparer reads all supported image extensions, uses four Naruto labels, and retains the original zero label only in the raw import for review. It crops HaGRID around the union of annotated natural hands with context, then converts both sources to centered 224-pixel RGB images. It samples one negative image per known subject, joins source/hash/near-duplicate families transitively, and keeps each family in one split. HaGRID negatives conflicting with their published split are excluded rather than reassigned. Unknown capture fields stay null.

The versioned output refuses to overwrite an existing dataset. Seed 42 produces 1,185 images; 16 negatives are excluded to preserve native split assignments:

| Split | Dragon | Ram | Snake | Tiger | no_seal | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 117 | 93 | 115 | 132 | 395 | 852 |
| Valid | 20 | 15 | 18 | 20 | 96 | 169 |
| Test | 19 | 14 | 18 | 20 | 93 | 164 |

The exploratory gate allows only missing capture fields (participant, session, lighting, background, distance, hand position). It still rejects missing metadata rows or core image metadata, empty classes, corruption, unsupported images, metadata mismatches, and detected leakage. The existing participant-strict audit and training.ps1 are unchanged and intentionally reject this baseline because its capture metadata is incomplete.

## Train, evaluate, and export

Run the exploratory gate and require a zero exit code before training:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; from data_tools.audit_dataset import audit_dataset, DEFAULT_CLASSES; from data_tools.prepare_external_dataset import baseline_issues; issues=baseline_issues(audit_dataset(Path('handseal_dataset_external_v2'), DEFAULT_CLASSES)); print(issues); raise SystemExit(1 if issues else 0)"
```

Then run these commands in sequence, requiring each to succeed:

```powershell
.\.venv\Scripts\python.exe -m training.train --config training/configs/external_baseline.yaml --run-name external_mobilenetv3_v1 --device cuda
.\.venv\Scripts\python.exe -m training.evaluate --config artifacts/external_mobilenetv3_v1/config.yaml --checkpoint artifacts/external_mobilenetv3_v1/best.pt --split test --output-dir reports/external_baseline_v1 --device cuda
.\.venv\Scripts\python.exe -m training.export --checkpoint artifacts/external_mobilenetv3_v1/best.pt --output-dir artifacts/exports
```

The configuration uses ImageNet MobileNetV3-Small weights, class-weighted cross-entropy without simultaneous balanced oversampling, validation macro-F1 selection, and early stopping. The test split is evaluated after model selection and is not used for tuning. Do not reuse the same run name for a new experiment, since the existing trainer writes into its run directory.

Raw archives, images, private subject identifiers, checkpoints, and local output artifacts are ignored by Git. The source manifests and audit evidence remain beside the dataset/run on this PC. Published result summaries should contain aggregate counts and limitations, not private local paths or subject IDs.

## Recognition check

After export, use the metadata with the existing app:

```powershell
.\.venv\Scripts\python.exe -m app.run_hand_sign --mode model --metadata artifacts/exports/external_mobilenetv3_v1.metadata.json
```

A camera is needed for live validation, but not for public dataset collection or training. Source differences and unknown Naruto identities limit the meaning of held-out scores; cropped natural-hand negatives also do not cover all empty backgrounds or transitions.
