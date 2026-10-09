# HandSeal Training Handoff

This repository contains the training setup and dataset tools for a local
Naruto hand-sign image classifier. Full training is intentionally deferred to a
machine with sufficient CPU/GPU resources.

## Scope

The configured classes are:

```text
dragon, ram, snake, tiger, no_seal
```

The package includes a local OpenCV two-player battle prototype. It does not
include mobile/web deployment, networked matches, learned sequence recognition,
or thesis materials.

Feed, QR, and Organizer functionality are outside this project scope and must
not be changed.

## Environment

Use Python 3.12 or a compatible supported Python version in a fresh virtual
environment. From this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a matching PyTorch and torchvision pair for the target machine. For a
CUDA machine, use the official PyTorch wheel command for that CUDA version;
then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

Run the fast setup tests before starting a long training job:

```powershell
python -m unittest discover -s tests -v
```

## Dataset expectations

The source dataset is expected at:

```text
handseal_dataset/
  train/<class>/*.jpg
  valid/<class>/*.jpg
  test/<class>/*.jpg
```

The current source dataset has an empty `no_seal` class. Training will stop
until every configured class has at least one training and validation image.
Collect new samples into `handseal_collected/` by default so the source dataset
is not changed accidentally.

For meaningful generalization, collect all five classes from multiple people,
backgrounds, distances, lighting conditions, and hand positions. Use unique
participant identifiers and keep people or recording sessions from leaking
across evaluation splits where possible.

## Audit and manifest

Run these commands from `HandSealProject`:

```powershell
python -m data_tools.audit_dataset --dataset-root handseal_dataset
python -m data_tools.split_manifest --dataset-root handseal_dataset
```

Outputs are written under `reports/`. The audit does not modify source images.
It reports class counts, corrupt files, dimensions, exact duplicate groups, and
source-key groups that cross splits.

## Data collection

Collect one class at a time. The default output is separate from the source
dataset:

```powershell
python -m data_tools.collect_webcam --participant P01 --split train --label no_seal
python -m data_tools.collect_webcam --participant P01 --split valid --label no_seal
```

Repeat for multiple participants and all five labels. For `no_seal`, capture
ordinary frames without a valid hand sign, including different backgrounds and
hand positions.

## Training

The default configuration is in `training/configs/default.yaml`. It uses a
224x224 input, ImageNet normalization, class-weighted cross-entropy, validation
macro-F1 model selection, and early stopping.

Train the baseline:

```powershell
python -m training.train --config training/configs/default.yaml --model small_cnn --run-name small_cnn_v1
```

Train the transfer-learning model:

```powershell
python -m training.train --config training/configs/default.yaml --model mobilenetv3_small --run-name mobilenetv3_small_v1 --device cuda
```

Use `--device cpu` when CUDA is unavailable. MobileNetV3 pretrained weights
may need to be downloaded or pre-cached on the training machine. Set
`training.pretrained: false` if the machine cannot access them.

Each run writes `last.pt`, `best.pt`, `config.yaml`, `history.json`, and
`class_names.json` under `artifacts/<run-name>/`.

## Resume training

Resume from the last checkpoint using the same configuration and model:

```powershell
python -m training.train --config artifacts/small_cnn_v1/config.yaml --model small_cnn --resume artifacts/small_cnn_v1/last.pt --run-name small_cnn_v1_resume
```

## Evaluation

Evaluate the held-out test split only after model selection:

```powershell
python -m training.evaluate --config training/configs/default.yaml --checkpoint artifacts/small_cnn_v1/best.pt --split test
```

Reports include accuracy, balanced accuracy, macro-F1, per-class precision/
recall/F1/support, and a confusion-matrix image.

Model selection should prioritize macro-F1 and per-class recall. Do not tune on
the test split.

## Export

Export the selected checkpoint as a self-contained TorchScript artifact:

```powershell
python -m training.export --checkpoint artifacts/small_cnn_v1/best.pt
```

The export produces a `.torchscript.pt` file and metadata containing the class
mapping, input size, normalization, model name, and source checkpoint. The
battle application loads this artifact rather than importing training loops or
training-only modules.

## Two-player webcam battle

Install the dependencies first, then run the model-free demo:

```powershell
python -m app.run_battle --mode demo
```

The demo uses the real webcam for the background and player zones while
keyboard inputs simulate sign predictions:

```text
Player 1: 1=dragon, 2=ram, 3=snake, 4=tiger, 5=no_seal
Player 2: 7=dragon, 8=ram, 9=snake, 0=tiger, -=no_seal
R = reset match, Q or Esc = quit
```

Hold a sign key long enough for the stable-hold rule to lock it. Set each
player to `no_seal` between the first and second signs. The starter jutsu table
is in `app/battle/jutsu.yaml`; unknown ordered pairs deal zero damage.

After your coworker exports a TorchScript model and metadata, run real model
mode with:

```powershell
python -m app.run_battle --mode model --metadata artifacts/exports/<run>.metadata.json
```

Model mode uses the same battle state machine as demo mode. It only replaces the
keyboard prediction source with the exported artifact.

## Local validation boundary

The laptop-side setup validation should be limited to imports, configuration,
dataset auditing, model construction, one-batch smoke tests, and command
interface checks. Do not treat local smoke-test output as model quality,
generalization, latency, or final training evidence.

## Known limitations

- The current dataset is imbalanced and has no populated `no_seal` class.
- Folder splits may contain related images or capture-session leakage until the
  audit report confirms otherwise.
- The collector uses a centered square crop and therefore assumes the subject
  is inside that region.
- Camera index `0` has been validated to open and return a frame locally; the
  two-person battle behavior still requires manual testing.
- The first battle version uses fixed left/right zones and does not track people
  automatically.
- No final model checkpoint or accuracy claim is included in this milestone.
