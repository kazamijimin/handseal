# HandSeal Training Handoff

This repository contains the training setup and dataset tools for a local
Naruto hand-sign image classifier. Full training is intentionally deferred to a
machine with sufficient CPU/GPU resources.

## Scope

The configured classes are:

```text
dragon, ram, snake, tiger, no_seal
```

The package includes a local OpenCV single-sign webcam check and an optional
two-player battle prototype. It does not include mobile/web deployment,
networked matches, learned sequence recognition, or thesis materials.

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
is not changed accidentally. The improved target is five participants with
approximately 100 images per class per participant. Assign three participants
to train, one to validation, and one to test; do not mix a participant across
splits.

For meaningful generalization, collect all five classes from multiple people,
backgrounds, distances, lighting conditions, and hand positions. Use unique
participant identifiers and keep people or recording sessions from leaking
across evaluation splits where possible.

## Audit and manifest

Run these commands from `HandSealProject`:

```powershell
python -m data_tools.audit_dataset --dataset-root handseal_dataset
python -m data_tools.split_manifest --dataset-root handseal_dataset
python -m data_tools.review_dataset --dataset-root handseal_dataset
```

Outputs are written under `reports/`. The audit does not modify source images.
It reports class counts, balance, corrupt and unsupported files, dimensions,
aspect ratios, exact and near-duplicate groups, missing metadata, and
participant/session/source groups that cross splits. Use `--strict` before a
training run to fail on empty classes, missing metadata, corruption, and
leakage:

```powershell
python -m data_tools.audit_dataset --dataset-root handseal_dataset_v2 --strict
```

Contact sheets are written to `reports/dataset_review/` for manual label and
cropping review.

## Data collection

Collect one class at a time. The default output is separate from the source
dataset:

```powershell
python -m data_tools.collect_webcam --participant P01 --session S01 --split train --label no_seal --images-per-clip 100 --seconds 30
```

Every collection command requires a participant and session ID and writes
`metadata.jsonl` beside the collected split folders. Optional metadata fields
describe lighting, background, distance, and hand position. Repeat for five
participants and all five labels. For `no_seal`, capture ordinary frames
without a valid hand sign, including no hand, hands lowered, empty background,
hand outside the crop, and sign transitions.

The collector window shows the current split and label, a three-second
`STARTING IN` countdown before capture, the recording countdown, and the
number of images captured in the current clip. The countdown duration can be
changed with `--countdown-seconds`.

Keep the participant split assignment fixed, for example:

```text
P01/P02/P03 -> train
P04          -> valid
P05          -> test
```

Do not copy images into `handseal_dataset` until the metadata, audit, and
contact-sheet review are complete. A versioned dataset can be trained by
passing its root to the wrapper:

```powershell
.\training.ps1 -DatasetRoot handseal_dataset_v2
```

## Training

The default configuration is in `training/configs/default.yaml`. It uses a
224x224 input, ImageNet normalization, class-weighted cross-entropy, validation
macro-F1 model selection, and early stopping.

For the normal coworker handoff, run the root-level PowerShell wrapper. It
trains the model, evaluates the best checkpoint on the test split, and exports
TorchScript plus metadata:

```powershell
.\training.ps1
```

The default wrapper run uses `small_cnn` and automatic device selection. For a
CUDA training machine, use the transfer-learning model explicitly:

```powershell
.\training.ps1 -Model mobilenetv3_small -Device cuda -RunName mobilenetv3_v1
```

The wrapper also supports resuming from a checkpoint:

```powershell
.\training.ps1 -Resume artifacts\small_cnn_v1\last.pt -RunName small_cnn_v1_resume
```

It first runs a strict audit. If any class is empty, metadata is missing,
images are corrupt, or a split leak is detected, it stops before training. It
writes checkpoints and the exact dataset manifest under
`artifacts/<run-name>/`, test reports under `reports/`, and the exported model
plus metadata under `artifacts/exports/`.

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
recall/F1/support, `no_seal` recall and missed-rate, and a confusion-matrix
image. Balanced replacement sampling is enabled by default when class counts
remain uneven.

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

## Single-sign webcam check

For now, use the single-sign check to verify the actual hand-sign model without
any fighting, health, or jutsu logic.

You can verify the camera and crop without a model:

```powershell
python -m app.run_hand_sign --mode preview
```

This displays the webcam and the centered crop used by the dataset collector,
but it intentionally does not claim a hand-sign prediction.

After your coworker exports a TorchScript model and metadata, run real
single-sign recognition:

```powershell
python -m app.run_hand_sign --mode model --metadata artifacts/exports/<run>.metadata.json
```

The default `center` input mode uses the same centered square crop as dataset
collection. To experiment with the complete camera frame, use full-frame
letterboxing:

```powershell
python -m app.run_hand_sign `
  --mode model `
  --metadata artifacts/exports/<run>.metadata.json `
  --input-mode full
```

Full mode preserves the entire camera frame by letterboxing it into the model's
square input. It is useful for diagnosis, but it is not expected to improve
accuracy until the model is retrained with full-frame examples.

The window displays human-readable words such as `Detected hand sign: Dragon`,
the best-guess label, and confidence. Weak predictions display `Uncertain hand
sign`; `no_seal` displays `No hand sign detected`. Use
`--confidence-threshold 0.80` to change the acceptance threshold. Press `Q` or
`Esc` to close it. This is the recommended current milestone.

## Target-sign challenge

After model mode is working, start the non-combat playable challenge:

```powershell
python -m app.run_hand_sign --mode challenge --metadata artifacts/exports/<run>.metadata.json
```

The challenge randomly requests Dragon, Ram, Snake, or Tiger for ten rounds.
The user must hold the matching sign with at least 80% confidence for 0.75
seconds. A neutral `no_seal` or low-confidence transition is required before
the next attempt. Correct signs score 100 points; wrong signs do not advance
the round; each round times out after ten seconds.

The screen shows the target word, detected word, confidence, stable progress,
round, score, streak, and final accuracy/response-time summary. Use `R` to
restart, `V` to toggle optional voice, and `Q` or `Esc` to quit. Voice is off by
default and can be installed separately:

```powershell
python -m pip install -r requirements-voice.txt
```

The challenge requires an exported trained model and never uses an untrained
model as recognition evidence.

## Webcam runtime validation

Runtime validation is stored separately from training data. After exporting a
model, capture centered, multi-person, distance, background, and transition
scenarios with:

```powershell
python -m data_tools.collect_runtime_validation `
  --metadata artifacts/exports/<run>.metadata.json `
  --participant P05 `
  --session runtime01 `
  --scenario transition `
  --expected-label transition `
  --input-mode center
```

Runtime-validation images are written under `runtime_validation/` and are not
read by the training loader. Use them to check words, confidence, stable-hold
behavior, neutral transitions, false `no_seal` detections, and challenge-mode
usability.

## Later: two-player webcam battle

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
- The legacy source dataset has no participant/session metadata. It cannot be
  used as evidence of person-disjoint generalization until a participant-aware
  dataset is assembled.
- Folder splits may contain related images or capture-session leakage until the
  strict audit and manifest are run.
- The collector uses a centered square crop and therefore assumes the subject
  is inside that region.
- Full-frame inference is available as an experimental diagnostic mode, but
  the current exported model was trained on square crops.
- Camera index `0` has been validated to open and return a frame locally; the
  two-person battle behavior still requires manual testing.
- The first battle version uses fixed left/right zones and does not track people
  automatically.
- No final model checkpoint or accuracy claim is included in this milestone.
