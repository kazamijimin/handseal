"""Train a HandSeal classifier and save reproducible checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .config import class_names, load_config, save_config, set_seed
from .datasets import build_loaders, class_weights
from .metrics import classification_metrics
from .models import build_model
from data_tools.split_manifest import write_manifest


def evaluate_loader(model, loader, device, names):
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(nn.functional.cross_entropy(logits, labels).item())
            batches += 1
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    metrics = classification_metrics(targets, predictions, names)
    metrics["loss"] = total_loss / max(1, batches)
    return metrics


def train_epoch(model, loader, optimizer, criterion, device, freeze_backbone=False, freeze_batchnorm=False):
    model.train()
    if freeze_backbone:
        model.features.eval()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    batches = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, config, names):
    checkpoint = {
        "epoch": epoch,
        "best_macro_f1": best_metric,
        "model_name": config["training"]["model"],
        "pretrained": bool(config["training"].get("pretrained", False)),
        "class_names": names,
        "image_size": int(config["data"]["image_size"]),
        "normalization": config["data"]["normalization"],
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/default.yaml")
    parser.add_argument("--model", choices=("small_cnn", "mobilenetv3_small"))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", help="Checkpoint path to resume from.")
    parser.add_argument(
        "--dataset-root",
        help="Override data.root in the config for this run.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("training", {})
    if args.dataset_root:
        config.setdefault("data", {})["root"] = args.dataset_root
    if args.model:
        config["training"]["model"] = args.model
    names = class_names(config)
    set_seed(int(config.get("seed", 42)), bool(config.get("deterministic", True)))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    training_config = config["training"]
    model_name = training_config.get("model", "small_cnn")
    run_name = args.run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(training_config.get("output_dir", "artifacts")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    try:
        train_loader, valid_loader, train_records = build_loaders(config)
        model = build_model(
            model_name,
            num_classes=len(names),
            pretrained=bool(training_config.get("pretrained", False)),
        ).to(device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = run_dir / "dataset_manifest.jsonl"
    manifest_rows = write_manifest(
        config["data"].get("root", "handseal_dataset"),
        names,
        manifest_path,
    )
    print(f"Dataset manifest rows: {manifest_rows}")
    train_counts = {name: 0 for name in names}
    for record in train_records:
        train_counts[record.class_name] += 1
    valid_counts = {name: 0 for name in names}
    for record in valid_loader.dataset.records:
        valid_counts[record.class_name] += 1
    print(f"Train class counts: {train_counts}")
    print(f"Valid class counts: {valid_counts}")
    weights = class_weights(train_records, names).to(device)
    freeze_backbone = bool(training_config.get("freeze_backbone", False))
    if freeze_backbone:
        if model_name != "mobilenetv3_small":
            raise SystemExit("freeze_backbone is supported only for mobilenetv3_small.")
        for parameter in model.features.parameters():
            parameter.requires_grad_(False)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    start_epoch = 1
    best_metric = -math.inf
    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_macro_f1", -math.inf))

    history = []
    patience = int(training_config.get("patience", 5))
    stale_epochs = 0
    epochs = int(training_config.get("epochs", 20))
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Run directory: {run_dir.resolve()}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, freeze_backbone,
            bool(training_config.get("freeze_batchnorm", False)),
        )
        valid_metrics = evaluate_loader(model, valid_loader, device, names)
        scheduler.step(valid_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "valid": valid_metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f}"
        )

        if valid_metrics["macro_f1"] > best_metric:
            best_metric = valid_metrics["macro_f1"]
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                config,
                names,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            config,
            names,
        )
        if stale_epochs >= patience:
            print(f"Early stopping after {patience} stale validation epochs.")
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"Best validation macro-F1: {best_metric:.4f}")
    print("Run complete. Test evaluation and export are separate commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
