"""Evaluate a saved HandSeal checkpoint on one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .datasets import build_dataset
from .metrics import classification_metrics, save_confusion_matrix, write_metrics
from .models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="training/configs/default.yaml")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    names = list(checkpoint["class_names"])
    config["data"]["classes"] = names
    config["data"]["image_size"] = int(checkpoint["image_size"])
    config["data"]["normalization"] = checkpoint["normalization"]
    dataset = build_dataset(config, args.split, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 0)),
    )
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    metrics = classification_metrics(targets, predictions, names)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "split": args.split,
            "device": str(device),
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{checkpoint_path.parent.name}_{args.split}"
    model_name = checkpoint["model_name"]
    write_metrics(metrics, output_dir / f"{stem}.json")
    save_confusion_matrix(
        metrics["confusion_matrix"],
        names,
        output_dir / f"{stem}_confusion_matrix.png",
        f"{model_name} - {args.split}",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
