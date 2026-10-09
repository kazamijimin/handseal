"""Export a checkpoint as a self-contained TorchScript artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .models import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/exports")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    names = list(checkpoint["class_names"])
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    example = torch.randn(1, 3, int(checkpoint["image_size"]), int(checkpoint["image_size"]))
    scripted = torch.jit.trace(model, example)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{checkpoint_path.parent.name}.torchscript.pt"
    scripted.save(model_path)
    metadata = {
        "model_path": str(model_path.resolve()),
        "model_name": checkpoint["model_name"],
        "class_names": names,
        "image_size": int(checkpoint["image_size"]),
        "normalization": checkpoint["normalization"],
        "source_checkpoint": str(checkpoint_path.resolve()),
    }
    (output_dir / f"{checkpoint_path.parent.name}.metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"TorchScript model: {model_path.resolve()}")
    print(f"Metadata: {(output_dir / f'{checkpoint_path.parent.name}.metadata.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
