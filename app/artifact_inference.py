"""TorchScript artifact loading and sign prediction for the battle app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class TorchScriptSignPredictor:
    """Load only an exported model artifact, not training modules."""

    def __init__(
        self,
        model,
        class_names: list[str],
        image_size: int,
        mean: list[float],
        std: list[float],
        device: str = "cpu",
        resize_mode: str = "fit",
    ) -> None:
        if not class_names:
            raise ValueError("Artifact metadata must contain class_names.")
        self.model = model
        self.model.eval()
        self.class_names = class_names
        self.image_size = int(image_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.device = torch.device(device)
        if resize_mode not in {"fit", "letterbox"}:
            raise ValueError("resize_mode must be 'fit' or 'letterbox'.")
        self.resize_mode = resize_mode
        self.model.to(self.device)

    @classmethod
    def from_metadata(
        cls,
        metadata_path: str | Path,
        device: str = "cpu",
        resize_mode: str = "fit",
    ):
        metadata_file = Path(metadata_path)
        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        model_path = Path(metadata["model_path"])
        if not model_path.is_absolute():
            model_path = metadata_file.parent / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")
        model = torch.jit.load(str(model_path), map_location=device)
        normalization = metadata.get("normalization", {})
        return cls(
            model=model,
            class_names=list(metadata["class_names"]),
            image_size=int(metadata.get("image_size", 224)),
            mean=list(normalization.get("mean", [0.485, 0.456, 0.406])),
            std=list(normalization.get("std", [0.229, 0.224, 0.225])),
            device=device,
            resize_mode=resize_mode,
        )

    def _preprocess(self, bgr_frame):
        if bgr_frame is None or len(bgr_frame.shape) != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels.")
        rgb_frame = bgr_frame[:, :, ::-1]
        image = Image.fromarray(np.ascontiguousarray(rgb_frame)).convert("RGB")
        if self.resize_mode == "fit":
            image = ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
        else:
            image.thumbnail(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            letterboxed = Image.new(
                "RGB",
                (self.image_size, self.image_size),
                color=(0, 0, 0),
            )
            offset = (
                (self.image_size - image.width) // 2,
                (self.image_size - image.height) // 2,
            )
            letterboxed.paste(image, offset)
            image = letterboxed
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, bgr_frame) -> Prediction:
        inputs = self._preprocess(bgr_frame)
        with torch.no_grad():
            logits = self.model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probabilities = torch.softmax(logits, dim=1)[0]
            index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[index].item())
        return Prediction(self.class_names[index], confidence)
