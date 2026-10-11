"""Dataset records, preprocessing, and DataLoader construction."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import class_names, dataset_root, image_size, normalization


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    class_name: str
    split: str


def list_records(
    root: Path,
    split: str,
    names: list[str],
    require_non_empty: bool = False,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label, name in enumerate(names):
        class_dir = root / split / name
        files = (
            sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if class_dir.exists()
            else []
        )
        if require_non_empty and not files:
            raise ValueError(
                f"No images found for class '{name}' in split '{split}' at {class_dir}."
            )
        records.extend(
            ImageRecord(path, label, name, split)
            for path in files
        )
    return records


def tensor_from_image(image: Image.Image, mean: list[float], std: list[float]):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


class HandSealTransform:
    def __init__(
        self,
        size: int,
        mean: list[float],
        std: list[float],
        training: bool,
        robust_augmentation: bool = False,
    ) -> None:
        self.size = size
        self.mean = mean
        self.std = std
        self.training = training
        self.robust_augmentation = robust_augmentation

    def __call__(self, image: Image.Image):
        image = ImageOps.fit(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        if self.training:
            if self.robust_augmentation:
                # Zoom out before translating: preserve fingers rather than randomly crop them.
                side = round(self.size * random.uniform(0.85, 0.95))
                margin = self.size - side
                canvas = Image.new('RGB', (self.size, self.size), image.getpixel((0, 0)))
                canvas.paste(image.resize((side, side), Image.Resampling.BILINEAR),
                             (random.randint(0, margin), random.randint(0, margin)))
                image = canvas
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.65, 1.35))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.35))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
                if random.random() < 0.2:
                    image = ImageOps.grayscale(image).convert('RGB')
            if random.random() < 0.35:
                image = image.rotate(
                    random.uniform(-8.0, 8.0),
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
            if random.random() < 0.25:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.85, 1.15)
                )
            if random.random() < 0.25:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.85, 1.15)
                )
        return tensor_from_image(image, self.mean, self.std)


class HandSealDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: HandSealTransform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, record.label


def class_weights(records: list[ImageRecord], names: list[str]) -> torch.Tensor:
    counts = torch.zeros(len(names), dtype=torch.float32)
    for record in records:
        counts[record.label] += 1
    if torch.any(counts == 0):
        missing = [names[index] for index, value in enumerate(counts) if value == 0]
        raise ValueError(
            "Training requires at least one image per class. Missing: "
            + ", ".join(missing)
        )
    return counts.sum() / (len(names) * counts)


def sample_weights(records: list[ImageRecord]) -> torch.Tensor:
    """Return inverse-frequency weights for balanced replacement sampling."""
    counts = Counter(record.label for record in records)
    if not records or any(count == 0 for count in counts.values()):
        raise ValueError("Cannot build balanced sampling weights for an empty dataset.")
    return torch.tensor(
        [1.0 / counts[record.label] for record in records],
        dtype=torch.double,
    )


def build_dataset(config: dict, split: str, training: bool = False) -> HandSealDataset:
    names = class_names(config)
    records = list_records(
        dataset_root(config),
        split,
        names,
        require_non_empty=training,
    )
    if not records:
        raise ValueError(f"Split '{split}' contains no supported image files.")
    mean, std = normalization(config)
    transform = HandSealTransform(image_size(config), mean, std, training,
                                  bool(config.get('data', {}).get('robust_augmentation', False)))
    return HandSealDataset(records, transform)


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, list[ImageRecord]]:
    names = class_names(config)
    root = dataset_root(config)
    train_records = list_records(root, "train", names, require_non_empty=True)
    valid_records = list_records(root, "valid", names, require_non_empty=True)
    train_dataset = HandSealDataset(
        train_records,
        HandSealTransform(
            image_size(config),
            *normalization(config),
            training=True,
            robust_augmentation=bool(config.get('data', {}).get('robust_augmentation', False)),
        ),
    )
    valid_dataset = HandSealDataset(
        valid_records,
        HandSealTransform(
            image_size(config),
            *normalization(config),
            training=False,
        ),
    )
    loader_config = config.get("training", {})
    batch_size = int(loader_config.get("batch_size", 32))
    workers = int(loader_config.get("num_workers", 0))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    use_balanced_sampling = bool(loader_config.get("balanced_sampling", True))
    sampler = (
        WeightedRandomSampler(
            sample_weights(train_records),
            num_samples=len(train_records),
            replacement=True,
            generator=generator,
        )
        if use_balanced_sampling
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, train_records
