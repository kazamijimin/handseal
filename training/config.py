"""Configuration and path helpers shared by training commands."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ["dragon", "ram", "snake", "tiger", "no_seal"]
DEFAULT_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path.resolve())
    return config


def save_config(config: dict, destination: str | Path) -> None:
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean_config, handle, sort_keys=False)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_root(config: dict) -> Path:
    return resolve_project_path(config.get("data", {}).get("root", "handseal_dataset"))


def class_names(config: dict) -> list[str]:
    names = config.get("data", {}).get("classes", DEFAULT_CLASSES)
    if not names:
        raise ValueError("At least one class must be configured.")
    return list(names)


def image_size(config: dict) -> int:
    return int(config.get("data", {}).get("image_size", 224))


def normalization(config: dict) -> tuple[list[float], list[float]]:
    values = config.get("data", {}).get("normalization", DEFAULT_NORMALIZATION)
    return list(values["mean"]), list(values["std"])


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
