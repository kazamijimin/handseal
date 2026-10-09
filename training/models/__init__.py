"""Model definitions used by the HandSeal training pipeline."""

from .mobilenetv3 import build_mobilenet_v3_small
from .small_cnn import SmallCNN


def build_model(name: str, num_classes: int, pretrained: bool = False):
    """Build a supported classifier by name."""
    normalized = name.strip().lower()
    if normalized in {"small_cnn", "cnn"}:
        return SmallCNN(num_classes=num_classes)
    if normalized in {"mobilenetv3_small", "mobilenet_v3_small", "mobilenet"}:
        return build_mobilenet_v3_small(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    raise ValueError(
        f"Unsupported model '{name}'. Choose 'small_cnn' or 'mobilenetv3_small'."
    )
