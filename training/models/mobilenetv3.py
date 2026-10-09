"""MobileNetV3-Small adapter with an optional torchvision dependency."""

from __future__ import annotations


def build_mobilenet_v3_small(num_classes: int, pretrained: bool = False):
    """Build MobileNetV3-Small and replace its classifier head.

    Importing torchvision is intentionally deferred so dataset auditing and the
    small-CNN smoke test can run before the coworker's training environment is
    installed.
    """
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise RuntimeError(
            "MobileNetV3-Small requires a compatible torchvision installation. "
            "Install matching torch and torchvision versions from the training "
            "environment instructions."
        ) from exc

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = model.classifier[3].__class__(
        model.classifier[3].in_features,
        num_classes,
    )
    return model
