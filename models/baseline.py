"""
Baseline CNN Models — Phase 3
Factory for ResNet50, DenseNet121, EfficientNet-B4 with ImageNet weights.
Uses timm for unified model loading.
"""

import timm
import torch
import torch.nn as nn
from typing import Optional


SUPPORTED_MODELS = {
    "resnet50":         "resnet50",
    "densenet121":      "densenet121",
    "efficientnet_b4":  "efficientnet_b4",
}


def get_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = True,
    drop_rate: float = 0.3,
) -> nn.Module:
    """
    Build a baseline classification model.

    Args:
        name: Model name ('resnet50', 'densenet121', 'efficientnet_b4')
        num_classes: Number of output classes
        pretrained: Load ImageNet pretrained weights
        drop_rate: Classifier dropout rate

    Returns:
        Initialized nn.Module ready for training
    """
    name = name.lower().replace("-", "_")
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: '{name}'. Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    timm_name = SUPPORTED_MODELS[name]
    weights = "imagenet" if pretrained else None

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] {name}")
    print(f"        Total params    : {n_params:,}")
    print(f"        Trainable params: {n_trainable:,}")
    print(f"        Pretrained      : {pretrained}")
    print(f"        Classes         : {num_classes}")

    return model


def count_parameters(model: nn.Module) -> dict:
    """Return parameter count statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


if __name__ == "__main__":
    for model_name in SUPPORTED_MODELS:
        print(f"\n{'='*50}")
        model = get_model(model_name, num_classes=7, pretrained=False)
        # Quick forward pass sanity check
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"        Output shape    : {out.shape}")
        print(f"{'='*50}")
