"""
ResNet-50 Baseline for Dermoscopic Classification.

Serves as the primary baseline in the benchmark comparison against the
custom dual-branch architecture. ResNet-50 with ImageNet pretrained weights
is the most commonly reported model in dermoscopic classification literature.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet50Baseline(nn.Module):
    """
    Fine-tuned ResNet-50 for dermoscopic multi-class classification.

    The final fully-connected layer is replaced with a custom head
    that matches the target number of classes.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet-pretrained weights.
        freeze_backbone: If True, freeze all layers except the head.
        dropout_p: Dropout probability before the final layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.4,
    ):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        in_features = backbone.fc.in_features

        # Replace classification head
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        # Returns only logits — consistent interface with baselines
        logits = self.model(x)
        return logits, None, None  # None placeholders for Grad-CAM compat

    def get_feature_extractor(self):
        """Return backbone without classification head for feature extraction."""
        return nn.Sequential(*list(self.model.children())[:-1])
