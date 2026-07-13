"""
EfficientNet-B4 Baseline for Dermoscopic Classification.

EfficientNet scales depth, width, and resolution using a compound coefficient,
achieving highly competitive accuracy with fewer parameters. It is often the
strongest single-backbone baseline but struggles with severe class imbalance.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class EfficientNetBaseline(nn.Module):
    """
    Fine-tuned EfficientNet-B4 for dermoscopic multi-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet-pretrained weights.
        freeze_backbone: If True, freeze all layers except the classifier.
        dropout_p: Dropout probability before the final layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.efficientnet_b4(weights=weights)
        in_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor):
        logits = self.model(x)
        return logits, None, None
