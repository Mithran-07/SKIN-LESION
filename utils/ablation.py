"""Ablation experiment helpers for the dual-branch architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
import torch.nn as nn


@dataclass(frozen=True)
class AblationConfig:
    use_shallow_branch: bool = True
    use_deep_branch: bool = True
    use_attention_gate: bool = True
    use_mtl_decoder: bool = False
    pretrained_init: bool = False

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def build_ablation_model(config: AblationConfig, num_classes: int = 7):
    """Factory returning a dual-branch model configured for a static ablation."""
    if config.use_mtl_decoder:
        from models import MTLDualBranchNet

        return MTLDualBranchNet(num_classes=num_classes, pretrained_init=config.pretrained_init)

    from models.dual_branch_net import DualBranchNet

    model = DualBranchNet(num_classes=num_classes, pretrained_init=config.pretrained_init)

    model.ablation_config = config  # type: ignore[attr-defined]
    # The model remains structurally intact, but a disabled gate is replaced
    # with an identity transform for clean ablation baselines.
    if not config.use_attention_gate:
        model.fusion.gate = nn.Identity()  # type: ignore[assignment]
    return model