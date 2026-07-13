"""
Attention-Gated Feature Fusion Module.

Synthesizes the texture vector (1024-dim from the shallow-wide branch) and
the structure vector (256-dim from the deep-narrow branch) into a unified
representation via a learned sigmoid gate.

The gate dynamically emphasizes texture features for lesion types where
textural cues dominate (SCC keratin texture, BCC arborizing vessels) and
prioritizes structure for morphologically distinctive lesions (asymmetric MEL).
"""

import torch
import torch.nn as nn


class AttentionFusionHead(nn.Module):
    """
    Attention-Gated Fusion: concat → sigmoid gate → MLP projection.

    Architecture:
        [texture(1024) ‖ structure(256)] → 1280-dim
        Sigmoid Gate: Linear(1280→1280) → Sigmoid  →  element-wise product
        FC1: Linear(1280→512) → GELU → Dropout(0.4)
        FC2: Linear(512→256) → GELU → Dropout(0.3)
        Output: 256-dim fused representation

    Args:
        texture_dim: Output dim of the shallow-wide branch (default 1024).
        structure_dim: Output dim of the deep-narrow branch (default 256).
        hidden_dim: Intermediate MLP dimension (default 512).
        output_dim: Final fused vector dimension (default 256).
        dropout_1: Dropout after first FC layer.
        dropout_2: Dropout after second FC layer.
    """

    def __init__(
        self,
        texture_dim: int = 1024,
        structure_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_1: float = 0.4,
        dropout_2: float = 0.3,
    ):
        super().__init__()
        concat_dim = texture_dim + structure_dim  # 1280

        # Independent scalar attention gates
        self.gate_t = nn.Sequential(
            nn.Linear(texture_dim, 1),
            nn.Sigmoid(),
        )
        self.gate_s = nn.Sequential(
            nn.Linear(structure_dim, 1),
            nn.Sigmoid(),
        )

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_2),
        )

        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, texture_vec: torch.Tensor, structure_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            texture_vec: (B, texture_dim) from shallow-wide branch
            structure_vec: (B, structure_dim) from deep-narrow branch

        Returns:
            fused: (B, output_dim) — gated, projected feature vector
        """
        g_t = self.gate_t(texture_vec)                              # (B, 1)
        g_s = self.gate_s(structure_vec)                            # (B, 1)
        tex_scaled = texture_vec * g_t                              # (B, 1024)
        str_scaled = structure_vec * g_s                            # (B, 256)
        combined = torch.cat([tex_scaled, str_scaled], dim=1)       # (B, 1280)
        fused = self.mlp(combined)                                  # (B, 256)
        return fused
