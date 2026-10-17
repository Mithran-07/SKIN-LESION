"""
Attention-Gated Feature Fusion Module with Explicit Architecture Versioning.

Synthesizes the texture vector (1024-dim from the shallow-wide branch) and
the structure vector (256-dim from the deep-narrow branch) into a unified
representation via a learned sigmoid gate.

Supported Fusion Versions:
- "v1_matrix": Joint element-wise matrix gate `Linear(1280 -> 1280) + Sigmoid`
  (2,426,624 fusion params; 10,669,639 total DualBranchNet params).
  Used by historical checkpoints `dual_branch_seed123` and `dual_branch_seed999`.
- "v2_scalar": Decoupled branch-level scalar gates `gate_t: Linear(1024 -> 1)`
  and `gate_s: Linear(256 -> 1)` (788,226 fusion params; 9,031,241 total
  DualBranchNet params). Used by checkpoint `dual_branch_seed42`.
"""

from typing import Literal, Tuple
import torch
import torch.nn as nn

FusionVersion = Literal["v1_matrix", "v2_scalar"]
SUPPORTED_FUSION_VERSIONS: Tuple[str, ...] = ("v1_matrix", "v2_scalar")


class AttentionFusionHead(nn.Module):
    """
    Versioned Attention-Gated Fusion Head for Dual-Branch CNN.

    Args:
        texture_dim: Output dim of the shallow-wide branch (default 1024).
        structure_dim: Output dim of the deep-narrow branch (default 256).
        hidden_dim: Intermediate MLP dimension (default 512).
        output_dim: Final fused vector dimension (default 256).
        dropout_1: Dropout after first FC layer (default 0.4).
        dropout_2: Dropout after second FC layer (default 0.3).
        fusion_version: Explicit fusion gating architecture ("v1_matrix" or "v2_scalar").
    """

    def __init__(
        self,
        texture_dim: int = 1024,
        structure_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout_1: float = 0.4,
        dropout_2: float = 0.3,
        fusion_version: FusionVersion = "v2_scalar",
    ):
        super().__init__()
        if fusion_version not in SUPPORTED_FUSION_VERSIONS:
            raise ValueError(
                f"Unsupported fusion_version '{fusion_version}'. "
                f"Expected one of {SUPPORTED_FUSION_VERSIONS}."
            )

        self.texture_dim = texture_dim
        self.structure_dim = structure_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_version: FusionVersion = fusion_version
        concat_dim = texture_dim + structure_dim  # 1280

        if fusion_version == "v1_matrix":
            # V1: Full 1280x1280 element-wise sigmoid attention gate
            self.gate = nn.Sequential(
                nn.Linear(concat_dim, concat_dim),
                nn.Sigmoid(),
            )
        else:
            # V2: Independent scalar attention gates per branch
            self.gate_t = nn.Sequential(
                nn.Linear(texture_dim, 1),
                nn.Sigmoid(),
            )
            self.gate_s = nn.Sequential(
                nn.Linear(structure_dim, 1),
                nn.Sigmoid(),
            )

        # Shared MLP projection structure
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_2),
        )

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
        if self.fusion_version == "v1_matrix":
            combined = torch.cat([texture_vec, structure_vec], dim=1)  # (B, 1280)
            gate_weights = self.gate(combined)                          # (B, 1280)
            gated = combined * gate_weights                             # (B, 1280)
            return self.mlp(gated)                                      # (B, 256)

        g_t = self.gate_t(texture_vec)                                  # (B, 1)
        g_s = self.gate_s(structure_vec)                                # (B, 1)
        tex_scaled = texture_vec * g_t                                  # (B, 1024)
        str_scaled = structure_vec * g_s                                # (B, 256)
        combined = torch.cat([tex_scaled, str_scaled], dim=1)           # (B, 1280)
        return self.mlp(combined)                                       # (B, 256)
