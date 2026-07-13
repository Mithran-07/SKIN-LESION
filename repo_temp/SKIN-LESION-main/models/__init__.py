from .dual_branch_net import DualBranchNet
from .mtl_head import MTLDualBranchNet
from .shallow_wide_branch import ShallowWideBranch
from .deep_narrow_branch import DeepNarrowBranch
from .fusion import AttentionFusionHead

__all__ = [
    "DualBranchNet",
    "MTLDualBranchNet",
    "ShallowWideBranch",
    "DeepNarrowBranch",
    "AttentionFusionHead",
]
