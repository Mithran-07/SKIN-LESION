"""
Test forward-pass output shapes and checkpoint compatibility for all model architectures.

Verifies that:
1. All models accept (B, 3, 224, 224) input without error
2. Output logits have shape (B, num_classes)
3. DualBranchNet supports both v1_matrix (10,669,639 params) and v2_scalar (9,031,241 params)
4. Trained Dual-Branch checkpoints load with strict=True via DualBranchNet.from_checkpoint
5. MTLDualBranchNet returns both logits and segmentation mask
"""

from pathlib import Path
import pytest
import torch

from models import DualBranchNet, MTLDualBranchNet
from models.baselines import ResNet50Baseline, DenseNet201Baseline, EfficientNetBaseline
from models.shallow_wide_branch import ShallowWideBranch
from models.deep_narrow_branch import DeepNarrowBranch
from models.fusion import AttentionFusionHead

NUM_CLASSES = 7
BATCH_SIZE = 2
H, W = 224, 224
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture
def dummy_input():
    return torch.randn(BATCH_SIZE, 3, H, W)


def test_shallow_wide_branch_shape(dummy_input):
    model = ShallowWideBranch(channels=[256, 512, 1024], pretrained_init=False)
    model.eval()
    with torch.no_grad():
        vec, fmap = model(dummy_input)
    assert vec.shape == (BATCH_SIZE, 1024), f"Expected (B, 1024), got {vec.shape}"
    assert fmap.shape[0] == BATCH_SIZE
    assert fmap.shape[1] == 1024


def test_deep_narrow_branch_shape(dummy_input):
    model = DeepNarrowBranch(pretrained_init=False)
    model.eval()
    with torch.no_grad():
        vec, fmap = model(dummy_input)
    assert vec.shape == (BATCH_SIZE, 256), f"Expected (B, 256), got {vec.shape}"
    assert fmap.shape[0] == BATCH_SIZE
    assert fmap.shape[1] == 256


def test_attention_fusion_v1_and_v2_shapes():
    texture = torch.randn(BATCH_SIZE, 1024)
    structure = torch.randn(BATCH_SIZE, 256)

    head_v1 = AttentionFusionHead(fusion_version="v1_matrix")
    fused_v1 = head_v1(texture, structure)
    assert fused_v1.shape == (BATCH_SIZE, 256)

    head_v2 = AttentionFusionHead(fusion_version="v2_scalar")
    fused_v2 = head_v2(texture, structure)
    assert fused_v2.shape == (BATCH_SIZE, 256)


def test_dual_branch_net_v1_and_v2_param_counts(dummy_input):
    model_v1 = DualBranchNet(
        num_classes=NUM_CLASSES, pretrained_init=False, fusion_version="v1_matrix"
    )
    assert model_v1.get_num_params() == 10669639
    model_v1.eval()
    with torch.no_grad():
        logits, texture_fmap, structure_fmap = model_v1(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert texture_fmap.shape[0] == BATCH_SIZE
    assert structure_fmap.shape[0] == BATCH_SIZE

    model_v2 = DualBranchNet(
        num_classes=NUM_CLASSES, pretrained_init=False, fusion_version="v2_scalar"
    )
    assert model_v2.get_num_params() == 9031241
    model_v2.eval()
    with torch.no_grad():
        logits2, _, _ = model_v2(dummy_input)
    assert logits2.shape == (BATCH_SIZE, NUM_CLASSES)


@pytest.mark.parametrize(
    "fusion_version,expected_params",
    [
        ("v1_matrix", 10669639),
        ("v2_scalar", 9031241),
    ],
)
def test_dual_branch_strict_checkpoint_loading(tmp_path, fusion_version, expected_params):
    orig_model = DualBranchNet(
        num_classes=NUM_CLASSES, pretrained_init=False, fusion_version=fusion_version
    )
    ckpt_path = tmp_path / f"dual_branch_{fusion_version}.pth"
    torch.save(
        {
            "epoch": 25,
            "model_name": "dual_branch",
            "state_dict": orig_model.state_dict(),
        },
        ckpt_path,
    )
    loaded_model = DualBranchNet.from_checkpoint(
        str(ckpt_path), fusion_version=fusion_version, device="cpu"
    )
    assert loaded_model.fusion_version == fusion_version
    assert loaded_model.get_num_params() == expected_params

    wrong_version = "v2_scalar" if fusion_version == "v1_matrix" else "v1_matrix"
    with pytest.raises(ValueError, match="Architecture version mismatch"):
        DualBranchNet.from_checkpoint(
            str(ckpt_path), fusion_version=wrong_version, device="cpu"
        )


def test_mtl_dual_branch_net_shape(dummy_input):
    model = MTLDualBranchNet(num_classes=NUM_CLASSES, pretrained_init=False)
    model.eval()
    with torch.no_grad():
        logits, seg_mask, texture_fmap, structure_fmap = model(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert seg_mask.shape == (BATCH_SIZE, 1, H, W), f"Seg mask shape: {seg_mask.shape}"


def test_resnet50_baseline_shape(dummy_input):
    model = ResNet50Baseline(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_densenet201_baseline_shape(dummy_input):
    model = DenseNet201Baseline(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_efficientnet_baseline_shape(dummy_input):
    model = EfficientNetBaseline(num_classes=NUM_CLASSES, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
