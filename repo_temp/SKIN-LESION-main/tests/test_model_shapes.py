"""
Test forward-pass output shapes for all model architectures.

Verifies that:
1. All models accept (B, 3, 224, 224) input without error
2. Output logits have shape (B, num_classes)
3. DualBranchNet returns feature maps for Grad-CAM
4. MTLDualBranchNet returns both logits and segmentation mask
"""

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


def test_attention_fusion_shape():
    model = AttentionFusionHead()
    texture = torch.randn(BATCH_SIZE, 1024)
    structure = torch.randn(BATCH_SIZE, 256)
    fused = model(texture, structure)
    assert fused.shape == (BATCH_SIZE, 256), f"Expected (B, 256), got {fused.shape}"


def test_dual_branch_net_shape(dummy_input):
    model = DualBranchNet(num_classes=NUM_CLASSES, pretrained_init=False)
    model.eval()
    with torch.no_grad():
        logits, texture_fmap, structure_fmap = model(dummy_input)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert texture_fmap.shape[0] == BATCH_SIZE
    assert structure_fmap.shape[0] == BATCH_SIZE


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
