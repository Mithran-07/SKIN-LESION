"""
CLI Training Script for ADL Dual-Branch CNN.

Usage:
    # Train dual-branch model (full pipeline)
    python scripts/train.py --config config/config.yaml --model dual_branch

    # Train a baseline for comparison
    python scripts/train.py --config config/config.yaml --model resnet50

    # Train with MTL head (requires segmentation masks)
    python scripts/train.py --config config/config.yaml --model mtl

    # Dry run (checks pipeline without full training)
    python scripts/train.py --config config/config.yaml --model dual_branch --dry-run
"""

import argparse
import logging
import sys
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import create_dataloaders, HAM10000Dataset
from data.sampler import compute_class_weights_tensor
from models import DualBranchNet, MTLDualBranchNet
from models.baselines import ResNet50Baseline, DenseNet201Baseline, EfficientNetBaseline
from models.dual_branch_net import build_model_from_config
from training import Trainer
from training.trainer import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, cfg: dict, device: torch.device):
    """Instantiate the requested model architecture."""
    num_classes = cfg["dataset"]["num_classes"]
    model_map = {
        "dual_branch": lambda: build_model_from_config(cfg),
        "mtl":         lambda: MTLDualBranchNet(num_classes=num_classes),
        "resnet50":    lambda: ResNet50Baseline(num_classes=num_classes),
        "densenet201": lambda: DenseNet201Baseline(num_classes=num_classes),
        "efficientnet":lambda: EfficientNetBaseline(num_classes=num_classes),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(model_map.keys())}")

    model = model_map[model_name]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model '{model_name}' | Trainable params: {n_params:,}")
    return model


def main():
    parser = argparse.ArgumentParser(description="ADL Dual-Branch CNN — Training Script")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument(
        "--model", type=str, default="dual_branch",
        choices=["dual_branch", "mtl", "resnet50", "densenet201", "efficientnet"],
        help="Model architecture to train",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one batch through the pipeline to check for errors")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Seed ─────────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])

    # ── Device ───────────────────────────────────────────────────────────────
    device = get_device(cfg.get("device", "auto"))
    logger.info(f"Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)

    # ── Class weights for Focal Loss ─────────────────────────────────────────
    alpha_weights = None
    if cfg["loss"]["focal"].get("alpha") == "auto":
        alpha_weights = compute_class_weights_tensor(
            train_loader.dataset.get_labels(),
            num_classes=cfg["dataset"]["num_classes"],
            device=device,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, cfg, device)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Resumed from checkpoint: {args.resume}")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=== DRY RUN ===")
        model = model.to(device)
        model.eval()
        batch = next(iter(train_loader))
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            output = model(images)
        logits = output[0] if isinstance(output, tuple) else output
        logger.info(f"Input shape : {images.shape}")
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"Labels      : {labels[:8].tolist()}")
        logger.info(f"Preds       : {logits.argmax(dim=1)[:8].tolist()}")
        logger.info("✅ Dry run passed — pipeline is working correctly.")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        alpha_weights=alpha_weights,
        device=device,
    )
    result = trainer.train()
    logger.info(f"Training complete. Best Val AUC: {result['best_val_auc']:.4f}")
    logger.info(f"Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
