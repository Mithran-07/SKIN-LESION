"""
Evaluation script: generates full benchmark comparison across all models.

Usage:
    python scripts/evaluate.py --checkpoint results/best_model.pt --model dual_branch
    python scripts/evaluate.py --all-checkpoints results/  # Evaluate all .pt files
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import create_dataloaders
from models import DualBranchNet
from models.baselines import ResNet50Baseline, DenseNet201Baseline, EfficientNetBaseline
from models.dual_branch_net import build_model_from_config
from training.metrics import MetricTracker
from training.trainer import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device, class_names) -> dict:
    """Run inference on the test set and compute all metrics."""
    model.eval()
    tracker = MetricTracker(num_classes=len(class_names), class_names=class_names)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
            import torch.nn.functional as F
            loss = F.cross_entropy(logits, labels)
            tracker.update(logits, labels, loss.item())

    metrics = tracker.compute()
    print(tracker.classification_report())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ADL Evaluation Script")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="dual_branch")
    parser.add_argument("--output", type=str, default="results/eval/benchmark.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg.get("device", "auto"))
    class_names = cfg["dataset"]["class_names"]

    _, _, test_loader = create_dataloaders(cfg)

    # Build model and load checkpoint
    model_builders = {
        "dual_branch": lambda: build_model_from_config(cfg),
        "resnet50": lambda: ResNet50Baseline(num_classes=len(class_names)),
        "densenet201": lambda: DenseNet201Baseline(num_classes=len(class_names)),
        "efficientnet": lambda: EfficientNetBaseline(num_classes=len(class_names)),
    }
    model = model_builders[args.model]()
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    metrics = evaluate_model(model, test_loader, device, class_names)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.insert(0, "model", args.model)
    df.insert(1, "checkpoint", args.checkpoint)
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Print key metrics
    print("\n=== BENCHMARK RESULTS ===")
    key = ["auc_macro", "f1_macro", "balanced_accuracy",
           "recall_BCC", "recall_AKIEC", "recall_MEL"]
    for k in key:
        if k in metrics:
            print(f"  {k:30s}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
