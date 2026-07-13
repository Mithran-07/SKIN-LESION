"""
Core Training Loop — Phase 3
Handles: AMP, AdamW, CosineAnnealingLR, gradient clipping,
         early stopping, checkpointing, TensorBoard, auto-resume.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_config import configure_gpu_performance, get_device, get_amp_settings
from checkpoint_manager import CheckpointManager
from training.metrics import MetricTracker, format_metrics


def get_logger(name: str, log_file: Path) -> logging.Logger:
    """Set up a logger that writes to file and stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


class EarlyStopping:
    """Stop training if monitored metric doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (self.mode == "max" and value > self.best + self.min_delta) or
            (self.mode == "min" and value < self.best - self.min_delta)
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─────────────────────────────────────────────────────────
class BaselineTrainer:
    """
    Trains a baseline CNN model with:
      - AMP (bfloat16 on Ampere GPU)
      - AdamW + CosineAnnealingLR
      - Gradient clipping
      - Early stopping
      - Checkpoint save/resume
      - TensorBoard + file logging
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_classes:  int = 7,
        epochs:       int = 50,
        lr:           float = 1e-4,
        weight_decay: float = 1e-2,
        grad_clip:    float = 1.0,
        early_stop_patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        checkpoints_dir: Path = Path("checkpoints"),
        results_dir:     Path = Path("results"),
        tensorboard_dir: Path = Path("tensorboard"),
        logs_dir:        Path = Path("logs"),
    ):
        self.model_name   = model_name
        self.num_classes  = num_classes
        self.epochs       = epochs
        self.grad_clip    = grad_clip

        # ── Device & AMP
        configure_gpu_performance()
        self.device = get_device()
        amp_enabled, amp_dtype = get_amp_settings()
        self.use_amp  = amp_enabled
        self.amp_dtype = amp_dtype

        # ── Model
        self.model = model.to(self.device)

        # ── Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # ── Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # ── Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        # ── AMP Scaler (only needed for float16; bfloat16 doesn't need it)
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = torch.amp.GradScaler(device="cuda")

        # ── Data
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # ── Early stopping
        self.early_stop = EarlyStopping(
            patience=early_stop_patience, mode="max"
        )

        # ── Directories
        self.model_ckpt_dir = checkpoints_dir / model_name
        self.model_res_dir  = results_dir / model_name
        self.tb_dir         = tensorboard_dir / model_name
        self.model_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_res_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # ── Infrastructure
        self.ckpt_manager = CheckpointManager(str(self.model_ckpt_dir))
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.logger = get_logger(
            f"trainer_{model_name}",
            logs_dir / f"{model_name}_training.log"
        )

        # ── History
        self.history = {k: [] for k in [
            "train_loss", "val_loss",
            "train_acc",  "val_acc",
            "train_f1",   "val_f1",
            "train_bal_acc", "val_bal_acc",
            "train_roc_auc", "val_roc_auc",
            "lr",
        ]}
        self.start_epoch = 0

    # ─────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker(self.num_classes)

        loop = tqdm(self.train_loader, desc=f"[{self.model_name}] Epoch {epoch+1} TRAIN",
                    leave=False, ncols=120)

        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    logits = self.model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss   = self.criterion(logits, labels)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
            else:
                logits = self.model(images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss   = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            tracker.update(logits.detach(), labels.detach(), loss.item())

            # Detect NaN
            if torch.isnan(loss):
                self.logger.error(f"NaN loss at epoch {epoch+1}, batch {batch_idx}!")

            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        return tracker.compute()

    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        tracker = MetricTracker(self.num_classes)

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"[{self.model_name}] Epoch {epoch+1} VAL  ",
                        leave=False, ncols=120)
            for images, labels in loop:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = self.model(images)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss   = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss   = self.criterion(logits, labels)

                tracker.update(logits, labels, loss.item())
                loop.set_postfix({"loss": f"{loss.item():.4f}"})

        return tracker.compute()

    # ─────────────────────────────────────────────────────
    def _try_resume(self):
        """Try to resume from latest checkpoint."""
        latest = self.model_ckpt_dir / "latest_checkpoint.pth"
        if latest.exists():
            ckpt = self.ckpt_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.scheduler
            )
            if ckpt:
                self.start_epoch = ckpt.get("epoch", 0) + 1
                self.history     = ckpt.get("history", self.history)
                self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def _log_to_tensorboard(self, epoch: int, train_m: dict, val_m: dict):
        keys = ["loss", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "pr_auc"]
        for key in keys:
            if key in train_m:
                self.writer.add_scalar(f"Train/{key}", train_m[key], epoch)
            if key in val_m:
                self.writer.add_scalar(f"Val/{key}", val_m[key], epoch)
        self.writer.add_scalar("LR/lr", self.optimizer.param_groups[0]["lr"], epoch)

        # GPU memory
        if torch.cuda.is_available():
            self.writer.add_scalar(
                "GPU/allocated_MB",
                torch.cuda.memory_allocated() / 1e6, epoch
            )
            self.writer.add_scalar(
                "GPU/reserved_MB",
                torch.cuda.memory_reserved() / 1e6, epoch
            )

    # ─────────────────────────────────────────────────────
    def train(self) -> dict:
        """Run the full training loop. Returns best_val_metrics."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"  Starting: {self.model_name}")
        self.logger.info(f"  Device: {self.device} | AMP: {self.use_amp} ({self.amp_dtype})")
        self.logger.info(f"  Epochs: {self.epochs} | Train batches: {len(self.train_loader)}")
        self.logger.info(f"{'='*60}")

        self._try_resume()

        best_val_metrics = {}
        best_bal_acc     = -1.0
        training_start   = time.perf_counter()

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.perf_counter()

            # ── Train + Val
            train_m = self._train_epoch(epoch)
            val_m   = self._val_epoch(epoch)

            # ── Scheduler step
            self.scheduler.step()

            epoch_time = time.perf_counter() - epoch_start

            # ── Update history
            self.history["train_loss"].append(train_m["loss"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["train_acc"].append(train_m["accuracy"])
            self.history["val_acc"].append(val_m["accuracy"])
            self.history["train_f1"].append(train_m["f1_macro"])
            self.history["val_f1"].append(val_m["f1_macro"])
            self.history["train_bal_acc"].append(train_m["balanced_accuracy"])
            self.history["val_bal_acc"].append(val_m["balanced_accuracy"])
            self.history["train_roc_auc"].append(train_m.get("roc_auc", float("nan")))
            self.history["val_roc_auc"].append(val_m.get("roc_auc", float("nan")))
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # ── TensorBoard
            self._log_to_tensorboard(epoch, train_m, val_m)

            # ── Log to console/file
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"LR={lr_now:.2e} | "
                f"Train [{format_metrics(train_m)}] | "
                f"Val [{format_metrics(val_m)}] | "
                f"Time={epoch_time:.1f}s"
            )

            # ── Checkpoint
            is_best = val_m["balanced_accuracy"] > best_bal_acc
            if is_best:
                best_bal_acc = val_m["balanced_accuracy"]
                best_val_metrics = val_m

            self.ckpt_manager.save_checkpoint(
                state={
                    "epoch":       epoch,
                    "state_dict":  self.model.state_dict(),
                    "optimizer":   self.optimizer.state_dict(),
                    "scheduler":   self.scheduler.state_dict(),
                    "history":     self.history,
                    "best_val_bal_acc": best_bal_acc,
                    "model_name":  self.model_name,
                },
                is_best=is_best,
                filename="latest_checkpoint.pth",
            )

            # ── Early stopping
            if self.early_stop.step(val_m["balanced_accuracy"]):
                self.logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(patience={self.early_stop.patience})"
                )
                break

        total_training_time = time.perf_counter() - training_start
        self.logger.info(f"\nTraining complete! Total time: {total_training_time:.1f}s")
        self.writer.close()

        return best_val_metrics, self.history, total_training_time

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Run inference on the test set using the best checkpoint.
        Also measures inference time and peak VRAM.
        """
        # Load best checkpoint
        best_ckpt = self.model_ckpt_dir / "best_checkpoint.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.logger.info("Loaded best checkpoint for test evaluation")

        self.model.eval()
        tracker = MetricTracker(self.num_classes)

        # Reset VRAM stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        n_images     = 0
        inference_start = time.perf_counter()

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"[{self.model_name}] TEST", ncols=100):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = self.model(images)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        loss   = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss   = self.criterion(logits, labels)

                tracker.update(logits, labels, loss.item())
                n_images += images.size(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - inference_start
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0
        )
        per_image_ms = inference_time / max(n_images, 1) * 1000

        test_metrics = tracker.compute()
        test_metrics["inference_time_s"]    = inference_time
        test_metrics["inference_per_img_ms"] = per_image_ms
        test_metrics["peak_vram_mb"]        = peak_vram_mb
        test_metrics["n_images"]            = n_images

        self.logger.info(
            f"\n[TEST ] {self.model_name}\n"
            f"        {format_metrics(test_metrics)}\n"
            f"        Inference: {per_image_ms:.2f} ms/img | "
            f"Peak VRAM: {peak_vram_mb:.0f} MB"
        )

        return test_metrics
