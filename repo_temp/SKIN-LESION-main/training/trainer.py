"""
Trainer class for dual-branch dermoscopic CNN training.

Handles:
- Full training loop with gradient accumulation
- Validation with comprehensive metric reporting
- Early stopping based on macro-AUC plateau
- Checkpoint saving (best model by val AUC)
- Device-agnostic operation (CUDA / MPS / CPU)
- Optional mixed-precision training (CUDA only)
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import box

from training.metrics import MetricTracker
from training.scheduler import get_cosine_schedule_with_warmup
from losses import MTLLoss, FocalLoss

logger = logging.getLogger(__name__)
console = Console()


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    Priority: cuda > mps > cpu (when device_str == 'auto').
    MPS is used automatically on Apple Silicon Macs.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class Trainer:
    """
    End-to-end trainer for DualBranchNet and baseline models.

    Args:
        model: A model instance (DualBranchNet, MTLDualBranchNet, or baseline).
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        cfg: Full config dict from config.yaml.
        alpha_weights: Per-class alpha tensor for FocalLoss. If None, uniform.
        device: Training device (resolved via get_device if not specified).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        alpha_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        train_cfg = cfg["training"]
        opt_cfg = cfg["optimizer"]
        sched_cfg = cfg["scheduler"]
        loss_cfg = cfg["loss"]
        ds_cfg = cfg["dataset"]

        self.device = device or get_device(cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = train_cfg["epochs"]
        self.grad_accum = train_cfg.get("grad_accum_steps", 1)
        self.use_amp = train_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.patience = train_cfg.get("early_stopping_patience", 10)
        self.save_dir = Path(train_cfg.get("save_dir", "results"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = train_cfg.get("log_interval", 10)

        # ── Loss ─────────────────────────────────────────────────────────────
        if alpha_weights is not None:
            alpha_weights = alpha_weights.to(self.device)
        self.criterion = FocalLoss(
            alpha=alpha_weights,
            gamma=loss_cfg["focal"]["gamma"],
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        )

        # ── Optimizer ────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )

        # ── Scheduler ────────────────────────────────────────────────────────
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_epochs=sched_cfg["warmup_epochs"],
            total_epochs=self.epochs,
            min_lr_ratio=sched_cfg.get("min_lr", 1e-6) / opt_cfg["lr"],
        )

        # ── AMP scaler (CUDA only) ────────────────────────────────────────────
        self.scaler = GradScaler() if self.use_amp else None

        # ── Metrics ──────────────────────────────────────────────────────────
        self.class_names = ds_cfg["class_names"]
        self.num_classes = ds_cfg["num_classes"]
        self.train_tracker = MetricTracker(self.num_classes, self.class_names)
        self.val_tracker = MetricTracker(self.num_classes, self.class_names)

        # ── State ─────────────────────────────────────────────────────────────
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.history: Dict[str, list] = {"train": [], "val": []}

        console.print(f"[bold green]Trainer initialized[/bold green]")
        console.print(f"  Device    : [cyan]{self.device}[/cyan]")
        console.print(f"  AMP       : [cyan]{self.use_amp}[/cyan]")
        console.print(f"  Grad Accum: [cyan]{self.grad_accum}[/cyan]")
        console.print(f"  Params    : [cyan]{sum(p.numel() for p in model.parameters() if p.requires_grad):,}[/cyan]")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_tracker.reset()
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1:03d} [Train]",
            leave=False,
        )

        for batch_idx, batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # ── Forward pass ─────────────────────────────────────────────────
            if self.use_amp:
                with autocast():
                    logits, *_ = self.model(images)
                    loss = self.criterion(logits, labels) / self.grad_accum
                self.scaler.scale(loss).backward()
            else:
                logits, *_ = self.model(images)
                loss = self.criterion(logits, labels) / self.grad_accum
                loss.backward()

            # ── Gradient accumulation step ────────────────────────────────────
            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            actual_loss = loss.item() * self.grad_accum
            self.train_tracker.update(logits.detach(), labels, actual_loss)

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss=f"{actual_loss:.4f}")

        return self.train_tracker.compute()

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        self.val_tracker.reset()

        for batch in tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1:03d} [Val]",
            leave=False,
        ):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            logits, *_ = self.model(images)
            loss = self.criterion(logits, labels)
            self.val_tracker.update(logits, labels, loss.item())

        return self.val_tracker.compute()

    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        """Print a rich formatted metrics table."""
        table = Table(
            title=f"Epoch {epoch+1}/{self.epochs}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", style="green")
        table.add_column("Val", style="yellow")

        key_metrics = ["loss", "auc_macro", "f1_macro", "balanced_accuracy"]
        for k in key_metrics:
            if k in train_metrics and k in val_metrics:
                table.add_row(k, f"{train_metrics[k]:.4f}", f"{val_metrics[k]:.4f}")

        # Per-class recall
        for name in self.class_names:
            key = f"recall_{name}"
            if key in val_metrics:
                table.add_row(f"recall_{name}", "-", f"{val_metrics[key]:.4f}")

        console.print(table)
        lr = self.optimizer.param_groups[0]["lr"]
        console.print(f"  LR: [dim]{lr:.2e}[/dim]")

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history and path to best checkpoint.
        """
        console.print(f"\n[bold]Starting training for {self.epochs} epochs[/bold]\n")
        best_checkpoint_path = self.save_dir / "best_model.pt"

        for epoch in range(self.epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)
            self.scheduler.step()

            elapsed = time.time() - t0
            self._log_epoch(epoch, train_metrics, val_metrics)

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            # ── Checkpoint & early stopping ──────────────────────────────────
            val_auc = val_metrics.get("auc_macro", 0.0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": val_auc,
                        "val_metrics": val_metrics,
                        "cfg": self.cfg,
                    },
                    best_checkpoint_path,
                )
                console.print(f"  [bold green]✓ New best model saved (AUC={val_auc:.4f})[/bold green]")
            else:
                self.epochs_without_improvement += 1
                console.print(
                    f"  [yellow]No improvement for {self.epochs_without_improvement}/{self.patience} epochs[/yellow]"
                )

            if self.epochs_without_improvement >= self.patience:
                console.print(
                    f"\n[bold red]Early stopping triggered at epoch {epoch+1}[/bold red]"
                )
                break

            console.print(f"  Epoch time: {elapsed:.1f}s\n")

        console.print(f"\n[bold green]Training complete. Best Val AUC: {self.best_val_auc:.4f}[/bold green]")
        console.print(f"  Best model: {best_checkpoint_path}")

        return {
            "history": self.history,
            "best_val_auc": self.best_val_auc,
            "best_checkpoint": str(best_checkpoint_path),
        }
