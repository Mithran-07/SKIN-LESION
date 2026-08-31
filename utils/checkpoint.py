"""
utils/checkpoint.py
===================
Checkpoint save and load helpers for model persistence.

Implementation is deferred to Phase 3 (training loop).  This module
defines the public interface so that other modules can import from it
without breaking during Phases 1–2.

Usage (Phase 3+)
----------------
>>> from utils.checkpoint import save_checkpoint, load_checkpoint
>>> save_checkpoint(state, path="checkpoints/best_model.pth")
>>> state = load_checkpoint("checkpoints/best_model.pth", device)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing: saving, loading, and best-model tracking.

    Tracks the best checkpoint according to a monitored metric and
    optionally keeps only the top-k most recent checkpoints on disk.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory where checkpoint files will be written.
    monitor : str
        Metric name to track for "best" model selection (e.g. ``"val_auc"``).
    mode : str
        ``"max"`` if higher metric is better; ``"min"`` if lower is better.
    save_top_k : int
        Keep only the top-k checkpoints.  Pass ``-1`` to keep all.

    Note
    ----
    Implementation is deferred to **Phase 3**.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_auc",
        mode: str = "max",
        save_top_k: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self._best_value: Optional[float] = None
        self._history: list[tuple[float, Path]] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "CheckpointManager initialised — dir=%s, monitor=%s, mode=%s",
            self.checkpoint_dir,
            self.monitor,
            self.mode,
        )

    def save(self, state: Dict[str, Any], metric_value: float) -> Path:
        """Save a checkpoint and track whether it is the best so far.

        Parameters
        ----------
        state : Dict[str, Any]
            Dictionary containing at minimum ``model_state_dict`` and
            ``optimizer_state_dict``.
        metric_value : float
            Current value of the monitored metric.

        Returns
        -------
        Path
            Path to the saved checkpoint file.

        Note
        ----
        **Not implemented — Phase 3.**
        """
        raise NotImplementedError("CheckpointManager.save() is implemented in Phase 3.")

    def load(
        self,
        path: str | Path,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a checkpoint from disk.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pth`` checkpoint file.
        device : Optional[torch.device]
            Target device for tensor loading.  Defaults to CPU.

        Returns
        -------
        Dict[str, Any]
            The deserialized checkpoint dictionary.

        Note
        ----
        **Not implemented — Phase 3.**
        """
        raise NotImplementedError("CheckpointManager.load() is implemented in Phase 3.")

    @property
    def best_checkpoint(self) -> Optional[Path]:
        """Return the path to the current best checkpoint, or ``None``."""
        if not self._history:
            return None
        key_fn = max if self.mode == "max" else min
        best_val, best_path = key_fn(self._history, key=lambda t: t[0])
        return best_path


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """Convenience function: save a checkpoint dictionary to disk.

    Parameters
    ----------
    state : Dict[str, Any]
        Checkpoint payload (model state, optimizer state, epoch, metrics…).
    path : str | Path
        Target file path.

    Note
    ----
    **Not implemented — Phase 3.**
    """
    raise NotImplementedError("save_checkpoint() is implemented in Phase 3.")


def load_checkpoint(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function: load a checkpoint from disk.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    device : Optional[torch.device]
        Device to map tensors to.  Defaults to CPU.

    Returns
    -------
    Dict[str, Any]
        Deserialized checkpoint dictionary.

    Note
    ----
    **Not implemented — Phase 3.**
    """
    raise NotImplementedError("load_checkpoint() is implemented in Phase 3.")
