"""
Checkpoint Manager
Handles saving and loading training checkpoints to resume training sessions safely.
"""

import os
import torch
import shutil

class CheckpointManager:
    """
    Manages saving and loading PyTorch training states.
    Supports keeping 'latest' and 'best' checkpoints to safeguard training progress.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state: dict, is_best: bool = False, filename: str = "latest_checkpoint.pth"):
        """
        Saves the training state. Copies it to 'best_checkpoint.pth' if is_best is True.
        
        Args:
            state (dict): State dictionary containing model weight, optimizer, epoch, etc.
            is_best (bool): If True, also duplicates the file as the best checkpoint.
            filename (str): Name of the checkpoint file.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save state dictionary (using PyTorch standard utility)
        # We save to a temporary file first, then rename, to avoid corrupted checkpoints if training crashes mid-save
        temp_filepath = filepath + ".tmp"
        torch.save(state, temp_filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        print(f"[CHECKPOINT] Saved checkpoint to {filepath}")
        
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            shutil.copyfile(filepath, best_filepath)
            print(f"[CHECKPOINT] Copied as new best checkpoint: {best_filepath}")

    def load_latest_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, 
                               scheduler = None, filename: str = "latest_checkpoint.pth") -> dict:
        """
        Loads the latest checkpoint and updates the model, optimizer, and scheduler.
        
        Args:
            model (torch.nn.Module): The model to load weights into.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
            scheduler (optional): The learning rate scheduler to load state into.
            filename (str): Name of the checkpoint file.
            
        Returns:
            dict: The original state dictionary, or None if no checkpoint is found.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            print(f"[CHECKPOINT] No checkpoint found at {filepath}. Starting from scratch.")
            return None
            
        print(f"[CHECKPOINT] Loading checkpoint from {filepath}...")
        # Map location to CPU first to prevent CUDA out-of-memory or device mismatch
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Load weights and states
        model.load_state_dict(checkpoint["state_dict"])
        
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            
        print(f"[CHECKPOINT] Checkpoint loaded. Resuming from epoch {checkpoint.get('epoch', 0) + 1}")
        return checkpoint

    def get_latest_checkpoint_epoch(self, filename: str = "latest_checkpoint.pth") -> int:
        """
        Queries the epoch of the latest checkpoint without loading weights.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            return 0
        try:
            checkpoint = torch.load(filepath, map_location="cpu")
            return checkpoint.get("epoch", 0)
        except Exception:
            return 0
