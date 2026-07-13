"""
Learning rate scheduling with linear warmup + cosine annealing.

Warmup phase prevents large early gradient updates from corrupting
pretrained branch stem weights during the first few epochs.
Cosine decay provides smooth LR reduction without hard resets.
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Creates a cosine annealing schedule with linear warmup.

    Schedule:
        [0, warmup_epochs):  LR linearly increases from 0 → base_lr
        [warmup_epochs, T):  LR follows cosine annealing from base_lr → min_lr

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        min_lr_ratio: Minimum LR as fraction of base_lr (default 0.0).

    Returns:
        LambdaLR scheduler instance.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
