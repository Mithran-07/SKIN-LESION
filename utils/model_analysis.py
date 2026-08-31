"""Model analysis utilities for parameter, FLOP, VRAM, and layer summaries."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sys
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class ParameterStats:
    total: int
    trainable: int
    frozen: int


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def _infer_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_dummy_input(input_size: Tuple[int, int, int], batch_size: int = 1) -> torch.Tensor:
    return torch.zeros((batch_size, *input_size), dtype=torch.float32)


def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int]) -> int:
    """Estimate FLOPs via forward hooks on common conv/linear layers."""
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal flops
        out = output
        batch = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        kernel_h, kernel_w = module.kernel_size
        groups = module.groups
        in_channels = module.in_channels
        out_channels = module.out_channels
        ops_per_position = (in_channels // groups) * kernel_h * kernel_w
        flops += batch * out_channels * out_h * out_w * ops_per_position * 2

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal flops
        batch = output.shape[0]
        flops += batch * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    device = _infer_device(model)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = _make_dummy_input(input_size).to(device)
        output = model(dummy)
        if isinstance(output, tuple):
            _ = output[0]
        elif hasattr(output, "logits"):
            _ = output.logits

    for hook in hooks:
        hook.remove()
    model.train(was_training)
    return int(flops)


def estimate_vram(model: nn.Module, batch_size: int, input_size: Tuple[int, int, int]) -> float:
    """Estimate VRAM footprint in MB using parameters + activation tensors."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    activation_bytes = batch_size * int(torch.tensor(input_size).prod().item()) * 4
    total_mb = (param_bytes + buffer_bytes + activation_bytes) / (1024**2)
    return float(total_mb)


def layer_summary(model: nn.Module, input_size: Tuple[int, int, int]):
    """Return a pandas DataFrame with layer output shapes and parameter counts."""
    import pandas as pd

    rows: List[Dict[str, object]] = []
    hooks = []

    def hook_fn(name: str):
        def _hook(module, inputs, output):
            params = sum(p.numel() for p in module.parameters(recurse=False))
            rows.append(
                {
                    "layer": name,
                    "type": module.__class__.__name__,
                    "output_shape": tuple(output.shape) if hasattr(output, "shape") else type(output).__name__,
                    "params": params,
                }
            )

        return _hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    device = _infer_device(model)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(_make_dummy_input(input_size).to(device))

    for hook in hooks:
        hook.remove()
    model.train(was_training)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from models import DualBranchNet

    model = DualBranchNet(pretrained_init=False)
    stats = count_parameters(model)
    print(stats)
    print(f"FLOPs: {estimate_flops(model, (3, 224, 224)):,}")
    print(f"VRAM MB: {estimate_vram(model, 1, (3, 224, 224)):.1f}")
    print(layer_summary(model, (3, 224, 224)).head())