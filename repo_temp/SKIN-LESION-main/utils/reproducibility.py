"""
utils/reproducibility.py
========================
Reproducibility utilities that guarantee deterministic behaviour across runs
on CUDA, MPS, and CPU backends.

Usage
-----
>>> from utils.reproducibility import set_seed
>>> set_seed(42)
"""

from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(
    seed: int = 42,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Set random seeds globally for full reproducibility.

    Seeds the following sources:
        - Python built-in ``random``
        - ``os.environ["PYTHONHASHSEED"]``
        - ``numpy.random``
        - ``torch`` (CPU and CUDA)
        - cuDNN determinism flags

    Parameters
    ----------
    seed : int
        The integer seed value. Defaults to ``42``.
    deterministic : bool
        If ``True``, sets ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False``.  This may reduce GPU
        throughput slightly but guarantees reproducibility.  Defaults to
        ``True``.
    warn_only : bool
        Passed to ``torch.use_deterministic_algorithms``.  When ``True``,
        PyTorch will emit a warning instead of raising an error when a
        non-deterministic operation is invoked.  Defaults to ``False``.

    Notes
    -----
    Full determinism on GPU is not always achievable due to hardware-level
    parallelism.  Setting ``deterministic=True`` covers the majority of cases.

    Examples
    --------
    >>> set_seed(42, deterministic=True)
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. Environment variable used by Python's hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # 5. cuDNN determinism flags
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except AttributeError:
            # Older PyTorch versions may not support this call
            logger.warning(
                "torch.use_deterministic_algorithms is not available in this "
                "PyTorch version (%s). Skipping.",
                torch.__version__,
            )
    else:
        # Allow cuDNN auto-tuner — faster but non-deterministic
        torch.backends.cudnn.benchmark = True

    logger.info(
        "Reproducibility configured — seed=%d, deterministic=%s",
        seed,
        deterministic,
    )
