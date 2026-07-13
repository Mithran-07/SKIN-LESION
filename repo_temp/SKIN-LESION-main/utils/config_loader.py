"""
utils/config_loader.py
======================
YAML configuration loader with dot-access, validation, and path resolution.

The loader returns an ``OmegaConf`` DictConfig so that callers can use both
dictionary-style (``cfg["lr"]``) and attribute-style (``cfg.lr``) access, and
can use ``OmegaConf.to_container`` for plain-dict conversion.

Usage
-----
>>> from utils.config_loader import load_config
>>> cfg = load_config("config/config.yaml")
>>> print(cfg.training.lr)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin DotDict wrapper (no external dependency)
# ---------------------------------------------------------------------------

class DotDict(dict):
    """A dictionary subclass that supports dot-notation attribute access.

    Examples
    --------
    >>> d = DotDict({"a": {"b": 1}})
    >>> d.a.b
    1
    """

    def __getattr__(self, key: str) -> Any:  # noqa: D401
        """Return value by attribute access."""
        try:
            value = self[key]
            return DotDict(value) if isinstance(value, dict) else value
        except KeyError:
            raise AttributeError(
                f"'DotDict' object has no attribute '{key}'"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __repr__(self) -> str:  # noqa: D401
        return f"DotDict({dict.__repr__(self)})"

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain Python dict."""
        result: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = DotDict(value).to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, DotDict) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path]) -> DotDict:
    """Load and parse a YAML configuration file.

    The function:
        1. Resolves the path relative to the caller's working directory.
        2. Parses the YAML file.
        3. Validates that the returned object is a mapping (not a list).
        4. Wraps the result in a :class:`DotDict` for dot-notation access.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the YAML configuration file.

    Returns
    -------
    DotDict
        Parsed configuration with dot-access support.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If the YAML file is empty or its top-level structure is not a mapping.
    yaml.YAMLError
        If the file contains invalid YAML syntax.

    Examples
    --------
    >>> cfg = load_config("config/config.yaml")
    >>> cfg.training.batch_size
    32
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Please check the path or create the file from the template."
        )

    logger.debug("Loading configuration from: %s", path)

    with path.open("r", encoding="utf-8") as fh:
        raw: Optional[Any] = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(raw, dict):
        raise ValueError(
            f"Top-level YAML structure must be a mapping, "
            f"got {type(raw).__name__} in: {path}"
        )

    cfg = DotDict(raw)
    logger.info("Configuration loaded from: %s", path)
    _log_config_summary(cfg)
    return cfg


def print_config(cfg: DotDict, indent: int = 0) -> None:
    """Recursively pretty-print a :class:`DotDict` configuration.

    Parameters
    ----------
    cfg : DotDict
        The configuration to print.
    indent : int
        Current indentation level (used for recursion).

    Examples
    --------
    >>> cfg = load_config("config/config.yaml")
    >>> print_config(cfg)
    """
    prefix = "  " * indent
    for key, value in cfg.items():
        if isinstance(value, (dict, DotDict)):
            print(f"{prefix}{key}:")
            print_config(DotDict(value) if isinstance(value, dict) else value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{key}: {value}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log_config_summary(cfg: DotDict) -> None:
    """Log a brief summary of key config values at DEBUG level."""
    try:
        logger.debug(
            "Config summary — project: %s | seed: %s | epochs: %s | "
            "batch_size: %s | lr: %s | device_force: %s",
            cfg.get("project", {}).get("name", "N/A"),
            cfg.get("seed", "N/A"),
            cfg.get("training", {}).get("epochs", "N/A"),
            cfg.get("training", {}).get("batch_size", "N/A"),
            cfg.get("optimizer", {}).get("lr", "N/A"),
            cfg.get("device", {}).get("force", "auto"),
        )
    except Exception:
        # Never let logging crash the training pipeline
        pass
