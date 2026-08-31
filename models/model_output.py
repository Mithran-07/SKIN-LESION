"""Typed outputs for dual-branch model forwards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Named output container for the dual-branch classifier.

    The class remains tuple-compatible so legacy callers can still unpack
    it as ``logits, texture_fmap, structure_fmap``.
    """

    logits: torch.Tensor
    texture_fmap: torch.Tensor
    structure_fmap: torch.Tensor
    seg_mask: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.seg_mask is not None:
            yield self.logits
            yield self.seg_mask
            yield self.texture_fmap
            yield self.structure_fmap
            return
        yield self.logits
        yield self.texture_fmap
        yield self.structure_fmap

    def __len__(self) -> int:
        return 4 if self.seg_mask is not None else 3

    def __getitem__(self, index: int) -> torch.Tensor:
        values = (self.logits, self.texture_fmap, self.structure_fmap)
        if self.seg_mask is not None:
            values = (self.logits, self.seg_mask, self.texture_fmap, self.structure_fmap)
        return values[index]

    def as_tuple(self):
        if self.seg_mask is not None:
            return self.logits, self.seg_mask, self.texture_fmap, self.structure_fmap
        return self.logits, self.texture_fmap, self.structure_fmap