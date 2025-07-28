from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch
    from torch import nn


class BaseExportServer:
    def __init__(self, model: nn.Module, device: torch.device):
        from torch import nn

        assert isinstance(model, nn.Module)
        self.model = model
        self.device = device

    def __call__(self, export_dir: str | Path = None):
        self.export(export_dir)

    def export(self, export_dir: str | Path = None) -> None:
        raise NotImplementedError
