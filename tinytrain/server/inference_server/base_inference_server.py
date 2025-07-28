from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class BaseInferenceServer:
    def __init__(self, model_file: str, device: torch.device):
        self.model_file = model_file
        self.device = device

    def __call__(self, data):
        return self.inference(data)

    def inference(self, data: torch.Tensor) -> list[torch.Tensor]:
        return [data]
