from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from TinyTrain.utils.register import TTRegistry

if TYPE_CHECKING:
    import torch
    from torch import nn


class ExportServerCore:
    """
    导出pt或pth模型到各个推理引擎的模型格式
    """

    def __init__(self,
                 config_manager,
                 model: nn.Module,
                 device:torch.device,
                 backend: str = "onnx",
                 **kwargs
                 ):
        self.config_manager = config_manager
        self.backend = backend

        self.export_server = self._server_select(model, device, **kwargs)

    def __call__(self, export_dir: str | Path = None):
        self.export(export_dir)

    def export(self, export_dir: str | Path = None):
        self.export_server.export(export_dir)

    def _server_select(self, model, device, **kwargs):
        """
        构造对应的导出服务
        @return:
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "export_server", self.backend)(model=model, device=device, **kwargs)
