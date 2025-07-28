from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.register import TTRegistry

if TYPE_CHECKING:
    import torch


class InferenceServerCore:
    """
    解析来自各个推理引擎的模型格式，不包含pt或pth
    """

    def __init__(self,
                 config_manager: ConfigManager,
                 model_file: str | Path,
                 device: torch.device,
                 backend: str = "onnx",
                 **kwargs
                 ):
        self.config_manager = config_manager
        self.model_file = model_file
        self.backend = backend
        self.device = device

        self.inference_server = self._server_select(**kwargs)

    def __call__(self, data):
        return self.inference(data)

    def inference(self, data):
        return self.inference_server.inference(data)

    def _server_select(self, **kwargs):
        """
        构造对应的推理服务
        @return:
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "inference_server", self.backend)(model_file=self.model_file, device=self.device, **kwargs)
