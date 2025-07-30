from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.register import TTRegistry

if TYPE_CHECKING:
    import torch


class InferenceServerCore:
    """
    推理服务器核心门面，负责加载非 PyTorch 格式（ONNX/TensorRT/…）的模型文件，
    提供统一的 inference 接口。

    说明
    ----
    - 仅接受已导出的外部模型文件（*.onnx、*.engine、*.plan …），
      不负责 .pt/.pth 的加载。
    - 内部通过 TTRegistry 根据 task + backend 动态选择对应推理服务实现。
    """
    def __init__(self,
                 config_manager: ConfigManager,
                 model_file: str | Path,
                 device: torch.device,
                 backend: str = "onnx",
                 **kwargs
                 ):
        """
        初始化推理服务器。

        Args
        ----
        config_manager : ConfigManager
            全局配置管理器，用于获取当前任务名称（classify / detect / …）。
        model_file : str | Path
            已导出的模型文件路径（如 *.onnx、*.engine）。
        device : torch.device
            运行推理的设备（cpu / cuda）。
        backend : str, default "onnx"
            推理后端名称，决定加载哪个具体推理服务类。
        **kwargs :
            透传给后端服务的构造参数，例如 batch_size、workspace_size、
            input_shapes、fp16 开关等。
        """
        self.config_manager = config_manager
        self.model_file = model_file
        self.backend = backend
        self.device = device

        self.inference_server = self._server_select(**kwargs)

    def __call__(self, data):
        """
        使实例可像函数一样直接调用，例如：
            logits = server_core(batch_tensor)

        Args
        ----
        data :
            待推理的输入数据，通常为 torch.Tensor 或 numpy.ndarray，
            具体格式由后端服务决定。

        Returns
        -------
        Any
            后端服务返回的推理结果，通常为 torch.Tensor 或 list[torch.Tensor]。
        """
        return self.inference(data)

    def inference(self, data):
        """
        执行一次推理。

        Args
        ----
        data :
            待推理的输入数据。

        Returns
        -------
        Any
            后端服务返回的推理结果。
        """
        return self.inference_server.inference(data)

    def _server_select(self, **kwargs):
        """
        根据当前任务与后端名称，从 TTRegistry 构造对应的推理服务实例。

        Args
        ----
        **kwargs :
            透传给推理服务的额外参数。

        Returns
        -------
        BaseInferenceServer
            已初始化的后端推理服务实例。
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "inference_server", self.backend)(model_file=self.model_file, device=self.device, **kwargs)
