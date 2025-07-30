from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from tinytrain.utils.register import TTRegistry

if TYPE_CHECKING:
    import torch
    from torch import nn


class ExportServerCore:
    """
    导出服务器核心门面，负责把 PyTorch 模型（.pt/.pth）转换成指定后端格式。

    支持链式调用：
        core = ExportServerCore(...)
        core("output_dir")          # 等价于 core.export("output_dir")

    内部通过 TTRegistry 根据 task + backend 自动匹配对应的导出服务实现。
    """
    def __init__(self,
                 config_manager,
                 model: nn.Module,
                 device:torch.device,
                 backend: str = "onnx",
                 **kwargs
                 ):
        """
        初始化导出服务器。

        Args:
            config_manager:
                全局配置管理器，用于读取 task 名称等。
            model:
                已加载权重的 PyTorch 模型（nn.Module）。
            device:
                模型所在的计算设备（cpu / cuda）。
            backend:
                目标后端名称，例如 "onnx"、"tensorrt"、"torchscript" 等。
            **kwargs:
                透传给对应导出服务的额外参数，例如 opset_version、fp16、
                input_names、output_names 等。
        """
        self.config_manager = config_manager
        self.backend = backend

        self.export_server = self._server_select(model, device, **kwargs)

    def __call__(self, export_dir: str | Path = None):
        """
        允许实例像函数一样被直接调用。

        Args:
            export_dir:
                导出目录，若为 None 则使用配置文件中的 save_dir。

        Returns:
            None
        """
        self.export(export_dir)

    def export(self, export_dir: str | Path = None):
        """
        执行导出流程。

        Args:
            export_dir:
                导出目录，若为 None 则使用配置文件中的 save_dir。

        Returns:
            None
        """
        self.export_server.export(export_dir)

    def _server_select(self, model, device, **kwargs):
        """
        根据当前任务与后端名称，从 TTRegistry 构造对应的导出服务实例。

        Args:
            model:
                已加载权重的 PyTorch 模型（nn.Module）。
            device:
                模型所在的计算设备（cpu / cuda）。
            **kwargs:
                透传给导出服务的额外参数。

        Returns:
            BaseExportServer 子类实例
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "export_server", self.backend)(model=model, device=device, **kwargs)
