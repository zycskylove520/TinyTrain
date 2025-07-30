from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Union

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from torch import nn
    from tinytrain.server.export_server import ExportServerCore


class BaseExporter:
    """
    BaseExporter 是一个通用模型导出基类，用于将 PyTorch 模型转换为部署格式
    （如 ONNX、TensorRT、TorchScript、CoreML、OpenVINO 等）。

    特性：
    - 支持本地 torch.nn.Module 导出。
    - 内置 ExportServerCore 统一封装导出逻辑，支持多种后端。
    - 提供完整的生命周期钩子（on_export_start / on_export_end）。
    - 输出目录可自定义，自动创建多级目录。

    使用方式：
    1. 创建实例：BaseExporter(cfg, model, callback, backend="onnx")
    2. 调用 export：exporter.export(Path("runs/export"))
    """
    def __init__(self,
                 config_manager: ConfigManager,
                 model: nn.Module,
                 callback: Callback,
                 backend: str | None = None,
                 **kwargs
                 ):
        """
        初始化导出器。

        Args:
            config_manager (ConfigManager): 全局配置管理器，包含 device、输入尺寸等信息。
            model (nn.Module): 已加载权重的 PyTorch 模型。
            callback (Callback): 回调对象，用于在导出前后插入自定义逻辑。
            backend (str | None, optional): 导出后端名称，如 "onnx"、"tensorrt"、"torchscript" 等。
            **kwargs: 透传给 ExportServerCore 的额外参数，如 opset_version、dynamic_axes 等。
        """
        self.config_manager = config_manager
        self.backend = backend

        # select device
        from tinytrain.utils.checks import check_device_mini
        self.device = check_device_mini(self.config_manager.core["device"])

        # 根据模型类型初始化导出服务
        self.export_server = self._setup_export_server(model=model, **kwargs)

        # callback
        self.callback = callback

    def export(self, export_dir: str | Path):
        """
        执行模型导出流程，将模型转换为目标格式并保存到指定目录。

        Args:
            export_dir (str | Path): 导出目录路径，不存在将自动创建。

        Raises:
            Exception: 导出过程中发生的任何错误。
        """
        import torch

        self.callback.run_callback(self, "on_export_start")
        with torch.inference_mode():
            self.export_server.export(export_dir)
        self.callback.run_callback(self, "on_export_end")

    def _setup_export_server(self, model: nn.Module, **kwargs) -> Union[nn.Module, ExportServerCore]:
        """
        根据模型类型初始化导出服务器（目前仅支持 PyTorch nn.Module）。

        Args:
            model (nn.Module): 待导出的 PyTorch 模型。
            **kwargs: 透传给 ExportServerCore 的额外参数。

        Returns:
            ExportServerCore: 已配置的导出服务器实例。

        Raises:
            TypeError: 如果模型不是 nn.Module。
        """
        from tinytrain.server.export_server import ExportServerCore
        from torch import nn

        if isinstance(model, nn.Module):
            model = model.to(self.device)
            model.eval()
            return ExportServerCore(config_manager=self.config_manager, model=model, backend=self.backend, device=self.device, **kwargs)
        else:
            raise TypeError(f"only supported pytorch model!")
