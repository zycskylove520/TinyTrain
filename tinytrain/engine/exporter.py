"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from tinytrain.cfg import TTConfigManager, TTEngineRegistry
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import create_iter_directory
from tinytrain.utils.callback import Callback, Events

if TYPE_CHECKING:
    import torch
    from torch import nn
    from tinytrain.server.export_server import TTBaseExportServer


class TTBaseExporter:
    """
    TTBaseExporter 是一个通用模型导出基类，用于将 PyTorch 模型转换为部署格式
    （如 ONNX、TensorRT、TorchScript、CoreML、OpenVINO 等）。

    特性：
    - 支持本地 torch.nn.Module 导出。
    - 内置 TTBaseExportServer 统一封装导出逻辑，支持多种后端。
    - 提供完整的生命周期钩子（on_export_start / on_export_end）。
    - 输出目录可自定义，自动创建多级目录。

    使用方式：
    1. 创建实例：TTBaseExporter(cfg, model, callback, backend="onnx")
    2. 调用 export：exporter.export(Path("runs/export"))
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self,
                 config_manager: TTConfigManager,
                 device: torch.device,
                 model: nn.Module,
                 callback: Callback,
                 backend: str | None = None,
                 **kwargs
                 ):
        """
        初始化导出器。

        Args:
            config_manager (TTConfigManager): 全局配置管理器，包含 device、输入尺寸等信息。
            model (nn.Module): 已加载权重的 PyTorch 模型。
            callback (Callback): 回调对象，用于在导出前后插入自定义逻辑。
            backend (str | None, optional): 导出后端名称，如 "onnx"、"tensorrt"、"torchscript" 等。
            **kwargs: 透传给 TTBaseExportServer 的额外参数，如 opset_version、dynamic_axes 等。
        """
        self.config_manager = config_manager
        self.backend = backend

        # select device
        self.device = device

        # 根据模型类型初始化导出服务
        self.export_server = self._setup_export_server(model=model, **kwargs)

        # callback
        self.callback = callback

        # save dir
        self.output_dir = None

    # ------------------------------------------------------------------
    # 2. 唯一公开主链
    # ------------------------------------------------------------------
    def export(self):
        """
        执行模型导出流程，将模型转换为目标格式并保存到指定目录。

        Raises:
            Exception: 导出过程中发生的任何错误。
        """
        import torch

        core = self.config_manager.core
        save_dir = Path(core["save_dir"]).resolve() / (core["project_name"] or "default_project") / core["task"] / "export"
        self.output_dir = create_iter_directory(save_dir, start_string="export_")

        self.callback.run_callback(Events.ON_EXPORT_START, self)
        with torch.inference_mode():
            self.export_server.export(self.output_dir)
        self.callback.run_callback(Events.ON_EXPORT_END, self)

        LOGGER.info(f"Export result saved in directory -> {self.output_dir}")

    # ------------------------------------------------------------------
    # 3. 内部工具（不建议重写）
    # ------------------------------------------------------------------
    def _setup_export_server(self, model: nn.Module, **kwargs) -> Union[nn.Module, TTBaseExportServer]:
        """
        根据模型类型初始化导出服务器（目前仅支持 PyTorch nn.Module）。

        Args:
            model (nn.Module): 待导出的 PyTorch 模型。
            **kwargs: 透传给 TTBaseExportServer 的额外参数。

        Returns:
            TTBaseExportServer: 已配置的导出服务器实例。

        Raises:
            TypeError: 如果模型不是 nn.Module。
        """
        from torch import nn

        if isinstance(model, nn.Module):
            model = model.to(self.device)
            model.eval()
            return TTEngineRegistry.get(self.config_manager, "export_server", self.backend)(model=model, device=self.device, **kwargs)
        else:
            raise TypeError(f"only supported pytorch model!")
