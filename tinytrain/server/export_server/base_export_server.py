from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch
    from torch import nn


class BaseExportServer:
    """
    导出服务器的抽象基类，定义了所有后端导出器必须实现的接口。

    子类需要实现 `export` 方法，完成从 PyTorch 模型到指定后端格式的转换。
    """
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化导出服务器基类。

        Args
        ----
        model : torch.nn.Module
            已加载权重的 PyTorch 模型。
        device : torch.device
            模型所在的计算设备（cpu / cuda）。
        """
        from torch import nn

        assert isinstance(model, nn.Module)
        self.model = model
        self.device = device

    def __call__(self, export_dir: str | Path = None):
        """
        允许实例像函数一样被直接调用，内部转发到 `export` 方法。

        Args
        ----
        export_dir : str | Path | None
            导出目录，若为 None 则使用配置文件中的默认路径。
        """
        self.export(export_dir)

    def export(self, export_dir: str | Path = None) -> None:
        """
        执行模型导出流程，由子类实现。

        Args
        ----
        export_dir : str | Path | None
            导出目录，若为 None 则使用配置文件中的默认路径。

        Raises
        ----
        NotImplementedError
            如果子类未实现该方法。
        """
        raise NotImplementedError
