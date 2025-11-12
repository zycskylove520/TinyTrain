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

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch
    from torch import nn


class TTBaseExportServer:
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

    def __call__(self, export_dir: Path):
        """
        允许实例像函数一样被直接调用，内部转发到 `export` 方法。

        Args
        ----
        export_dir : Path
            导出目录。
        """
        self.export(export_dir)

    def export(self, export_dir:Path) -> None:
        """
        执行模型导出流程，由子类实现。

        Args
        ----
        export_dir : Path
            导出目录。

        Raises
        ----
        NotImplementedError
            如果子类未实现该方法。
        """
        raise NotImplementedError
