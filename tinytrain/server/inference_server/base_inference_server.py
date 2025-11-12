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

if TYPE_CHECKING:
    import torch


class TTBaseInferenceServer:
    """
    推理服务器基类，定义所有后端实现必须遵循的统一接口。

    子类仅需要实现/重写 `inference` 方法即可完成具体推理逻辑。
    """
    def __init__(self, model_file: str, device: torch.device):
        """
        初始化推理服务器基类。

        Args
        ----
        model_file : str
            已导出的模型文件路径（例如 *.onnx、*.engine）。
        device : torch.device
            运行推理的设备（cpu / cuda）。
        """
        self.model_file = model_file
        self.device = device

    def __call__(self, data):
        """
        使实例可直接被调用，内部转发至 `inference`。

        Args
        ----
        data : torch.Tensor
            输入张量，形状由具体任务决定，如 (B,C,H,W)。

        Returns
        -------
        list[torch.Tensor]
            推理结果列表，每个元素对应一个输出节点。
        """
        return self.inference(data)

    def inference(self, data: torch.Tensor) -> list[torch.Tensor]:
        """
        执行一次前向推理，由子类实现。

        Args
        ----
        data : torch.Tensor
            输入张量。

        Returns
        -------
        list[torch.Tensor]
            包含所有输出张量的列表。
        """
        return [data]
