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

import torch


class TTBaseLoss(torch.nn.Module):
    """
    所有损失函数的基类，定义了统一的损失计算接口。

    所有具体的损失函数都应该继承此类，并实现 forward 方法。
    该类确保所有损失函数返回统一的格式，便于损失监控、日志记录和训练流程管理。

    Attributes:
        无显式属性定义，子类可根据需要添加自己的参数。

    Example:
        >>> base_loss = TTBaseLoss()
        >>> loss, loss_items = base_loss()
        >>> print(f"Total loss: {loss}, Loss items: {loss_items}")
    """

    def __init__(self, *args, **kwargs):
        super(TTBaseLoss, self).__init__()

    def forward(self, *args, **kwargs):
        """
        计算损失的核心方法，必须由子类实现或直接使用。

        Returns:
            tuple: 包含以下元素的元组：
                - loss (torch.Tensor): 标量损失张量，requires_grad=True，用于反向传播。
                - loss_items (dict): 记录各分量损失的字典，所有值都应是detach副本，用于监控和日志记录。

        Note:
            loss_items 格式要求：
            1. 必须返回字典，键为损失名称，值为对应的损失张量或标量
            2. 所有损失值必须使用 .detach() 从计算图中分离，避免内存泄漏
            3. 值必须是标量张量或Python标量，便于日志记录和监控
            4. 建议的字典格式示例：
               - {"cls_loss": torch.tensor(0.), "mse_loss": 0, ...}
        """
        loss = torch.tensor(0., dtype=torch.float, requires_grad=True)
        loss_items = {"my_loss": loss.detach()}
        return loss, loss_items
