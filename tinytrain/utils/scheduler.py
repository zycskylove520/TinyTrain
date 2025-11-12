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

import math


class LinearWarmupLR:
    """
    Linear-Warmup 学习率调度器
    --------------------------
    在前 `warmup_epochs` 个 epoch 内，将学习率从 `warmup_lr` **线性**提升到 `lr0`；
    当 epoch ≥ warmup_epochs 时，保持 `lr0` 不变。

    公式
    ----
    lr(epoch) = warmup_lr + (lr0 - warmup_lr) * min(epoch / warmup_epochs, 1.0)

    用法示例
    --------
    >>> scheduler = LinearWarmupLR(optimizer, warmup_lr=1e-4, lr0=1e-2, warmup_epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """

    def __init__(
            self,
            optimizer,
            warmup_lr: float,
            lr0: float,
            warmup_epochs: int,
            last_epoch: int = -1,
    ):
        """
        初始化调度器并立即将学习率设为 warmup_lr。

        Args:
            optimizer: torch.optim.Optimizer 实例。
            warmup_lr (float): 起始（warmup）学习率。
            lr0 (float): 目标学习率（warmup 结束值）。
            warmup_epochs (int): 线性 warmup 持续 epoch 数。
            last_epoch (int): 用于恢复训练时的 epoch 偏移量，默认为 -1。
        """
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.warmup_epochs = warmup_epochs
        self.last_epoch = last_epoch
        self.step()

    def step(self, epoch = None) -> float:
        """
        更新学习率并记录 epoch 索引。

        Args:
            epoch (int, optional): 如果传入，则使用该 epoch 索引计算 lr；
                否则使用内部计数器 `self.last_epoch`。

        Returns:
            float: 当前学习率。
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        progress = min(epoch / self.warmup_epochs, 1.0)
        lr = self.warmup_lr + (self.lr0 - self.warmup_lr) * progress

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


class ConstantWarmupLR:
    """
    Constant-Warmup 学习率调度器
    --------------------------------
    在整个 warmup_epochs 内保持固定学习率 warmup_lr；
    当 epoch ≥ warmup_epochs 时，一次性跳转到 lr0（不再变化）。

    主要功能
    --------
    1. 支持手动指定 epoch 或内部计数器两种更新方式。
    2. 兼容 torch.optim.Optimizer 的所有 param_groups。

    用法示例
    --------
    >>> scheduler = ConstantWarmupLR(optimizer, warmup_lr=1e-4, lr0=1e-2, warmup_epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_lr, lr0, warmup_epochs):
        """
        初始化调度器并立即将学习率设为 warmup_lr。

        Args:
            optimizer: torch.optim.Optimizer 实例。
            warmup_lr (float): Warmup 阶段的学习率。
            lr0 (float): Warmup 结束后固定的学习率。
            warmup_epochs (int): Warmup 持续 epoch 数。
        """
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.step()

    def step(self, epoch=None):
        """
        更新学习率并记录 epoch 索引。

        Args:
            epoch (int | None): 如果传入，则使用该 epoch 索引计算 lr；
                否则使用内部计数器 `self.epoch`。

        Returns:
            float: 当前学习率。
        """
        if epoch is None:
            epoch = self.epoch
        self.epoch = epoch + 1

        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.lr0

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


class CosineWarmUpLR:
    """
    Cosine-Warmup 学习率调度器
    ----------------------------
    在前 `epochs` 个 epoch 内，将学习率从 `warmup_lr` 平滑地提升到 `lr0`
    （余弦上升曲线），超出后保持 `lr0` 不变。

    公式
    ----
    lr(epoch) = warmup_lr + 0.5 * (lr0 - warmup_lr) * (1 - cos(π * epoch / epochs))

    用法示例
    --------
    >>> scheduler = CosineWarmUpLR(optimizer, warmup_lr=1e-4, lr0=1e-2, epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_lr, lr0, epochs, last_epoch=-1):
        """
        初始化调度器并立即将学习率设为 warmup_lr。

        Args:
            optimizer: torch.optim.Optimizer 实例。
            warmup_lr (float): 起始学习率。
            lr0 (float): 目标学习率（cosine 结束值）。
            epochs (int): cosine warmup 总 epoch 数。
            last_epoch (int): 用于恢复训练时的 epoch 偏移量，默认为 -1。
        """
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.epochs = epochs
        self.last_epoch = last_epoch
        self.step()

    def step(self, epoch=None):
        """
        更新学习率并记录 epoch 索引。

        Args:
            epoch (int | None): 如果传入，则使用该 epoch 索引计算 lr；
                否则使用内部计数器 `self.last_epoch`。

        Returns:
            float: 当前学习率。
        """
        if epoch is None:
            epoch = getattr(self, "last_epoch", -1) + 1
        self.last_epoch = epoch
        progress = min(epoch / self.epochs, 1.0)
        lr = self.warmup_lr + 0.5 * (self.lr0 - self.warmup_lr) * (1 - math.cos(math.pi * progress))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


class ExponentialWarmUpLR:
    """
    Exponential-Warmup 学习率调度器
    -------------------------------
    在前 `epochs` 个 epoch 内，将学习率从 `warmup_lr`
    指数上升到 `lr0`（指数底 γ = (lr0 / warmup_lr)^(1/epochs)），
    超出后保持 `lr0` 不变。

    公式
    ----
    lr(epoch) = warmup_lr * γ^epoch,  γ = (lr0 / warmup_lr)^(1/epochs)

    用法示例
    --------
    >>> scheduler = ExponentialWarmUpLR(optimizer, warmup_lr=1e-4, lr0=1e-2, epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_lr, lr0, epochs, last_epoch=-1):
        """
        初始化调度器并立即将学习率设为 warmup_lr。

        Args:
            optimizer: torch.optim.Optimizer 实例。
            warmup_lr (float): 起始学习率。
            lr0 (float): 目标学习率（指数结束值）。
            epochs (int): 指数 warmup 总 epoch 数。
            last_epoch (int): 用于恢复训练时的 epoch 偏移量，默认为 -1。
        """
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.epochs = epochs
        self.gamma = (lr0 / warmup_lr) ** (1.0 / epochs)
        self.last_epoch = last_epoch
        self.step()

    def step(self, epoch=None):
        """
        更新学习率并记录 epoch 索引。

        Args:
            epoch (int | None): 如果传入，则使用该 epoch 索引计算 lr；
                否则使用内部计数器 `self.last_epoch`。

        Returns:
            float: 当前学习率。
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.warmup_lr * (self.gamma ** min(epoch, self.epochs))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr
