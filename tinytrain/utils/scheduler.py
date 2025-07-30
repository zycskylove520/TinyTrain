import math


class ConstantWarmupLR:
    """
   Constant-Warmup 学习率调度器
   --------------------------------
   在整个 warmup_epochs 内保持固定学习率 warmup_lr；
   当 epoch ≥ warmup_epochs 时，一次性跳转到 lr0（不再变化）。

   用法示例
   --------
   >>> scheduler = ConstantWarmupLR(optimizer, warmup_lr=0.0001, lr0=0.01, warmup_epochs=10)
   >>> for epoch in range(100):
   ...     train_one_epoch(...)
   ...     scheduler.step()
   """
    def __init__(self, optimizer, warmup_lr, lr0, warmup_epochs):
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.step()

    def step(self, epoch=None):
        """
        更新学习率并记录 epoch 索引。

        参数
        ----
        epoch : int, optional
            如果传入，则使用该 epoch 索引计算 lr；
            否则使用内部计数器 `self.epoch`。
        返回
        ----
        float
            当前学习率。
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
    >>> scheduler = CosineWarmUpLR(optimizer, warmup_lr=0.0001, lr0=0.01, epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """
    def __init__(self, optimizer, warmup_lr, lr0, epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.lr0 = lr0
        self.epochs = epochs
        self.last_epoch = last_epoch
        self.step()

    def step(self, epoch=None):
        """
        更新学习率并记录 epoch 索引。

        参数
        ----
        epoch : int, optional
            如果传入，则使用该 epoch 索引计算 lr；
            否则使用内部计数器 `self.last_epoch`。
        返回
        ----
        float
            当前学习率。
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
    >>> scheduler = ExponentialWarmUpLR(optimizer, warmup_lr=0.0001, lr0=0.01, epochs=10)
    >>> for epoch in range(100):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """
    def __init__(self, optimizer, warmup_lr, lr0, epochs, last_epoch=-1):
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

        参数
        ----
        epoch : int, optional
            如果传入，则使用该 epoch 索引计算 lr；
            否则使用内部计数器 `self.last_epoch`。
        返回
        ----
        float
            当前学习率。
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.warmup_lr * (self.gamma ** min(epoch, self.epochs))
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr