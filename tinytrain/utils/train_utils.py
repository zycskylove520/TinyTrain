import math
import torch

from copy import deepcopy

from tinytrain.utils import LOGGER


class ModelEMA:
    """
    指数移动平均（EMA）封装，用于模型权重平滑。

    主要功能
    --------
    1. 维护一份与原始模型同结构的 FP32 EMA 权重副本。
    2. 支持 DDP 场景（world_size > 1）。
    3. 使用指数升温策略 `decay = decay * (1 - exp(-updates / tau))` 缓解训练初期抖动。
    4. 自动缓存 decay 值，避免重复计算。

    用法示例
    --------
    >>> ema = ModelEMA(model, world_size=2, decay=0.9999, tau=2000)
    >>> for batch in dataloader:
    ...     loss = train_step(batch)
    ...     ema.update(model)
    """

    def __init__(self, model, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """
        初始化 EMA 实例。

        Args:
            model (torch.nn.Module): 原始训练模型。
            decay (float): 目标衰减系数；训练初期会按升温公式动态调整。
            tau (int): 升温时间常数，越小升温越快。
            updates (int): 已完成的更新次数，用于继续训练时恢复状态。
        """
        self.ema_model = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        self.decay_cache = {}  # cache decay values to avoid repeated computation

        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """
        执行一次 EMA 更新。

        Args:
            model (torch.nn.Module): 当前训练模型。
        """
        if not self.enabled:
            return

        self.updates += 1
        d = self.decay(self.updates)

        # Cache decay value to avoid repeated computation
        if self.updates not in self.decay_cache:
            self.decay_cache[self.updates] = d

        msd = model.state_dict()  # model state_dict
        ema_sd = self.ema_model.state_dict()  # EMA state_dict
        for k, v in ema_sd.items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= self.decay_cache[self.updates]
                v += (1 - self.decay_cache[self.updates]) * msd[k].detach()


class EarlyStopping:
    """
    早停控制器，监控验证指标并在连续若干 epoch 无提升时终止训练。

    主要功能
    --------
    1. 支持最大化 / 最小化两种模式。
    2. 支持动态提示“下一次可能停止”，便于日志打印。
    3. 与 TinyTrain Trainer 无缝集成（直接返回 stop 布尔值）。

    用法示例
    --------
    >>> es = EarlyStopping(patience=10, mode='max')
    >>> for epoch in range(100):
    ...     val_fitness = validate()
    ...     if es(epoch, val_fitness):
    ...         break
    """

    def __init__(self, patience: int = 50, mode: str = 'max'):
        """
        初始化早停器。

        Args:
            patience (int): 容忍无提升的最大 epoch 数；0 或 None 表示禁用。
            mode (str): 'max' 或 'min'，决定“更好”的定义。
        """
        self.best_fitness = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.mode = mode

    def __call__(self, epoch: int, fitness: float):
        """
        判断是否需要停止训练。

        Args:
            epoch (int): 当前 epoch。
            fitness (float | None): 当前 epoch 的验证指标；None 表示跳过。

        Returns:
            bool: True 表示应停止训练，False 继续。
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if self.mode == 'max':
            if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
                self.best_epoch = epoch
                self.best_fitness = fitness
        elif self.mode == 'min':
            if fitness <= self.best_fitness:
                self.best_epoch = epoch
                self.best_fitness = fitness

        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"EarlyStopping: Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. use `patience=0` to disable EarlyStopping."
            )
        return stop
