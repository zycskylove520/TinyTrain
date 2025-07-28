import math

from copy import deepcopy

import torch

from tinytrain.utils import LOGGER


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models.
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, world_size: int, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        """
        Initialize EMA for 'model' with given arguments.
        """
        self.world_size = world_size
        self.ema_model = deepcopy(model.module if world_size > 1 else model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        self.decay_cache = {}  # cache decay values to avoid repeated computation

        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """
        Update EMA parameters.
        """
        if not self.enabled:
            return

        self.updates += 1
        d = self.decay(self.updates)

        # Cache decay value to avoid repeated computation
        if self.updates not in self.decay_cache:
            self.decay_cache[self.updates] = d

        model = model.module if self.world_size > 1 else model
        msd = model.state_dict()  # model state_dict
        ema_sd = self.ema_model.state_dict()  # EMA state_dict
        for k, v in ema_sd.items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= self.decay_cache[self.updates]
                v += (1 - self.decay_cache[self.updates]) * msd[k].detach()


class EarlyStopping:
    """
    Early stopping class that stops training when a specified number of epochs have passed without improvement.
    """
    def __init__(self, patience: int = 50, mode: str = 'max'):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
            mode (str, optional): 'max' or 'min'. Default: 'max'.
        """
        self.best_fitness = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.mode = mode

    def __call__(self, epoch: int, fitness: float):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            bool: True if training should stop, False otherwise
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
