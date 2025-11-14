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

import torch
import torch.distributed as dist

from typing import TYPE_CHECKING, Any, Dict
from torch import nn

from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.utils.progress_bar import TTProgressBar
from tinytrain.cfg import TTConfigManager
from .. import WORLD_SIZE
from ..utils.callback import Events

if TYPE_CHECKING:
    from .trainer import TTBaseTrainer


class TTBaseValidator:
    """
    TTBaseValidator 是验证器的抽象基类，负责在训练过程中或训练结束后对模型进行验证。
    它提供了完整的验证流程骨架，包括数据预处理、推理、后处理以及指标计算与汇总。
    子类只需按需实现/重写特定方法即可快速适配不同任务（如分类、检测、分割等）。

    验证时机：
    1. 训练阶段：每个 epoch 结束后立即验证（用于早停、调度器、日志记录）。
    2. 训练完成：所有 epoch 结束后最终验证（用于报告最终性能）。

    主要流程：
        start_metrics_*   -> 初始化指标
        update_metrics_*  -> 逐 batch 更新指标
        end_metrics_*     -> 汇总并输出指标
        get_fitness       -> 返回一个标量，用于衡量模型好坏（越大越好）

    注意：
    - 所有耗时运算应放在 @torch.inference_mode() 下执行，以关闭梯度计算。
    - 支持 DDP；若需跨进程聚合指标，请在子类中自行实现。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, trainer: TTBaseTrainer):
        """
        初始化验证器。

        Args:
            trainer (TTBaseTrainer): 训练器实例，用于获取配置、模型、数据加载器等。
        """

        self.trainer: TTBaseTrainer = trainer
        self.config_manager: TTConfigManager = trainer.config_manager
        self.save_dir = trainer.save_dir

        # device
        self.device: torch.device = trainer.device

        # dataloader
        self.val_dataloader = trainer.val_dataloader

        # callback
        self.callbacks = trainer.callbacks

    def __call__(self, *args, **kwargs):
        """
        使验证器实例可调用，等价于 self.validate(*args, **kwargs)。
        """
        self.validate(*args, **kwargs)

    # ------------------------------------------------------------------
    # 2. 主验证流程（唯一公开主链）
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def validate(self):
        """
        执行验证流程。

        Returns:
            float: fitness 值，越大表示模型越好。
        """
        # 用于判断训练何时结束
        stop = self.trainer.stop

        # 获取模型
        model = self.get_model_instance()
        model.eval()

        self.callbacks.run_callback(Events.ON_VAL_START, self)

        pbar = TTProgressBar(self.val_dataloader, total=len(self.val_dataloader), info_color="blue")

        if stop:
            self.start_metrics_on_train_completed(pbar)
        else:
            self.start_metrics_on_training(pbar)

        for i, batch_samples in enumerate(pbar):
            self.callbacks.run_callback(Events.ON_VAL_BATCH_START, self)

            # Preprocess
            batch_samples = self.preprocess(batch_samples)  # type: ignore[arg-type]

            # Inference
            preds = self.inference(model, batch_samples)

            # Postprocess
            outputs = self.postprocess(preds)

            if stop:
                self.update_metrics_on_train_completed(outputs, batch_samples, pbar)
            else:
                self.update_metrics_on_training(outputs, batch_samples, pbar)

            self.callbacks.run_callback(Events.ON_VAL_BATCH_END, self)

        if stop:
            self.end_metrics_on_train_completed(pbar)
        else:
            self.end_metrics_on_training(pbar)

        fitness = self.get_fitness()
        self.callbacks.run_callback(Events.ON_VAL_END, self)

        if WORLD_SIZE > 1:
            dist.barrier()
        return fitness

    def inference(self, model: nn.Module, batch_samples: BaseBatchDataInfo) -> list[torch.Tensor]:
        return model.inference(batch_samples.data)

    # ------------------------------------------------------------------
    # 3. 数据流水（子类需重写）
    # ------------------------------------------------------------------
    def preprocess(self, batch_samples: BaseBatchDataInfo) -> BaseBatchDataInfo:
        """
        对输入 batch 进行预处理，如将数据移到目标设备。

        Args:
            batch_samples (BaseBatchDataInfo): 原始 batch 数据。

        Returns:
            BaseBatchDataInfo: 预处理后的 batch 数据。
        """
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> Any:
        """
        对模型原始输出进行后处理，如 NMS、阈值过滤、softmax 等。

        Args:
            preds (list[torch.Tensor]): 模型输出张量列表。

        Returns:
            Any: 后处理后的结果，将提交给 update_metrics_on_train_completed 函数或 update_metrics_on_training 函数。
        """
        return preds

    # ------------------------------------------------------------------
    # 4. 指标生命周期：训练过程中验证（子类需重写）
    # ------------------------------------------------------------------
    def start_metrics_on_training(self, pbar: TTProgressBar):
        """
        训练阶段验证开始前，初始化或重置指标。

        Args:
            pbar (TTProgressBar): 进度条实例，可用于动态展示信息。
        """
        pass

    def update_metrics_on_training(self, outputs: Any, batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        """
        训练阶段验证过程中，每处理一个 batch 即更新指标。

        Args:
            outputs (Any): 接收来自 postprocess函数 的输出。
            batch_samples (BaseBatchDataInfo): 当前 batch 的输入与标签。
            pbar (TTProgressBar): 进度条实例。
        """
        pass

    def end_metrics_on_training(self, pbar: TTProgressBar):
        """
        训练阶段验证结束时，汇总指标并输出（例如打印、写日志、更新 TensorBoard）。

        Args:
            pbar (TTProgressBar): 进度条实例。
        """
        pass

    def get_fitness(self) -> float:
        """
        返回一个标量，用于衡量当前模型在验证集上的优劣。
        训练器会把该值用于 checkpoint 选择、早停、学习率调度等。

        Returns:
            float: 越大表示模型越好；默认返回 0（子类必须重写以提供有效指标）。
        """
        return 0.

    # ------------------------------------------------------------------
    # 5. 指标生命周期：训练完成后最终验证（子类需重写）
    # ------------------------------------------------------------------
    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        """
        训练全部完成后验证开始前，初始化或重置指标。

        Args:
            pbar (TTProgressBar): 进度条实例。
        """
        pass

    def update_metrics_on_train_completed(self, outputs: Any, batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        """
        训练完成后验证过程中，每处理一个 batch 即更新指标。

        Args:
            outputs (Any):  接收来自 postprocess函数 的输出。
            batch_samples (BaseBatchDataInfo): 当前 batch 的输入与标签。
            pbar (TTProgressBar): 进度条实例。
        """
        pass

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        """
        训练完成后验证结束时，汇总指标并输出。

        Args:
            pbar (TTProgressBar): 进度条实例。
        """
        pass

    # ------------------------------------------------------------------
    # 6. 分布式辅助 & 内部工具
    # ------------------------------------------------------------------
    def get_model_instance(self):
        if self.trainer.ema:
            model = self.trainer.ema.ema_model
        else:
            model = self.trainer.model.module if WORLD_SIZE > 1 else self.trainer.model
        return model

    @classmethod
    def all_reduce_tensor(cls, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """把 tensor 在所有 rank 上做 all_reduce（原地）"""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=op)

    @classmethod
    def all_reduce_mean(cls, tensor: torch.Tensor):
        """把 tensor 在所有 rank 上做 all_reduce 并求平均"""
        cls.all_reduce_tensor(tensor)
        if dist.is_available() and dist.is_initialized():
            tensor /= dist.get_world_size()
