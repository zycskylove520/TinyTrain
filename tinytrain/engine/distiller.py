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

from typing import TYPE_CHECKING
from torch import nn

from tinytrain.cfg import TTConfigManager
from tinytrain.utils.callback import Callback

from .trainer import TTBaseTrainer

if TYPE_CHECKING:
    from .model import TTBaseModel


class TTBaseDistiller(TTBaseTrainer):
    """
    通用知识蒸馏训练器基类。

    功能：
    1. 复用 TTBaseTrainer 的全部训练流程（DDP、AMP、EMA、断点续训等）。
    2. 额外引入「教师模型」提供软标签，学生模型通过蒸馏损失进行学习。
    3. 默认仅冻结教师模型参数；子类可重写蒸馏损失计算逻辑。

    使用：
        distiller = TTBaseDistiller(cfg, device, student_model, teacher_model, callback)
        distiller.train()
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: TTConfigManager, device: torch.device, student_model: TTBaseModel, teacher_model: TTBaseModel, callback: Callback):
        """
        初始化蒸馏器。

        Args:
            config_manager: 全局配置管理器。
            device: 当前进程设备。
            student_model: 学生模型（可训练）。
            teacher_model: 教师模型（仅推理）。
            callback: 训练生命周期钩子。
        """
        super().__init__(config_manager=config_manager, device=device, model=student_model, callback=callback)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    # ------------------------------------------------------------------
    # 2. 蒸馏逻辑（子类可重写）
    # ------------------------------------------------------------------
    def model_inference_and_loss_calculate(self, teacher_model, student_model, inputs) -> tuple[float, dict]:
        """
        前向传播 + 蒸馏损失计算模板。

        Args:
            teacher_model: 教师模型（已冻结）。
            student_model: 学生模型（可训练）。
            inputs: 输入张量或列表。

        Returns:
            tuple:
                - total_loss: 标量，用于反向传播。
                - loss_items: 字典，记录各分项损失（仅用于日志）。
        """

        # 计算教师模型的输出
        teacher_class_scores, teacher_bbox_coords = teacher_model(inputs)
        # 计算学生模型的输出
        student_class_scores, student_bbox_coords = student_model(inputs)

        # 计算知识蒸馏的损失
        loss, loss_items = self.sample_distillation_loss(student_class_scores, student_bbox_coords, teacher_class_scores, teacher_bbox_coords, temperature=5.0, alpha=0.5)
        return loss, loss_items

    def sample_distillation_loss(self, student_class_scores, student_bbox_coords, teacher_class_scores, teacher_bbox_coords, temperature, alpha):
        """
        默认蒸馏损失：KL 散度 + L1 框回归。

        Args:
            student_class_scores: 学生类别 logits。
            student_bbox_coords: 学生框预测。
            teacher_class_scores: 教师类别 logits。
            teacher_bbox_coords: 教师框预测。
            temperature: 蒸馏温度。
            alpha: KL 损失权重 (0~1)，1-alpha 给框回归。

        Returns:
            tuple:
                - total_loss: 标量。
                - loss_items: {"kl_loss": tensor, "bbox_loss": tensor}。
        """

        # 计算类别置信度的KL散度损失
        student_softmax = nn.functional.softmax(student_class_scores / temperature, dim=1)
        teacher_softmax = nn.functional.softmax(teacher_class_scores / temperature, dim=1)
        kl_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_class_scores / temperature, dim=1),
                                 teacher_softmax)

        # 计算边界框的L1损失
        bbox_loss = nn.L1Loss()(student_bbox_coords, teacher_bbox_coords)

        # 总损失是KL散度损失和边界框损失的加权和
        total_loss = alpha * kl_loss + (1 - alpha) * bbox_loss

        loss_items = {"kl_loss": kl_loss.deatch(), "bbox_loss": bbox_loss.deatch()}
        return total_loss, loss_items

    # ------------------------------------------------------------------
    # 3. 训练流程钩子（不建议重写）
    # ------------------------------------------------------------------
    def freeze_layers(self, model: TTBaseModel):
        """
        蒸馏场景下仅冻结教师模型参数（学生模型照常训练）。

        Args:
            model: 此处实际传入的是学生模型（TTBaseTrainer.model）。
        """

        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def execute_forward(self, batch_samples):
        """
        训练器每 batch 会调用的前向入口。
        内部转发给蒸馏专用前向函数。

        Args:
            batch_samples: 当前批次数据（含输入与标签）。

        Returns:
            tuple[float, dict]: (总损失, 分项损失字典)。
        """
        return self.model_inference_and_loss_calculate(self.teacher_model, self.model, batch_samples.data)
