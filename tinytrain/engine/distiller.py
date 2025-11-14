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
    # 2. 前向传播+损失计算（要求子类必须重写）
    # ------------------------------------------------------------------
    def execute_forward(self, batch_samples):
        """
        训练器每 batch 会调用的前向入口。

        Args:
            batch_samples: 当前批次数据（含输入与标签）。

        Returns:
            tuple:
                - total_loss: 标量，用于反向传播。
                - loss_items: 字典，记录各分项损失（仅用于日志）。

        Note:
            loss_items 格式要求：
            1. 必须返回字典，键为损失名称，值为对应的损失张量
            2. 所有损失值必须使用 .detach() 从计算图中分离
            3. 值必须是标量张量或标量
            4. 例如：
               - {"cls_loss": torch.tensor(0.), "mse_loss": 0, ...}

        Examples:
            >>> # 示例1：基础分类蒸馏
            >>> def execute_forward(self, batch_samples):
            >>>     inputs, labels = batch_samples
            >>>
            >>>     # 学生模型前向
            >>>     student_outputs = self.model(inputs)
            >>>
            >>>     # 教师模型前向（不计算梯度）
            >>>     with torch.no_grad():
            >>>         teacher_outputs = self.teacher_model(inputs)
            >>>
            >>>     # 计算硬标签损失（学生vs真实标签）
            >>>     cls_loss = F.cross_entropy(student_outputs.logits, labels)
            >>>
            >>>     # 计算蒸馏损失（学生vs教师软标签）
            >>>     distill_loss = F.kl_div(
            >>>         F.log_softmax(student_outputs.logits / self.temperature, dim=-1),
            >>>         F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
            >>>         reduction='batchmean'
            >>>     ) * (self.temperature ** 2)
            >>>
            >>>     # 总损失 = α * 蒸馏损失 + (1-α) * 分类损失
            >>>     total_loss = self.alpha * distill_loss + (1 - self.alpha) * cls_loss
            >>>
            >>>     # 记录损失项（必须使用detach()）
            >>>     loss_items = {
            >>>         "cls_loss": cls_loss.detach(),
            >>>         "distill_loss": distill_loss.detach(),
            >>>         "total_loss": total_loss.detach()
            >>>     }
            >>>
            >>>     return total_loss, loss_items

            >>> # 示例2：多任务蒸馏（分类+回归）
            >>> def execute_forward(self, batch_samples):
            >>>     inputs, labels = batch_samples
            >>>     cls_labels, reg_labels = labels
            >>>
            >>>     # 前向传播
            >>>     student_outputs = self.model(inputs)
            >>>
            >>>     with torch.no_grad():
            >>>         teacher_outputs = self.teacher_model(inputs)
            >>>
            >>>     # 分类损失
            >>>     cls_loss = F.cross_entropy(student_outputs.cls_logits, cls_labels)
            >>>
            >>>     # 回归损失
            >>>     reg_loss = F.mse_loss(student_outputs.reg_pred, reg_labels)
            >>>
            >>>     # 分类蒸馏损失
            >>>     cls_distill = F.kl_div(
            >>>         F.log_softmax(student_outputs.cls_logits / 2.0, dim=-1),
            >>>         F.softmax(teacher_outputs.cls_logits / 2.0, dim=-1),
            >>>         reduction='batchmean'
            >>>     ) * 4.0
            >>>
            >>>     # 回归蒸馏损失
            >>>     reg_distill = F.mse_loss(student_outputs.reg_pred, teacher_outputs.reg_pred)
            >>>
            >>>     # 总损失
            >>>     total_loss = (cls_loss + reg_loss + 0.5 * cls_distill + 0.5 * reg_distill)
            >>>
            >>>     loss_items = {
            >>>         "cls_loss": cls_loss.detach(),
            >>>         "reg_loss": reg_loss.detach(),
            >>>         "cls_distill": cls_distill.detach(),
            >>>         "reg_distill": reg_distill.detach(),
            >>>         "total_loss": total_loss.detach()
            >>>     }
            >>>
            >>>     return total_loss, loss_items

            >>> # 示例3：仅蒸馏（无真实标签）
            >>> def execute_forward(self, batch_samples):
            >>>     inputs = batch_samples  # 无标签数据
            >>>
            >>>     student_outputs = self.model(inputs)
            >>>
            >>>     with torch.no_grad():
            >>>         teacher_outputs = self.teacher_model(inputs)
            >>>
            >>>     # 仅使用蒸馏损失
            >>>     distill_loss = F.mse_loss(student_outputs.features, teacher_outputs.features)
            >>>
            >>>     loss_items = {
            >>>         "distill_loss": distill_loss.detach(),
            >>>         "total_loss": distill_loss.detach()  # 总损失就是蒸馏损失
            >>>     }
            >>>
            >>>     return distill_loss, loss_items
        """
        return super().execute_forward(batch_samples)

    # ------------------------------------------------------------------
    # 3. 训练流程钩子
    # ------------------------------------------------------------------
    def freeze_layers(self, model: TTBaseModel):
        """
        蒸馏场景下仅冻结教师模型参数（学生模型照常训练）。

        Args:
            model: 此处实际传入的是学生模型（TTBaseTrainer.model）。
        """
        super().freeze_layers(model)

        for param in self.teacher_model.parameters():
            param.requires_grad = False
