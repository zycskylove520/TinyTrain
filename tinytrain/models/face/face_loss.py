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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Callable
from torch import distributed

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.global_var import RANK, WORLD_SIZE
from tinytrain.loss.base.base_loss import TTBaseLoss


class PartialFCLoss(TTBaseLoss):
    """
    分布式 Partial FC 训练的头层 + 损失封装。

    功能
    ----
    1. 仅在本卡（rank）维护 **部分分类中心** (`num_local`)，大幅减少 GPU 显存与通信量。
    2. 支持 **随机采样负类中心** (`sample_rate < 1`)，进一步加速超大规模类别训练。
    3. 对特征与中心做 **L2 归一化**，保证度量学习性质。
    4. 通过 `margin_softmax` 回调（ArcFace / CosFace 等）施加 margin 后再计算 `DistCrossEntropy`。
    5. 最终 loss 乘以 `cls_loss_gain` 后回传。

    要求
    ----
    - 必须已初始化 `torch.distributed`，且 world_size ≥ 2（否则断言失败）。
    - 每张卡输入的 `batch_size` 必须完全相同（all_gather 对齐）。
    - 标签 -1 会被忽略（与 margin 模块约定一致）。

    Args
    ----
    margin_loss : Callable
        margin 变换函数，签名 `(logits, labels) -> logits`；通常传入 `ArcFace / CosFace / CombinedMargin` 实例。
    device : torch.device
        当前 rank 的计算设备。
    embedding_size : int
        特征维度。
    num_classes : int
        全局总类别数。
    sample_rate : float, default 1.0
        负类中心采样比例。1.0 表示使用全部本地中心；<1.0 则随机保留对应比例。
    cls_loss_gain : float, default 1.0
        分类损失权重，最终 `loss = ce_loss * cls_loss_gain`。
    """

    def __init__(self, margin_loss: Callable, device, embedding_size: int, num_classes: int, sample_rate: float = 1.0, cls_loss_gain: float = 1.0):
        super().__init__()
        assert WORLD_SIZE >= 2, "At least two GPUs are required."
        self.device = device

        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.num_local: int = num_classes // WORLD_SIZE + int(
            RANK < num_classes % WORLD_SIZE
        )
        self.class_start: int = num_classes // WORLD_SIZE * RANK + min(
            RANK, num_classes % WORLD_SIZE
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0

        self.weight = nn.Parameter(torch.empty(self.num_local, embedding_size))
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0, std=1)
            self.weight.copy_(F.normalize(self.weight, p=2, dim=1))

        self.margin_softmax = margin_loss
        self.dist_cross_entropy = DistCrossEntropy()
        self.cls_loss_gain: float = cls_loss_gain

    def sample(self, labels, index_positive):
        """
        为当前迭代采样负类中心，保证正类一定被包含。

        Args
        ----
        labels : Tensor[batch, 1]
            全局标签（已映射到本地索引或 -1）。
        index_positive : Tensor[bool]
            指示哪些位置属于本卡正类。

        Returns
        -------
        Tensor[num_sample, embedding_size]
            采样后的中心权重子集。
        同时就地更新 `labels`，使其指向采样后的索引。
        """
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).to(self.device)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local]).to(self.device)
                perm[positive] = 2.0  # 强制保留正类
                index = torch.topk(perm, k=self.num_sample)[1].to(self.device)
                index = index.sort()[0].to(self.device)
            else:
                index = positive  # 采样数不足，仅保留正类
            weight_index = index

            # 将全局标签重新映射到采样后的局部索引
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[weight_index]

    def forward(self, local_embeddings: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        前向：收集全局特征 → 计算局部 logits → 施加 margin → 计算 DistCrossEntropy。

        Args
        ----
        local_embeddings : Tensor[batch, embedding_size]
            当前卡提取的特征。
        batch : ClassifyBatchDataInfo
            必须包含 `batch.target`（全局标签）。

        Returns
        ----
        loss : Tensor[]
            缩放后的分类损失。
        loss_items : dict
            仅包含 `cls_loss` 的 detach 值，用于日志。
        """

        local_labels: torch.Tensor = batch.target
        local_labels = local_labels.long()

        batch_size = local_embeddings.shape[0]

        # 要保证每张卡的batch_size相同，否则无法做all_gather
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        # 收集所有卡的local_embeddings和local_labels
        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size), device=self.device, dtype=local_embeddings.dtype) for _ in range(WORLD_SIZE)
        ]
        _gather_labels = [
            torch.zeros(batch_size, device=self.device).long() for _ in range(WORLD_SIZE)
        ]
        _list_embeddings = all_gather_with_grad(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
                labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            weight = self.weight

        norm_embeddings = F.normalize(embeddings)
        norm_weight_activated = F.normalize(weight).to(self.device)
        logits = F.linear(norm_embeddings, norm_weight_activated)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels) * self.cls_loss_gain
        loss_items = {"cls_loss": loss.detach()}

        return loss, loss_items


class FCLoss(nn.Module):
    """
    单卡 / 数据并行 场景下的标准分类头损失。

    功能
    ----
    1. 维护 **全量分类中心** (`nn.Parameter`)。
    2. 对特征与中心做 **L2 归一化**。
    3. 通过 `margin_softmax` 回调施加 margin（ArcFace / CosFace 等）。
    4. 使用 `nn.CrossEntropyLoss` 计算分类损失，并乘以 `cls_loss_gain`。

    Args
    ----
    margin_loss : Callable
        margin 变换函数，签名 `(logits, labels) -> logits`。
    device : torch.device
        计算设备。
    embedding_size : int
        特征维度。
    num_classes : int
        总类别数。
    cls_loss_gain : float, default 1.0
        分类损失权重。
    """

    def __init__(self, margin_loss: Callable, device, embedding_size: int, num_classes: int, cls_loss_gain: float = 1.0):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size).to(device))
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0, std=1)
            self.weight.copy_(F.normalize(self.weight, p=2, dim=1))

        self.margin_softmax = margin_loss
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cls_loss_gain: float = cls_loss_gain

    def forward(self, embeddings: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        前向：归一化 → 线性映射 → 施加 margin → CrossEntropyLoss。

        Args
        ----
        embeddings : Tensor[batch, embedding_size]
            网络输出特征。
        batch : ClassifyBatchDataInfo
            必须包含 `batch.target`（全局标签，-1 表示忽略）。

        Returns
        ----
        loss : Tensor[]
            缩放后的分类损失。
        loss_items : dict
            仅包含 `cls_loss` 的 detach 值。
        """
        labels: torch.Tensor = batch.target
        labels = labels.long()

        norm_embeddings = F.normalize(embeddings)
        norm_weight_activated = F.normalize(self.weight).to(self.device)
        logits = F.linear(norm_embeddings, norm_weight_activated)

        logits = self.margin_softmax(logits, labels)
        loss = self.cross_entropy(logits, labels) * self.cls_loss_gain
        loss_items = {"cls_loss": loss.detach()}

        return loss, loss_items


def all_gather_with_grad(tensor, *gather_list):
    """AllGather op with gradient backward"""
    return AllGatherWithGradFunc.apply(tensor, *gather_list)


class AllGatherWithGradFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        dist.all_gather(gather_list, tensor)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        rank = dist.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            dist.reduce(grad_out, rank, dist.ReduceOp.SUM, async_op=True)
            if i == rank
            else dist.reduce(
                grad_list[i], i, dist.ReduceOp.SUM, async_op=True
            )
            for i in range(dist.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()

        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return grad_out, *[None for _ in range(len(grad_list))]


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        dist.all_reduce(max_logits, dist.ReduceOp.MAX)
        logits = logits - max_logits
        logits = logits.exp()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        dist.all_reduce(sum_logits_exp, dist.ReduceOp.SUM)
        logits = logits / sum_logits_exp
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        dist.all_reduce(loss, dist.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)
