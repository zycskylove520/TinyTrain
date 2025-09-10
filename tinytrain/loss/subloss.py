import torch
import torch.nn.functional as F

from torch import nn

from tinytrain.utils.box_utils import bbox_iou_torch
from tinytrain.utils.tal import bbox2dist


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class DFLoss(nn.Module):
    """
    Distribution Focal Loss（DFL）
    —— 用于对边界框位置进行“分布”建模的连续标签损失。
    出自论文《Generalized Focal Loss: Towards Efficient Representation Learning for Dense Object Detection》
    https://ieeexplore.ieee.org/document/9792391

    作用
    ----
    传统回归将框位置视为单点（Dirac δ），GF Loss 把位置离散化为 `reg_max` 个区间，
    用 Softmax 输出分布，再用加权交叉熵逼近连续标签，从而：
    1. 提高定位精度；
    2. 对边界模糊或遮挡目标更鲁棒。

    公式
    ----
    对某一维坐标 y，将其落在区间 [tl, tr] 内，则
    loss = wl * CE(p, tl) + wr * CE(p, tr)
    其中 wl = (tr - y)，wr = (y - tl)。
    """

    def __init__(self, reg_max=16):
        """
        Args:
            reg_max (int): 离散区间数量（论文默认 16）。
        """
        super(DFLoss, self).__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target_dist):
        """
        计算 DFL 损失。

        Args:
            pred_dist (Tensor): 模型输出的分布 logits，形状 (..., reg_max)。
            target_dist (Tensor): 连续标签，取值 [0, reg_max-1]，形状与 pred_dist 相同。

        Returns:
            Tensor: 逐元素 DFL 损失，形状与输入一致，最后再 mean(-1, keepdim=True)。
        """
        target_dist = target_dist.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target_dist.long()  # target_dist left
        tr = tl + 1  # target_dist right
        wl = tr - target_dist  # weight left
        wr = 1 - wl  # weight right
        # loss = -(y_(i+1)-y)*log(p_i)+(y-Y_i)*log(p_(i+1)))
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BoxLoss(nn.Module):
    """
    框回归损失类，计算预测框和真实框之间的损失。
    使用 IoU 损失作为框回归损失。
    """

    def __init__(self):
        super(BoxLoss, self).__init__()

    def forward(self, pred_bboxes, gt_bboxes):
        """
        计算框回归损失。

        Args:
            pred_bboxes (Tensor): 预测的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。
            gt_bboxes (Tensor): 真实的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。

        Returns:
            Tensor: 框回归损失。
        """
        return self.iou_loss(pred_bboxes, gt_bboxes)

    def iou_loss(self, pred_bboxes, gt_bboxes):
        """
        计算 IoU 损失。

        Args:
            pred_bboxes (Tensor): 预测的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。
            gt_bboxes (Tensor): 真实的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。

        Returns:
            Tensor: IoU 损失。
        """
        ious = self.compute_iou(pred_bboxes, gt_bboxes)
        iou_loss = 1 - ious.mean()
        return iou_loss

    def compute_iou(self, pred_bboxes, gt_bboxes):
        """
        计算 IoU。

        Args:
            pred_bboxes (Tensor): 预测的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。
            gt_bboxes (Tensor): 真实的边界框，形状为 (B, N, 4)，格式为 (lx, ly, rx, ry)。

        Returns:
            Tensor: IoU 值。
        """
        # 计算交集
        inter_x1 = torch.max(pred_bboxes[:, :, 0], gt_bboxes[:, :, 0])
        inter_y1 = torch.max(pred_bboxes[:, :, 1], gt_bboxes[:, :, 1])
        inter_x2 = torch.min(pred_bboxes[:, :, 2], gt_bboxes[:, :, 2])
        inter_y2 = torch.min(pred_bboxes[:, :, 3], gt_bboxes[:, :, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算并集
        pred_area = (pred_bboxes[:, :, 2] - pred_bboxes[:, :, 0]) * (pred_bboxes[:, :, 3] - pred_bboxes[:, :, 1])
        gt_area = (gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]) * (gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1])
        union_area = pred_area + gt_area - inter_area

        # 计算 IoU
        ious = inter_area / union_area
        return ious


class BboxLossWithDFL(nn.Module):
    """
    边界框回归损失 = IoU 损失 + 可选 DFL 损失。

    说明
    ----
    1. IoU 损失：CIoU 形式，仅对正样本（fg_mask）计算。
    2. DFL 损失：当 reg_max>1 时启用，对正样本应用 DFLoss。
    3. 所有损失按 `target_scores_sum` 归一化，确保不同 batch 大小可比。
    """

    def __init__(self, reg_max=16):
        """
        Args:
            reg_max (int): 离散区间数量。>1 时启用 DFL，否则仅 IoU。
        """
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        计算 IoU 损失与 DFL 损失。

        Args:
            pred_dist (Tensor): 分布 logits (B, N, 4*reg_max)。
            pred_bboxes (Tensor): 解码后的框坐标 (B, N, 4)。
            anchor_points (Tensor): 锚点中心 (N, 2)。
            target_bboxes (Tensor): 匹配后的 GT 框 (B, N, 4)。
            target_scores (Tensor): 匹配后的 GT 分数 (B, N, 1)。
            target_scores_sum (Tensor): 分数和，用于归一化。
            fg_mask (Tensor): 正样本布尔掩码 (B, N)。

        Returns:
            tuple:
                - loss_iou (Tensor): 平均 IoU 损失。
                - loss_dfl (Tensor): 平均 DFL 损失（若 reg_max=1 则返回 0）。
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou_torch(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
