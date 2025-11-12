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

from torch import nn

from tinytrain.utils import LOGGER

from tinytrain.utils.box_utils import box_iou_1v1


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner（TAL）

    主要功能
    --------
    1. 将任意数量的 GT 框动态地分配给 anchor，解决正负样本极度不平衡问题。
    2. 引入 **Task-Aligned Metric**（分类分 * α + IoU * β）作为 anchor 与 GT 的匹配标准。
    3. 支持 CUDA OOM 自动回退 CPU，保证训练稳定。

    用法示例
    --------
    >>> tal = TaskAlignedAssigner(topk=13, num_classes=80, alpha=1.0, beta=6.0)
    >>> target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = tal(
    ...     pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    ... )
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """
        初始化 TAL 超参数。

        Args:
            topk (int): 每个 GT 最多保留的候选 anchor 数。
            num_classes (int): 类别数，用于生成 one-hot。
            alpha (float): 分类分数在 Task-Aligned Metric 中的权重。
            beta (float): IoU 在 Task-Aligned Metric 中的权重。
            eps (float): 防止除零的小量。
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        前向计算，完成 anchor-to-GT 的匹配并生成训练目标。

        Args:
            pd_scores (Tensor): (bs, num_total_anchors, num_classes) 预测类别概率。
            pd_bboxes (Tensor): (bs, num_total_anchors, 4) 已解码预测框（xyxy格式，训练图尺寸）。
            anc_points (Tensor): (num_total_anchors, 2) anchor 中心点坐标。
            gt_labels (Tensor): (bs, n_max_boxes, 1) GT 类别。
            gt_bboxes (Tensor): (bs, n_max_boxes, 4) 已解码GT 框（xyxy格式，训练图尺寸）。
            mask_gt (Tensor): (bs, n_max_boxes, 1) 有效 GT 掩码（非无效框）。

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                target_labels   (bs, num_total_anchors)   每个 anchor 的类别标签
                target_bboxes   (bs, num_total_anchors,4) 每个 anchor 的回归目标
                target_scores   (bs, num_total_anchors,C) one-hot+对齐分数
                fg_mask         (bs, num_total_anchors)   前景掩码
                target_gt_idx   (bs, num_total_anchors)   anchor 对应 GT 索引
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """CPU/GPU 统一的实际计算逻辑，见 forward 的返回说明。"""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """生成正样本掩码、对齐度量和 IoU。"""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """计算 Task-Aligned Metric = cls^α * IoU^β。"""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算水平框 IoU，返回 shape (N,)"""
        return box_iou_1v1(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        为每个 GT 保留 top-k anchor。

        Args:
            metrics (Tensor): (b, n_max_boxes, h*w) Task-Aligned Metric。
            topk (int): 保留数。
            largest (bool): True 取最大，False 取最小。
            topk_mask (Tensor, optional): (b, n_max_boxes, topk) 额外掩码。

        Returns:
            Tensor: (b, n_max_boxes, h*w) 0/1 mask。
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k: k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        根据匹配结果生成训练目标。

        Args:
            gt_labels (Tensor): (b, n_max_boxes, 1) GT 类别。
            gt_bboxes (Tensor): (b, n_max_boxes, 4) GT 框。
            target_gt_idx (Tensor): (b, h*w) anchor 对应 GT 索引。
            fg_mask (Tensor): (b, h*w) 前景掩码。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                target_labels, target_bboxes, target_scores
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        判断 anchor 中心是否落在 GT 框内。

        Args:
            xy_centers (Tensor): (h*w, 2) anchor 中心。
            gt_bboxes (Tensor): (b, n_max_boxes, 4) GT 框 [x1, y1, x2, y2]。
            eps (float): 边缘容差。

        Returns:
            Tensor: (b, n_max_boxes, h*w) bool mask。
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        处理一个 anchor 被多个 GT 选中的冲突：保留 IoU 最大者。

        Args:
            mask_pos (Tensor): (b, n_max_boxes, h*w) 正样本掩码。
            overlaps (Tensor): (b, n_max_boxes, h*w) IoU。
            n_max_boxes (int): 最大 GT 数。

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                target_gt_idx, fg_mask, 更新后的 mask_pos
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


def dist2bbox(distance, anchor_points, xywh: bool = True, dim: int = -1):
    """
    将距离预测 (ltrb) 转换为边界框坐标。

    Args:
        distance (Tensor): (..., 4) 预测距离 [left, top, right, bottom]。
        anchor_points (Tensor): (..., 2) anchor 中心点。
        xywh (bool): True 返回 [cx, cy, w, h]，False 返回 [x1, y1, x2, y2]。
        dim (int): 指定分割维度。

    Returns:
        Tensor: 转换后的边界框，形状与 distance 相同。
    """
    lt, rb = distance.chunk(2, dim)
    # 计算左上角点的预测值标
    x1y1 = anchor_points - lt
    # 计算右下角点的预测值标
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    # 返回(batch, num_anchors, 4)
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """
    将边界框转换为距离格式 (ltrb)。

    Args:
        anchor_points (Tensor): (..., 2) anchor 中心点。
        bbox (Tensor): (..., 4) 边界框 [x1, y1, x2, y2]。
        reg_max (float): 距离上限，防止溢出。

    Returns:
        Tensor: (..., 4) 距离 [left, top, right, bottom]，范围 [0, reg_max)。
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    将旋转框距离预测解码为旋转框坐标 [cx, cy, w, h, angle(rad)]。

    Args:
        pred_dist (Tensor): (..., 4) 距离 [left, top, right, bottom]。
        pred_angle (Tensor): (..., 1) 角度（弧度）。
        anchor_points (Tensor): (..., 2) anchor 中心点。
        dim (int): 指定分割维度。

    Returns:
        Tensor: (..., 4) [cx, cy, w, h]（已旋转）。
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
