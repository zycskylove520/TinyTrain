from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from typing import TYPE_CHECKING

from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry, lxlyrxry_2_cxcywh, make_anchors
from tinytrain.modules.assigner.taa_assigner import TaskAlignedAssigner, dist2bbox
from tinytrain.utils.segment_utils import crop_mask

from .base.base_loss import BaseLoss
from .subloss import BboxLossWithDFL, KeypointLoss

if TYPE_CHECKING:
    from tinytrain.data.data_format import ClassifyBatchDataInfo, DetectBatchDataInfo, PoseBatchDataInfo, SegmentBatchDataInfo
    from tinytrain.models.ocr.ocr_data_format import LPRBatchDataInfo


class ClassificationLoss(BaseLoss):
    """
    通用分类损失封装，默认使用 CrossEntropyLoss。
    可通过 cls_loss_gain 对最终损失进行缩放。
    """

    def __init__(self, cls_loss_gain: float):
        """
        Args:
            cls_loss_gain: 分类损失整体权重系数。
        """
        super().__init__()
        self.cls_loss_gain = cls_loss_gain
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        计算分类损失。

        Args:
            pred: 模型输出 logits，形状 (B, C)。
            batch: 批数据，需包含 `.target` 字段，形状 (B,)，值为类别索引。

        Returns:
            loss: 标量损失。
            loss_items: dict，记录各分量，便于日志打印。
        """
        loss = self.criterion(pred, batch.target) * self.cls_loss_gain
        loss_items = {"cls_loss": loss.detach()}
        return loss, loss_items


class ClassificationWithFocalLoss(BaseLoss):
    """
    Focal Loss，用于缓解类别不平衡问题。
    在 CE 基础上加入 (1-pt)^γ 调制因子，并可指定类别权重 alpha。
    """

    def __init__(self, cls_loss_gain: float, alpha=None, gamma=0, eps=1e-7):
        """
        Args:
            cls_loss_gain: 损失权重系数。
            alpha: 类别权重，可为 float 或长度=C 的 Tensor。
            gamma: 聚焦参数，越大则“易分样本”权重越低。
            eps: 数值保护小量，防止 log(0)。
        """
        super().__init__()
        self.cls_loss_gain = cls_loss_gain
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        计算 Focal Loss。

        Args:
            pred: 模型输出 logits，(B, C)。
            batch: 批数据，含 `.target`，(B,)。

        Returns:
            loss: 标量。
            loss_items: dict。
        """
        target = batch.target

        # logp = self.criterion(pred, target)
        # p = torch.exp(-logp)
        # loss = (1 - p) ** self.gamma * logp
        # loss = loss * self.cls_loss_gain
        # loss_items = {"cls_loss": loss.detach()}
        # return loss, loss_items

        # 计算交叉熵损失
        ce_loss = self.criterion(pred, target)

        # 计算每个样本的预测概率
        probabilities = F.softmax(pred, dim=1)
        pt = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)

        # 计算 Focal Loss
        focal_term = (1 - pt + self.eps) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = self.alpha
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, target)
            else:
                raise ValueError("alpha must be a float or a Tensor")
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        focal_loss = focal_loss.mean() * self.cls_loss_gain
        loss_items = {"cls_loss": focal_loss.detach()}
        return focal_loss, loss_items


class LPRCTCLoss(BaseLoss):
    def __init__(self, lpr_loss_gain: float, blank: int, reduction: str = "mean"):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank, reduction=reduction)
        self.lpr_loss_gain = lpr_loss_gain

    def forward(self, pred: torch.Tensor, batch: LPRBatchDataInfo):
        log_probs = pred.permute(2, 0, 1)  # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(dim=-1).requires_grad_()

        lengths = batch.lengths
        input_lengths, target_lengths = self.sparse_tuple_for_ctc(log_probs.size(0), lengths)

        loss = self.criterion(log_probs, batch.target, input_lengths=input_lengths, target_lengths=target_lengths) * self.lpr_loss_gain
        loss_items = {"lpr_loss": loss.detach()}
        return loss, loss_items

    @staticmethod
    def sparse_tuple_for_ctc(T_length, lengths):
        input_lengths = []
        target_lengths = []

        for ch in lengths:
            input_lengths.append(T_length)
            target_lengths.append(ch)

        return tuple(input_lengths), tuple(target_lengths)


class YOLOV8DetectionLoss(BaseLoss):
    """
    YOLOv8 检测头损失，包含：
        1. 分类损失 (BCE)
        2. 框回归损失 (CIoU + DFL)
    使用 Task-Aligned Assigner 完成正负样本分配。
    """

    def __init__(self, nc, strides, reg_max, imgsz, device, cls_gain=1, box_gain=1, dfl_gain=1, tal_topk=10):
        """
        Args:
            nc: 类别数。
            strides: 各检测层下采样步长，如 [8,16,32]。
            reg_max: DFL 分支离散区间数，>1 时启用 DFL。
            imgsz: 训练输入分辨率 (W,H) 或单值。
            device: 运行设备。
            cls_gain: 分类损失权重。
            box_gain: 框回归权重。
            dfl_gain: DFL 损失权重。
            tal_topk: Assigner 的 top-k 参数。
        """
        super().__init__()
        self.device = device
        self.strides = strides
        self.nc = nc
        self.reg_max = reg_max
        self.output_num = self.nc + self.reg_max * 4
        self.imgsz = list(imgsz) if isinstance(imgsz, (list, tuple)) else [imgsz, imgsz]  # w,h
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain

        self.use_dfl = self.reg_max > 1

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLossWithDFL(self.reg_max)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        将输入 targets 整理成 (B, max_obj, 5) 并缩放到输入分辨率。

        Args:
            targets: [N,6] (img_idx, cls, cx, cy, w, h)
            batch_size: 批大小
            scale_tensor: [4] (W,H,W,H) 用于乘以归一化坐标

        Returns:
            Tensor: (B, max_obj, 5) (cls, lx, ly, rx, ry)
        """
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            # counts获取的是该批次中每个图片里的目标的个数，以列表的形式返回
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # 选取目标数最多的建立out矩阵，目的是让所有的图片的目标都能用相同大小的矩阵进行合批
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                # 获取第j张图片有多少个目标
                n = matches.sum()  # type: ignore[arg-type]
                if n:
                    # 把第j张图片的前n个目标设置上对应的目标
                    out[j, :n] = targets[matches, 1:]
            # 这里将bbox格式转成了lxlyrxry格式，并且通过scale_tensor缩放回了训练图大小
            out[..., 1:5] = cxcywh_2_lxlyrxry(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        将锚点 + DFL 分布解码为框坐标。

        Args:
            anchor_points: [H*W, 2] 锚点中心
            pred_dist: [B, H*W, 4*reg_max]

        Returns:
            [B, H*W, 4] (lx,ly,rx,ry)
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 将最后一个维度归一化，计算ltrb四个坐标的预测值
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds: list[torch.Tensor], batch: DetectBatchDataInfo):
        """
        计算检测损失。

        Args:
            preds: 三个特征层输出，每个 [B, C+4*reg_max, Hi, Wi]
            batch: 批数据，含 bboxes / target / bboxes_idx

        Returns:
            total_loss: 标量
            loss_items: dict
        """
        dtype = preds[0].dtype
        batch_size = preds[0].shape[0]

        # 分离类别和boxes
        pred_distri, pred_scores = torch.cat([pred.view(batch_size, self.output_num, -1) for pred in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds, self.strides, 0.5)

        # Targets
        targets = torch.cat((batch.bboxes_idx.view(-1, 1), batch.target.view(-1, 1), batch.bboxes.view(-1, 4)), 1)
        targets = self.preprocess(targets, batch_size, torch.tensor(self.imgsz)[[0, 1, 0, 1]].to(self.device))

        # 获得真实类别标签和真实bboxes
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 通过对gt_bboxes进行sum操作来找出正样本的边界框，再通过gt_来获取非0的边界框的bool矩阵
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 解码后的preds的bbox，此时的坐标值为lxlyrxry
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            # 乘以stride_tensor将pred_bboxes恢复训练图大小，对anchor_points同理
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        """开始计算loss"""
        cls_loss = torch.tensor(0., dtype=dtype).to(self.device)
        box_loss = torch.tensor(0., dtype=dtype).to(self.device)
        dfl_loss = torch.tensor(0., dtype=dtype).to(self.device)

        # cls loss
        # 这里限制target_scores_sum最小必须为1
        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss & DFL loss
        # fg_mask.sum()保证一定有锚框有真实边界框对应
        if fg_mask.sum():
            # 缩放到特征图大小
            target_bboxes /= stride_tensor
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        cls_loss *= self.cls_gain  # cls gain
        box_loss *= self.box_gain  # box gain
        dfl_loss *= self.dfl_gain  # dfl gain
        total_loss = (cls_loss + box_loss + dfl_loss) * batch_size
        loss_items = {"cls_loss": cls_loss.detach(), "box_loss": box_loss.detach(), "dfl_loss": dfl_loss.detach()}

        return total_loss, loss_items  # loss(box, cls, dfl)


class YOLOV8PoseLoss(YOLOV8DetectionLoss):
    """
    YOLOv8 姿态估计头损失,在检测损失基础上增加：
        4. 关键点回归损失 (OKS)
        5. 关键点可见性 loss (BCE)
    """
    OKS_SIGMA = (
            np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
            / 10.0
    )

    def __init__(self, nc, strides, reg_max, imgsz, device, kpt_shape, cls_gain=1, box_gain=1, dfl_gain=1, pose_gain=1, kobj_gain=1, tal_topk=10):  # model must be de-paralleled
        """
        Args:
            kpt_shape: 关键点维度, 如 [17,3] 表示 17 个点, 每个点 x,y,visible
            pose_gain: 关键点回归 loss 权重
            kobj_gain: 关键点可见性 loss 权重
        """
        super().__init__(nc, strides, reg_max, imgsz, device, cls_gain, box_gain, dfl_gain, tal_topk)
        self.pose_gain = pose_gain
        self.kobj_gain = kobj_gain
        self.kpt_shape = kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        humanoid_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(self.OKS_SIGMA).to(self.device) if humanoid_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def forward(self, preds: tuple[list[torch.Tensor], torch.Tensor], batch: PoseBatchDataInfo):
        """
        Args:
            preds[0]: 三个检测层输出
            preds[1]: 关键点头输出 [B, nkpt*dim, Hi, Wi]

        Returns:
            total_loss, loss_items
        """
        features: list[torch.Tensor] = preds[0]
        pred_kpts: torch.Tensor = preds[1]
        dtype = features[0].dtype
        batch_size = features[0].shape[0]

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.output_num, -1) for xi in features], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(features, self.strides, 0.5)

        # Targets
        targets = torch.cat((batch.bboxes_idx.view(-1, 1), batch.target.view(-1, 1), batch.bboxes.view(-1, 4)), 1)
        targets = self.preprocess(targets, batch_size, torch.tensor(self.imgsz)[[0, 1, 0, 1]].to(self.device))

        # 获得真实类别标签和真实bboxes
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 通过对gt_bboxes进行sum操作来找出正样本的边界框，再通过gt_来获取非0的边界框的bool矩阵
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 解码后的preds的bbox，此时的坐标值为lxlyrxry
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            # 乘以stride_tensor将pred_bboxes恢复训练图大小，对anchor_points同理
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        """开始计算loss"""
        cls_loss = torch.tensor(0., dtype=dtype).to(self.device)
        box_loss = torch.tensor(0., dtype=dtype).to(self.device)
        dfl_loss = torch.tensor(0., dtype=dtype).to(self.device)
        pose_loss = torch.tensor(0., dtype=dtype).to(self.device)
        kobj_loss = torch.tensor(0., dtype=dtype).to(self.device)

        # cls loss
        # 这里限制target_scores_sum最小必须为1
        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss & DFL loss & Pose loss & KObj loss
        # fg_mask.sum()保证一定有锚框有真实边界框对应
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            # 把归一化的keypoints放大回训练图
            keypoints = batch.batch_keypoints.to(self.device).float().clone()
            keypoints[..., 0] *= self.imgsz[0]
            keypoints[..., 1] *= self.imgsz[1]

            pose_loss, kobj_loss = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch.bboxes_idx.view(-1, 1), stride_tensor, target_bboxes, pred_kpts
            )

        cls_loss *= self.cls_gain  # cls gain
        box_loss *= self.box_gain  # box gain
        dfl_loss *= self.dfl_gain  # dfl gain
        pose_loss *= self.pose_gain  # pose_gain
        kobj_loss *= self.kobj_gain  # kobj_gain
        total_loss = (cls_loss + box_loss + dfl_loss + pose_loss + kobj_loss) * batch_size
        loss_items = {
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "pose_loss": pose_loss.detach(),
            "kobj_loss": kobj_loss.detach(),
        }

        return total_loss, loss_items

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """
        将关键点头输出解码为坐标。

        Args:
            anchor_points: [H*W,2]
            pred_kpts: [B,H*W,nkpt,3]

        Returns:
            [B,H*W,nkpt,3] x,y 已加上 anchor 坐标
        """
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts):
        """
        计算关键点回归损失与可见性损失。

        Args:
            fg_mask: [B,H*W] 正样本锚框 mask
            target_gt_idx: [B,H*W] 每个正锚框对应哪个 gt
            keypoints: [N_gt,nkpt,3] 归一化 gt 关键点
            batch_idx: [N_gt,1] 每张关键点所属 batch 索引
            stride_tensor: [H*W,1] 特征层 stride
            target_bboxes: [B,H*W,4] 分配后的 gt 框（已除以 stride）
            pred_kpts: [B,H*W,nkpt,3] 预测关键点

        Returns:
            pose_loss: 关键点回归 loss
            kobj_loss: 可见性 loss
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = lxlyrxry_2_cxcywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class YOLOV8SegmentLoss(YOLOV8DetectionLoss):
    """
    YOLOv8 实例分割头损失,在检测损失基础上增加：
        6. 实例分割 mask 损失
    """

    def __init__(self, nc, strides, reg_max, imgsz, device, overlap_mask, cls_gain=1, box_gain=1, dfl_gain=1, seg_gain=1, tal_topk=10):  # model must be de-paralleled
        """
        Args:
            overlap_mask: 数据集是否采用重叠 mask（COCO 式）
            seg_gain: mask 损失权重
        """
        super().__init__(nc, strides, reg_max, imgsz, device, cls_gain, box_gain, dfl_gain, tal_topk)
        self.seg_gain = seg_gain
        self.overlap = overlap_mask

    def forward(self, preds: tuple[list[torch.Tensor], torch.Tensor, torch.Tensor], batch: SegmentBatchDataInfo):
        """
        Args:
            preds[0]: 检测层输出
            preds[1]: mask 系数头 [B, nm, Hi, Wi]
            preds[2]: proto mask [B, nm, Hp, Wp]

        Returns:
            total_loss, loss_items
        """
        features, pred_masks, proto = preds
        dtype = features[0].dtype
        batch_size, _, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = torch.cat([xi.view(batch_size, self.output_num, -1) for xi in features], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(features, self.strides, 0.5)

        # Targets
        targets = torch.cat((batch.bboxes_idx.view(-1, 1), batch.target.view(-1, 1), batch.bboxes.view(-1, 4)), 1)
        targets = self.preprocess(targets, batch_size, torch.tensor(self.imgsz)[[0, 1, 0, 1]].to(self.device))

        # 获得真实类别标签和真实bboxes
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 通过对gt_bboxes进行sum操作来找出正样本的边界框，再通过gt_来获取非0的边界框的bool矩阵
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 解码后的preds的bbox，此时的坐标值为lxlyrxry
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            # 乘以stride_tensor将pred_bboxes恢复训练图大小，对anchor_points同理
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        """开始计算loss"""
        cls_loss = torch.tensor(0., dtype=dtype).to(self.device)
        box_loss = torch.tensor(0., dtype=dtype).to(self.device)
        dfl_loss = torch.tensor(0., dtype=dtype).to(self.device)
        seg_loss = torch.tensor(0., dtype=dtype).to(self.device)

        # cls loss
        # 这里限制target_scores_sum最小必须为1
        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss & DFL loss & Segment loss
        # fg_mask.sum()保证一定有锚框有真实边界框对应
        if fg_mask.sum():
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask
            )

            masks = batch.batch_masks.to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            seg_loss = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch.bboxes_idx.view(-1, 1), proto, pred_masks, torch.tensor(self.imgsz)[[1, 0]], self.overlap
            )
        else:
            seg_loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        cls_loss *= self.cls_gain  # cls gain
        box_loss *= self.box_gain  # box gain
        dfl_loss *= self.dfl_gain  # dfl gain
        seg_loss *= self.seg_gain  # segment gain
        total_loss = (cls_loss + box_loss + dfl_loss + seg_loss) * batch_size
        loss_items = {
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "seg_loss": seg_loss.detach()
        }

        return total_loss, loss_items

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        """
        单张图片的实例分割损失。

        Args:
            gt_mask: [n, Hp, Wp] 对应 n 个正样本的 gt
            pred: [n, nm] mask 系数
            proto: [nm, Hp, Wp] 原型 mask
            xyxy: [n,4] 归一化 gt 框
            area: [n] 框面积

        Returns:
            mask 损失标量
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self,
                                    fg_mask: torch.Tensor,
                                    masks: torch.Tensor,
                                    target_gt_idx: torch.Tensor,
                                    target_bboxes: torch.Tensor,
                                    batch_idx: torch.Tensor,
                                    proto: torch.Tensor,
                                    pred_masks: torch.Tensor,
                                    imgsz: torch.Tensor,
                                    overlap: bool,
                                    ) -> torch.Tensor:
        """
        计算整批分割损失。

        Args:
            fg_mask: [B, H*W] 正样本
            masks: 若 overlap=False -> [B, Hp, Wp]；否则为 [B, ?, Hp, Wp]
            target_gt_idx: [B, H*W] 正样本对应 gt 索引
            target_bboxes: [B, H*W, 4] 已缩放为输入分辨率
            batch_idx: [N_gt,1]
            proto: [B, nm, Hp, Wp]
            pred_masks: [B, H*W, nm]
            imgsz: (H,W) 输入分辨率
            overlap: 是否采用重叠 mask

        Returns:
            seg 损失标量
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]].to(self.device)

        # Areas of target bboxes
        marea = lxlyrxry_2_cxcywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()  # type: ignore[arg-type]
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()
