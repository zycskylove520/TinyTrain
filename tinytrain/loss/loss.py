import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from tinytrain.data.data_format import ClassifyBatchDataInfo, DetectBatchDataInfo, PoseBatchDataInfo
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry, lxlyrxry_2_cxcywh, make_anchors
from tinytrain.utils.tal import TaskAlignedAssigner, dist2bbox

from .subloss import BboxLossWithDFL, KeypointLoss


class ClassificationLoss(nn.Module):
    """
    通用分类损失封装，默认使用 CrossEntropyLoss。
    支持通过 cls_loss_gain 进行损失缩放。
    """

    def __init__(self, cls_loss_gain: float):
        """
        Args:
            cls_loss_gain (float): 分类损失权重系数。
        """
        super().__init__()
        self.cls_loss_gain = cls_loss_gain
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        计算分类损失。

        Args:
            pred (torch.Tensor): 模型输出，通常只含一个 (B, C) 张量。
            batch (ClassifyBatchDataInfo): 批数据，包含 `target` 标签，形状为 (B,)，值为类别索引。

        Returns:
            tuple:
                - loss (torch.Tensor): 标量损失值。
                - loss_items (dict): 各分量损失，便于日志记录。
        """
        loss = self.criterion(pred, batch.target) * self.cls_loss_gain
        loss_items = {"cls_loss": loss.detach()}
        return loss, loss_items


class ClassificationWithFocalLoss(nn.Module):
    """
    Focal Loss，一种改进的分类损失函数，用于解决类别不平衡问题。
    Focal Loss 在 CrossEntropyLoss 的基础上，对易分类样本的损失进行加权降低，
    从而让模型更关注难分类的样本，有助于提高模型在不平衡数据集上的性能。
    支持通过 gamma 参数调整对易分类样本的惩罚程度，并通过 alpha 参数对不同类别进行加权。
    """

    def __init__(self, cls_loss_gain: float, alpha=None, gamma=0, eps=1e-7):
        """
        初始化 ClassificationWithFocalLoss。

        Args:
            cls_loss_gain (float): 分类损失权重系数。
            alpha (float or Tensor, optional): 类别权重。如果为标量，则所有类别共享相同的权重；
                                               如果为 Tensor，则应与类别数量相同。默认为 None。
            gamma (float, optional): 聚焦参数，控制对易分类样本的惩罚程度。gamma 越大，对易分类样本的损失惩罚越小。默认为 0。
            eps (float, optional): 数值稳定性参数，避免 log(0) 等数值问题。默认为 1e-7。
        """
        super().__init__()
        self.cls_loss_gain = cls_loss_gain
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, batch: ClassifyBatchDataInfo):
        """
        计算 Loss。

        Args:
            pred (torch.Tensor): 模型输出的 logits，形状为 (B, C)，其中 B 是批次大小，C 是类别数。
            batch (ClassifyBatchDataInfo): 批数据，包含 `target` 标签，形状为 (B,)，值为类别索引。

        Returns:
            tuple:
                - torch.Tensor: Focal Loss 的标量值。
                - loss_items (dict): 各分量损失，便于日志记录。
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


class YOLOV8DetectionLoss(nn.Module):
    """
    YOLOv8检测损失，包括：
    - 分类损失（cls_loss）—— BCEWithLogitsLoss
    - 框回归损失（box_loss）—— IoU + DFL
    - DFL 损失（dfl_loss）—— Distribution Focal Loss
    使用 Task-Aligned Assigner 进行正负样本匹配。
    """

    def __init__(self, nc, strides, reg_max, imgsz, device, cls_gain=1, box_gain=1, dfl_gain=1, tal_topk=10):
        """
        Args:
            model (nn.Module): 检测模型，用于提取 head 超参数。
            imgsz (int | list[int] | tuple[int, int]): 训练输入分辨率 (W, H)。
            cls_gain (float): 分类损失权重。
            box_gain (float): 框回归损失权重。
            dfl_gain (float): DFL 损失权重。
            tal_topk (int): Task-Aligned Assigner 的 top-k 参数。
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
        将原始 targets 整理为 (B, max_n, 5) 格式，并缩放到输入分辨率。

        Args:
            targets (Tensor): [N, 6] 格式（img_idx, cls, cx, cy, w, h）。
            batch_size (int): 批大小。
            scale_tensor (Tensor): [4] (W, H, W, H) 用于缩放 bbox。

        Returns:
            Tensor: [B, max_n, 5] 格式 (cls, lx, ly, rx, ry)。
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
                n = matches.sum()
                if n:
                    # 把第j张图片的前n个目标设置上对应的目标
                    out[j, :n] = targets[matches, 1:]
            # 这里将bbox格式转成了lxlyrxry格式，并且通过scale_tensor缩放回了训练图大小
            out[..., 1:5] = cxcywh_2_lxlyrxry(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        将锚点 + 分布解码为边界框坐标。

        Args:
            anchor_points (Tensor): [H*W, 2] 锚点中心。
            pred_dist (Tensor): [B, H*W, 4*reg_max] 分布预测。

        Returns:
            Tensor: [B, H*W, 4] (lx, ly, rx, ry)。
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 将最后一个维度归一化，计算ltrb四个坐标的预测值
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds: list[torch.Tensor], batch: DetectBatchDataInfo):
        """
        计算检测三件套损失。

        Args:
            preds (torch.Tensor): 单 head 输出 [B, C+4*reg_max, H*W]。
            batch (DetectBatchDataInfo): 批数据，含 bboxes、labels、bboxes_idx。

        Returns:
            tuple:
                - total_loss (Tensor): 标量。
                - loss_items (dict): 各分量损失。
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
            # 乘以stride_tensor将pred_bboxes恢复原图大小，对anchor_points同理
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        """开始计算loss"""
        cls_loss, box_loss, dfl_loss = torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype)

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
    """Criterion class for computing training losses."""
    OKS_SIGMA = (
            np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
            / 10.0
    )

    def __init__(self, nc, strides, reg_max, imgsz, device, kpt_shape, cls_gain=1, box_gain=1, dfl_gain=1, pose_gain=1, kobj_gain=1, tal_topk=10):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(nc, strides, reg_max, imgsz, device, cls_gain, box_gain, dfl_gain, tal_topk)
        self.pose_gain = pose_gain
        self.kobj_gain = kobj_gain
        self.kpt_shape = kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        humanoid_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(self.OKS_SIGMA).to(self.device) if humanoid_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: tuple[list[torch.Tensor], torch.Tensor], batch: PoseBatchDataInfo):
        """Calculate the total loss and detach it."""

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
        cls_loss = torch.tensor(0., dtype=dtype)
        box_loss = torch.tensor(0., dtype=dtype)
        dfl_loss = torch.tensor(0., dtype=dtype)
        pose_loss = torch.tensor(0., dtype=dtype)
        kobj_loss = torch.tensor(0., dtype=dtype)

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
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
            self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
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
