import torch

from torch import nn

from tinytrain.data.data_format import ClassifyBatchDataInfo, DetectBatchDataInfo
from tinytrain.loss.subloss import BboxLossWithDFL
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry
from tinytrain.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox


class ClassificationLoss(nn.Module):
    """
    适用于所有分类任务的通用分类损失函数。
    """

    def __init__(self, cls_loss_gain: float):
        super().__init__()
        self.cls_loss_gain = cls_loss_gain
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds: list[torch.Tensor], batch: ClassifyBatchDataInfo):
        """Compute the classification loss between predictions and true target."""
        loss = self.criterion(preds[0], batch.target) * self.cls_loss_gain
        loss_items = {"cls_loss": loss.detach()}
        return loss, loss_items


class YOLOV8DetectionLoss(nn.Module):
    """
    该loss由cls_loss、box_loss、dfl_loss组成。
    """

    def __init__(self, model, imgsz, cls_gain=1, box_gain=1, dfl_gain=1, tal_topk=10):
        """
        @param nc: 类别个数
        @param tal_topk:
        """
        super().__init__()
        self.device = next(model.parameters()).device  # get model device
        head = model.module_list[-1]
        self.stride = head.stride
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.output_num = self.nc + self.reg_max * 4
        self.imgsz = list(imgsz) if isinstance(imgsz, (list,tuple)) else [imgsz, imgsz]  # w,h
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain

        self.use_dfl = self.reg_max > 1

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLossWithDFL(self.reg_max)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
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
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 将最后一个维度归一化，计算ltrb四个坐标的预测值
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, preds: list[torch.Tensor], batch: DetectBatchDataInfo):
        preds = preds[0]  # 只有一个head module
        dtype = preds[0].dtype
        batch_size = preds[0].shape[0]

        # 分离类别和boxes
        pred_distri, pred_scores = torch.cat([pred.view(batch_size, self.output_num, -1) for pred in preds], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)

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
