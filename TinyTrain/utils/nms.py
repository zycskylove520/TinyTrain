import torch
import torchvision

from .box_utils import cxcywh_2_lxlyrxry, lxlyrxry_2_cxcywh


def detect_nms(pred: torch.Tensor, conf_threshold=0.25, nms_threshold=0.45, max_detect_num=300, max_nms_num=30000):
    """
    专用于目标检测的nms，使用类别的概率作为置信度。
    @param pred: 模型预测输出的结果，要求shape:[batch, num_boxes, 4+classes_num],有如下要求：
    1.boxes的format必须为cxcywh格式;
    2.类别值必须转为置信度.
    @param conf_threshold: 置信度阈值
    @param nms_threshold: iou阈值
    @param max_detect_num: nms过滤后最多返回的目标的个数
    @param max_nms_num: 执行nms前，根据置信度排序后只取前max_nms_num个目标进行nms，其余舍弃
    @return: batch个输出的列表，列表每个元素都是nms后的目标，每个元素6列分别为boxes坐标4列,format为cxcywh格式，对应的置信度分数一列，类别id一列
    """
    B, N, C = pred.shape
    out = [torch.empty((0, 6), device=pred.device, dtype=pred.dtype) for _ in range(B)]

    # 1) 置信度掩码一次性计算
    scores_all, cls_all = pred[..., 4:].max(dim=-1)
    mask = scores_all > conf_threshold

    for i in range(B):
        m = mask[i]
        p = pred[i][m]  # [M, C]
        if p.numel() == 0:
            continue

        boxes = p[:, :4].clone()  # 保留 cxcywh 供后续使用
        boxes = cxcywh_2_lxlyrxry(boxes)  # 转 lxlyrxry

        scores = scores_all[i][m]
        cls = cls_all[i][m].to(dtype=p.dtype, device=pred.device)

        # 2) topk 代替 sort + slice，避免同步
        if scores.shape[0] > max_nms_num:
            scores, topk = torch.topk(scores, k=max_nms_num)
            boxes = boxes[topk]
            cls = cls[topk]

        # 3) 一次性 batch_nms
        keep = torchvision.ops.batched_nms(
            boxes, scores, cls.to(torch.int64), iou_threshold=nms_threshold
        )
        keep = keep[:max_detect_num]

        if keep.shape[0] == 0:
            continue

        # 4) 组装输出：cxcywh 直接来自原 boxes
        res = torch.empty((keep.shape[0], 6), device=pred.device, dtype=pred.dtype)
        res[:, :4] = boxes[keep]
        res[:, :4] = lxlyrxry_2_cxcywh(res[:, :4])  # 再转回 cxcywh
        res[:, 4] = scores[keep]
        res[:, 5] = cls[keep]
        out[i] = res

    return out


def detect_nms_with_score(pred: torch.Tensor, conf_threshold=0.25, nms_threshold=0.45, max_detect_num=300, max_nms_num=30000):
    """
    专用于目标检测的nms，自带预测输出的置信度
    @param pred: 模型预测输出的结果，要求shape:[batch, num_boxes, 4+1+classes_num],要求boxes的format必须为cxcywh格式
    @param conf_threshold: 置信度阈值
    @param nms_threshold: iou阈值
    @param max_detect_num: nms过滤后最多返回的目标的个数
    @param max_nms_num: 执行nms前，根据置信度排序后只取前max_nms_num个目标进行nms，其余舍弃
    @return: batch个输出的列表，列表每个元素都是nms后的目标，每个元素6列分别为boxes坐标4列,format为cxcywh格式，对应的置信度分数一列，类别id一列
    """
    B, N, C = pred.shape
    out = [torch.empty((0, 6), device=pred.device, dtype=pred.dtype) for _ in range(B)]

    # 1) 置信度掩码一次性计算
    scores_all = pred[..., 4]
    mask = scores_all > conf_threshold

    for i in range(B):
        m = mask[i]
        p = pred[i][m]  # [M, C]
        if p.numel() == 0:
            continue

        boxes = p[:, :4].clone()  # 保留 cxcywh
        cxcywh_2_lxlyrxry(boxes)  # 就地转 lxlyrxry

        scores = scores_all[i][m]
        cls = p[:, 5:].argmax(dim=1).to(dtype=pred.dtype, device=pred.device)

        # 2) topk 代替 sort + slice
        if scores.shape[0] > max_nms_num:
            scores, topk = torch.topk(scores, k=max_nms_num)
            boxes = boxes[topk]
            cls = cls[topk]

        # 3) batched_nms
        keep = torchvision.ops.batched_nms(
            boxes, scores, cls.to(torch.int64), iou_threshold=nms_threshold
        )
        keep = keep[:max_detect_num]

        if keep.shape[0] == 0:
            continue

        # 4) 组装输出（cxcywh 直接来自原 boxes）
        res = torch.empty((keep.shape[0], 6), device=pred.device, dtype=pred.dtype)
        res[:, :4] = boxes[keep]
        lxlyrxry_2_cxcywh(res[:, :4])  # 转回 cxcywh
        res[:, 4] = scores[keep]
        res[:, 5] = cls[keep]
        out[i] = res

    return out
