import torch
import torchvision

from .box_utils import cxcywh_2_lxlyrxry, lxlyrxry_2_cxcywh


def detect_nms(pred: torch.Tensor, conf_threshold=0.25, nms_threshold=0.45, max_detect_num=100, max_nms_num=30000):
    """
    专用于目标检测的非极大值抑制（NMS）。
    使用每类别最大概率作为置信度进行筛选与去重。

    适用场景
    适用于 YOLO/SSD 等单阶段检测器输出格式
    `[batch, num_boxes, 4 + num_classes]`，其中
    - `pred[..., :4]` 为 **中心点宽高 (cx, cy, w, h)** 格式；
    - `pred[..., 4:]` 为各类别概率（已做 softmax / sigmoid）。

    算法步骤
    1. 一次性计算每个预测框在所有类别中的最大概率 `scores_all` 及其类别索引 `cls_all`。
    2. 用 `conf_threshold` 过滤低置信度框。
    3. 逐图片处理：
       a. 坐标由 cxcywh → lxlyrxry（左上角右下角）以计算 IoU。
       b. 若剩余框数 > `max_nms_num`，采用 `torch.topk` 保留置信度最高的前 `max_nms_num` 个框，避免大规模排序。
       c. 调用 `torchvision.ops.batched_nms`（类别感知 NMS）按 IoU 阈值去重。
       d. 取前 `max_detect_num` 个结果，坐标再转回 cxcywh，并组装 6 列输出。
    4. 无框时返回空张量 `[0, 6]`。

    Args:
        pred (torch.Tensor):
            模型预测输出，形状 `[batch, num_boxes, 4 + num_classes]`。
        conf_threshold (float, optional):
            置信度阈值，低于该值的框直接丢弃。默认 0.25。
        nms_threshold (float, optional):
            IoU 阈值，用于 NMS。默认 0.45。
        max_detect_num (int, optional):
            NMS 后单张图片最多保留的目标数。默认 300。
        max_nms_num (int, optional):
            进入 NMS 前最多保留的框数，防止显存/内存溢出。默认 30000。

    Returns:
        list[torch.Tensor]:
            长度为 batch 的列表，每元素为 `[M, 6]`：
            `[cx, cy, w, h, score, cls_id]`。无目标时 `M = 0`。
    """
    B, N, C = pred.shape
    out = [torch.empty((0, 6), device=pred.device, dtype=pred.dtype) for _ in range(B)]

    # 1) 置信度掩码一次性计算
    scores_all, cls_all = pred[..., 4:].max(dim=-1)
    mask = scores_all > conf_threshold

    for i in range(B):
        m = mask[i]  # type: ignore[arg-type]
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
    专用于目标检测的非极大值抑制（NMS），采用模型自带的置信度输出作为筛选依据。

    适用场景：
    适用于 YOLO 等检测器输出的标准格式 [batch, num_boxes, 4 + 1 + num_classes]，
    其中：
    - 前 4 个通道为边界框坐标，格式为 **中心点宽高 (cx, cy, w, h)**；
    - 第 5 个通道（即索引 4）为 **目标存在的置信度（confidence）**；
    - 第 6 个通道起为各类别的 **类别概率**。

    算法流程：
    1. 从所有预测框中提取第 4 通道的置信度，与 `conf_threshold` 比较，生成布尔掩码。
    2. 逐张图片处理：
       a. 根据掩码过滤掉低置信度框，得到剩余框张量 `p`。
       b. 将剩余框的坐标从 **cxcywh** → **lxlyrxry**（左上角右下角）以便计算 IoU。
       c. 提取置信度 `scores` 和类别索引 `cls`（由类别概率 argmax 得到）。
       d. 若剩余框数量超过 `max_nms_num`，使用 `torch.topk` 保留置信度最高的前 `max_nms_num` 个框，避免大规模 NMS 计算。
       e. 调用 `torchvision.ops.batched_nms`，以框坐标、置信度、类别索引为输入，按 IoU 阈值 `nms_threshold` 执行非极大值抑制。
       f. 保留前 `max_detect_num` 个框，转回 **cxcywh** 格式，并组装输出张量。
    3. 若某张图片无有效框，则返回空张量 `[0, 6]`。

    Args:
        pred (torch.Tensor):
            模型预测张量，形状为 `[batch, num_boxes, 4 + 1 + num_classes]`，其中：
            - `pred[..., :4]` 为 `cxcywh` 格式边界框；
            - `pred[..., 4]` 为目标置信度；
            - `pred[..., 5:]` 为各类别概率。
        conf_threshold (float, optional):
            置信度阈值，低于该值的框会被直接丢弃。默认 0.25。
        nms_threshold (float, optional):
            IoU 阈值，用于 NMS 去重。默认 0.45。
        max_detect_num (int, optional):
            NMS 后单张图片最多保留的目标数。默认 300。
        max_nms_num (int, optional):
            进入 NMS 前，根据置信度排序后最多保留的框数，防止显存溢出。默认 30000。

    Returns:
        list[torch.Tensor]:
            长度为 batch 的列表，每个元素为形状 `[M, 6]` 的张量，其中：
            - `M` 为 NMS 后该图片保留的目标数；
            - 6 列依次为 `[cx, cy, w, h, score, cls_id]`，
              坐标为 **cxcywh** 格式，`score` 为最终置信度，`cls_id` 为类别索引（整数）。
            若无目标，则返回空张量 `[0, 6]`。
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


def detect_nms_with_keypoint(pred: torch.Tensor, keypoint_shape: list[int], conf_threshold=0.25, nms_threshold=0.45, max_detect_num=100, max_nms_num=30000):
    """
    带关键点的非极大值抑制（NMS）。
    以“类别最大概率”作为框置信度进行过滤与去重。

    适用模型输出格式
    ----------------
    单阶段检测头，如 YOLO/SSD：
        [batch, num_boxes, 4 + num_classes + keypoint_num×3]
    其中
        pred[..., :4]              -> 中心点+宽高 (cx, cy, w, h)
        pred[..., 4: 4+num_classes] -> 各类别概率（已 softmax/sigmoid）
        pred[..., 4+num_classes: ]  -> 关键点，每个关键点 (x, y, mask) 共 3 列

    算法流程
    --------
    1. 计算每张图每个框的 max 类别得分 scores_all 及对应类别索引 cls_all。
    2. 用 conf_threshold 过滤低置信度框。
    3. 逐张图片：
        a. 框格式由 cxcywh → 左上角右下角 ltrb，便于 IoU 计算。
        b. 若剩余框数 > max_nms_num，仅保留置信度 top-k。
        c. 调用 torchvision.ops.batched_nms 做类别感知 NMS。
        d. 保留最多 max_detect_num 个结果，坐标再转回 cxcywh。
    4. 返回 list[Tensor]，每张图形状 [M, 6 + keypoint_num×3]；
       无目标时 M=0。

    Args
    ----
    pred : torch.Tensor
        模型预测输出，形状 [B, N, 4 + num_classes + keypoint_num×3]。
    keypoint_shape : list[int]
        关键点描述，如 [17, 3] 表示 17 个关键点，每个 3 维 (x,y,mask)。
    conf_threshold : float, optional
        置信度阈值，默认 0.25。
    nms_threshold : float, optional
        IoU 阈值，默认 0.45。
    max_detect_num : int, optional
        单图最多保留框数，默认 100。
    max_nms_num : int, optional
        进入 NMS 前最多框数，防止显存溢出，默认 30000。

    Returns
    ----
    list[torch.Tensor]
        长度 = batch，每个元素形状 [M, 6 + keypoint_num×3]，列顺序：
        [cx, cy, w, h, score, cls_id, x1, y1, mask1, ..., xK, yK, maskK]
        无检测时 M=0。
    """
    B, N, C = pred.shape
    k_num = keypoint_shape[0]  # 17
    total_kpt_cols = k_num * keypoint_shape[1]  # 51
    cls_num = C - total_kpt_cols - 4
    out = [torch.empty((0, 6 + total_kpt_cols), device=pred.device, dtype=pred.dtype) for _ in range(B)]

    # 1) 置信度掩码一次性计算
    scores_all, cls_all = pred[..., 4:4 + cls_num].max(dim=-1)
    mask = scores_all > conf_threshold

    for i in range(B):
        m = mask[i]  # type: ignore[arg-type]
        p = pred[i][m]  # [M, C]
        if p.numel() == 0:
            continue

        boxes_cxcywh = p[:, :4]
        boxes_ltrb = cxcywh_2_lxlyrxry(boxes_cxcywh.clone())
        scores = scores_all[i][m]
        cls_int = cls_all[i][m].to(torch.int64)  # .to(dtype=p.dtype, device=pred.device)
        keypoints = p[:, 4 + cls_num:]

        # 2) topk 代替 sort + slice，避免同步
        if scores.shape[0] > max_nms_num:
            scores, topk = torch.topk(scores, k=max_nms_num)
            boxes_ltrb = boxes_ltrb[topk]
            cls_int = cls_int[topk]
            keypoints = keypoints[topk]

        # 3) 一次性 batch_nms
        keep = torchvision.ops.batched_nms(boxes_ltrb, scores, cls_int, nms_threshold)
        keep = keep[:max_detect_num]

        if keep.shape[0] == 0:
            continue

        # 4) 组装输出：cxcywh 直接来自原 boxes
        res = torch.empty((keep.shape[0], 6 + total_kpt_cols), device=pred.device, dtype=pred.dtype)
        res[:, :4] = lxlyrxry_2_cxcywh(boxes_ltrb[keep])
        res[:, 4] = scores[keep]
        res[:, 5] = cls_int[keep].to(pred.dtype)
        res[:, 6:] = keypoints[keep]
        out[i] = res

    return out


def detect_nms_with_mask(pred: torch.Tensor, nc: int, conf_threshold=0.25, nms_threshold=0.45, max_detect_num=100, max_nms_num=30000):
    """
   带掩码向量的非极大值抑制（NMS）。
   以“类别最大概率”作为框置信度进行过滤与去重。

   适用模型输出格式
   ----------------
   单阶段检测头，如 YOLO-Seg：
       [batch, num_boxes, 4 + num_classes + num_masks]
   其中
       pred[..., :4]              -> 中心点+宽高 (cx, cy, w, h)
       pred[..., 4: 4+num_classes] -> 各类别概率（已 softmax/sigmoid）
       pred[..., 4+num_classes: ]  -> 掩码特征向量，共 num_masks 列

   算法流程
   --------
   与 detect_nms_with_keypoint 相同，仅将关键点列换成掩码列。

   参数
   ----
   pred : torch.Tensor
       模型预测输出，形状 [B, N, 4 + num_classes + num_masks]。
   nc : int
       类别数。
   conf_threshold : float, optional
       置信度阈值，默认 0.25。
   nms_threshold : float, optional
       IoU 阈值，默认 0.45。
   max_detect_num : int, optional
       单图最多保留框数，默认 100。
   max_nms_num : int, optional
       进入 NMS 前最多框数，默认 30000。

   返回
   ----
   list[torch.Tensor]
       长度 = batch，每个元素形状 [M, 6 + num_masks]，列顺序：
       [cx, cy, w, h, score, cls_id, m1, m2, ..., m_num_masks]
       无检测时 M=0。
   """
    B, N, C = pred.shape
    num_masks = C - nc - 4
    out = [torch.empty((0, 6 + num_masks), device=pred.device, dtype=pred.dtype) for _ in range(B)]

    # 1) 置信度掩码一次性计算
    scores_all, cls_all = pred[..., 4:4 + nc].max(dim=-1)
    mask = scores_all > conf_threshold

    for i in range(B):
        m = mask[i]  # type: ignore[arg-type]
        p = pred[i][m]  # [M, C]
        if p.numel() == 0:
            continue

        boxes_cxcywh = p[:, :4]
        boxes_ltrb = cxcywh_2_lxlyrxry(boxes_cxcywh.clone())
        scores = scores_all[i][m]
        cls_int = cls_all[i][m].to(torch.int64)  # .to(dtype=p.dtype, device=pred.device)
        masks = p[:, 4 + nc:]

        # 2) topk 代替 sort + slice，避免同步
        if scores.shape[0] > max_nms_num:
            scores, topk = torch.topk(scores, k=max_nms_num)
            boxes_ltrb = boxes_ltrb[topk]
            cls_int = cls_int[topk]
            masks = masks[topk]

        # 3) 一次性 batch_nms
        keep = torchvision.ops.batched_nms(boxes_ltrb, scores, cls_int, nms_threshold)
        keep = keep[:max_detect_num]

        if keep.shape[0] == 0:
            continue

        # 4) 组装输出：cxcywh 直接来自原 boxes
        res = torch.empty((keep.shape[0], 6 + num_masks), device=pred.device, dtype=pred.dtype)
        res[:, :4] = lxlyrxry_2_cxcywh(boxes_ltrb[keep])
        res[:, 4] = scores[keep]
        res[:, 5] = cls_int[keep].to(pred.dtype)
        res[:, 6:] = masks[keep]
        out[i] = res

    return out
