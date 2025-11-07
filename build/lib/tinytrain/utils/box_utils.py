import math
import torch
import numpy as np


def lxlyrxry_2_lxlywh(x):
    """
    (lx, ly, rx, ry) → (lx, ly, w, h)
    不复制内存，返回与输入同类型空张量/数组。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    lxly = x[..., :2]
    wh = x[..., 2:] - lxly
    y[..., :2] = lxly
    y[..., 2:] = wh
    return y


def lxlywh_2_lxlyrxry(x):
    """
    (lx, ly, w, h) → (lx, ly, rx, ry)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    lxly = x[..., :2]
    rxry = x[..., :2] + x[..., 2:]
    y[..., :2] = lxly
    y[..., 2:] = rxry
    return y


def lxlyrxry_2_cxcywh(x):
    """
    (lx, ly, rx, ry) → (cx, cy, w, h)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    lxly = x[..., :2]
    rxry = x[..., 2:]
    cxcy = (lxly + rxry) / 2
    wh = rxry - lxly
    y[..., :2] = cxcy
    y[..., 2:] = wh
    return y


def cxcywh_2_lxlyrxry(x):
    """
    (cx, cy, w, h) → (lx, ly, rx, ry)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    cxcy = x[..., :2]
    wh = x[..., 2:]
    lxly = cxcy - 0.5 * wh
    rxry = cxcy + 0.5 * wh
    y[..., :2] = lxly
    y[..., 2:] = rxry
    return y


def lxlywh_2_cxcywh(x):
    """
    (lx, ly, w, h) → (cx, cy, w, h)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    lxly = x[..., :2]
    wh = x[..., 2:]
    cxcy = lxly + 0.5 * wh
    y[..., :2] = cxcy
    y[..., 2:] = wh
    return y


def cxcywh_2_lxlywh(x):
    """
    (cx, cy, w, h) → (lx, ly, w, h)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    cxcy = x[..., :2]
    wh = x[..., 2:]
    lxly = cxcy - 0.5 * wh
    y[..., :2] = lxly
    y[..., 2:] = wh
    return y


def box_iou_torch(box1: torch.Tensor, box2: torch.Tensor, mode='IoU', eps=1e-7):
    """
    与原 torchvision.ops.box_iou 接口 100 % 兼容，仅增加 mode 参数
    mode : 'IoU' | 'GIoU' | 'DIoU' | 'CIoU'
    返回 (N,M) 矩阵
    """
    assert mode in {'IoU', 'GIoU', 'DIoU', 'CIoU'}
    box1 = box1.clamp(min=0)
    box2 = box2.clamp(min=0)

    # 与原代码完全一致的张量布局
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)  # (N,M)

    # 面积
    area1 = (a2 - a1).prod(2)  # (N,1)
    area2 = (b2 - b1).prod(2)  # (1,M)
    union = area1 + area2 - inter + eps
    iou = inter / union

    if mode == 'IoU':
        return iou

    # convex 对角线平方
    c_wh = (torch.max(a2, b2) - torch.min(a1, b1)).clamp_min(0)  # (N,M,2)
    c2 = c_wh.pow(2).sum(2) + eps  # (N,M)

    if mode == 'GIoU':
        c_area = c_wh.prod(2) + eps
        return iou - (c_area - union) / c_area

    # center distance
    ctr1 = (a1 + a2) * 0.5  # (N,1,2)
    ctr2 = (b1 + b2) * 0.5  # (1,M,2)
    rho2 = (ctr1 - ctr2).pow(2).sum(2) / 4  # (N,M)

    if mode == 'DIoU':
        return iou - rho2 / c2

    # CIoU
    w1, h1 = (a2 - a1).unbind(-1)  # (N,1)
    w2, h2 = (b2 - b1).unbind(-1)  # (1,M)
    v = (4 / math.pi ** 2) * (w2.div(h2 + eps).atan() - w1.div(h1 + eps).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # (N,M)


def box_iou_numpy(box1: np.ndarray, box2: np.ndarray, mode: str = 'IoU', eps: float = 1e-7):
    """
    NumPy 版 IoU 系列计算，与 box_iou_torch 完全对标。
    输入: box1 (N,4), box2 (M,4)  格式 x1,y1,x2,y2
    返回: (N,M) 矩阵
    mode: 'IoU' | 'GIoU' | 'DIoU' | 'CIoU'
    """
    assert mode in {'IoU', 'GIoU', 'DIoU', 'CIoU'}
    box1 = box1.clip(min=0)
    box2 = box2.clip(min=0)

    # 转成 (N,1,2) 与 (1,M,2) 方便广播
    (a1, a2), (b1, b2) = np.split(box1[:, None, :], 2, axis=2), \
        np.split(box2[None, :, :], 2, axis=2)

    # 交集
    inter_wh = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(min=0)
    inter = inter_wh.prod(axis=2)  # (N,M)

    # 面积
    area1 = (a2 - a1).prod(axis=2)  # (N,1)
    area2 = (b2 - b1).prod(axis=2)  # (1,M)
    union = area1 + area2 - inter + eps
    iou = inter / union

    if mode == 'IoU':
        return iou

    # 最小外接框对角线平方
    c_wh = (np.maximum(a2, b2) - np.minimum(a1, b1)).clip(min=0)
    c2 = (c_wh ** 2).sum(axis=2) + eps  # (N,M)

    if mode == 'GIoU':
        c_area = c_wh.prod(axis=2) + eps
        return iou - (c_area - union) / c_area

    # 中心点距离平方
    ctr1 = (a1 + a2) * 0.5  # (N,1,2)
    ctr2 = (b1 + b2) * 0.5  # (1,M,2)
    rho2 = ((ctr1 - ctr2) ** 2).sum(axis=2)  # (N,M)

    if mode == 'DIoU':
        return iou - rho2 / c2

    # CIoU
    w1, h1 = (a2 - a1)[..., 0], (a2 - a1)[..., 1]  # (N,1)
    w2, h2 = (b2 - b1)[..., 0], (b2 - b1)[..., 1]  # (1,M)
    v = (4 / np.pi ** 2) * (np.arctan(w2 / (h2 + eps)) -
                            np.arctan(w1 / (h1 + eps))) ** 2
    with np.errstate(invalid='ignore'):  # 防止 0/0 警告
        alpha = v / (v - iou + (1 + eps))
    alpha = np.where(np.isfinite(alpha), alpha, 0)  # 兜底
    return iou - (rho2 / c2 + v * alpha)


def box_iou_1v1(box1: torch.Tensor, box2: torch.Tensor, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算 IoU / GIoU / DIoU / CIoU (PyTorch 版)。

    Args:
        box1: (n, 4)
        box2: (n, 4)
        xywh: 若为 True，输入为 (cx, cy, w, h)；否则为 (x1, y1, x2, y2)
        GIoU/DIoU/CIoU: 仅可单选其一
        eps: 数值稳定项

    Returns:
        Tensor: 计算box1和box2逐元素一一对应的iou值，shape为[n,1]，对应指标 (IoU/GIoU/DIoU/CIoU)
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    # b1_x2.minimum(b2_x2):右下角点x值选最小值
    # b1_x1.maximum(b2_x1)：左上角点x值选最大值
    # (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1))：相减就是相交矩形的宽
    # 同理，(b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1))就是相交矩形的高
    # inter：bbox相交的面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    # 并集的面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                           (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
                   ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    """
    根据多尺度特征图生成 anchor 中心点及其 stride。

    Args:
        feats (list[Tensor]): 网络输出特征图列表，每个 shape (B, C, H, W)。
        strides (Tensor): (n_levels,) 各层相对于输入图像的步长。
        grid_cell_offset (float): 网格偏移，默认 0.5 表示中心点。

    Returns:
        Tuple[Tensor, Tensor]:
            anchor_points (Tensor): (total_anchors, 2) 所有 anchor 中心坐标。
            stride_tensor (Tensor): (total_anchors, 1) 每个 anchor 对应的 stride。
    """
    assert feats is not None, "Feature maps (feats) cannot be None"

    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(strides.shape[0]):
        stride: torch.Tensor = strides[i]
        # Get feature map height and width
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))

        # Generate grid cell coordinates
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y

        # Generate meshgrid for x and y coordinates
        sx, sy = torch.meshgrid(sx, sy, indexing="xy")

        # Stack x and y coordinates and reshape to (h*w, 2)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))

        # Create stride tensor for each anchor point
        # Use broadcasting to fill the tensor more efficiently
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))  # type: ignore[arg-type]

    # Concatenate all anchor points and stride tensors
    anchor_points = torch.cat(anchor_points)
    stride_tensor = torch.cat(stride_tensor)

    return anchor_points, stride_tensor
