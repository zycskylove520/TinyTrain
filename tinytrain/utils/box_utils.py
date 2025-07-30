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


def bbox_iou_torch(box1: torch.Tensor, box2: torch.Tensor, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算 IoU / GIoU / DIoU / CIoU (PyTorch 版)。

    Args:
        box1: (1, 4) 或 (B, 4)
        box2: (n, 4) 或 (B, n, 4)
        xywh: 若为 True，输入为 (cx, cy, w, h)；否则为 (x1, y1, x2, y2)
        GIoU/DIoU/CIoU: 仅可单选其一
        eps: 数值稳定项

    Returns:
        Tensor: 对应指标 (IoU/GIoU/DIoU/CIoU)
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

def bbox_iou_numpy(box1, box2, eps=1e-7):
    """
    NumPy 版 IoU，输入为 (x1, y1, x2, y2)。

    Args:
        box1: (N, 4)
        box2: (M, 4)
        eps: 数值稳定项

    Returns:
        ndarray: (N, M) IoU
    """
    N = box1.shape[0]
    M = box2.shape[0]

    # reshape 成 (N,1,4) 和 (1,M,4) 以便广播
    b1 = box1.reshape(N, 1, 4)
    b2 = box2.reshape(1, M, 4)

    # 分别取坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[..., 0], b1[..., 1], b1[..., 2], b1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[..., 0], b2[..., 1], b2[..., 2], b2[..., 3]

    # 交集
    inter_w = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0)
    inter_h = np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0)
    inter = inter_w * inter_h

    # 并集
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + eps

    return inter / union

def box_invert_affine_transform(boxes: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """
    将经过仿射变换的框坐标恢复到原图坐标系。

    Args:
        boxes: 变换后的框坐标 (N, 8) 或 (N, 4)。
        affine_matrix: 2×3 仿射矩阵。

    Returns:
        ndarray: 与输入同形状的原始坐标。
    """
    # 将仿射矩阵扩展为 3x3 矩阵
    affine_matrix_homo = np.vstack([affine_matrix, [0, 0, 1]])

    # 求逆矩阵
    inv_matrix_homo = np.linalg.inv(affine_matrix_homo)
    inv_matrix = inv_matrix_homo[:2, :]

    # 将 box 坐标转换为 (N*4, 2) 形状
    num_boxes = boxes.shape[0]
    corners = boxes.reshape(-1, 2)  # (N*4, 2)

    # 转换为齐次坐标 (N*4, 3)
    ones = np.ones((corners.shape[0], 1))
    corners_homo = np.hstack([corners, ones])

    # 应用逆变换矩阵
    original_corners = (inv_matrix @ corners_homo.T).T

    # 恢复为原始 box 形状 (N, 4)
    original_boxes = original_corners.reshape(num_boxes, -1)

    return original_boxes
