import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tinytrain.utils.box_utils import lxlyrxry_2_cxcywh


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    # 这是从一系列的点找出一个能把所有点框进去的矩形坐标
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return lxlyrxry_2_cxcywh(np.array(boxes))


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    # 该函数为生成稠密分段点
    for i, s in enumerate(segments):
        # 为每个segment生成闭环
        s = np.concatenate((s, s[0:1, :]), axis=0)
        # 为segment分段生成n个采样点
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        # 对每个采样点进行插值后，生成一个段数为n的segment
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    # mask.shape=(640, 640)
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)  # [[]]  1x2000
    polygons = polygons.reshape((polygons.shape[0], -1, 2))  # 1x1000x2
    cv2.fillPoly(mask, polygons, color=color)
    # nh,nw=160,160
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    # masks。shape=(160, 160)
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    # 这里取负号是为了将argsort默认的从小到大排序转为从大到小对面积排序
    index = np.argsort(-areas)
    # 将mask图像进行从大到小排序
    ms = np.array(ms)[index]
    # 将一张图片里所有的目标的mask图像放进同一张masks图片中
    for i in range(len(segments)):
        # 乘以i+1的意义：
        # 区分不同目标的mask，由于每个mask乘以了i+1，因此在masks中的值就不一样
        # 当多个目标的mask重叠时，将会是面积小的mask重叠在面积大的mask上，这样可以防止小的目标mask被大的完全掩盖
        mask = ms[i] * (i + 1)
        masks = masks + mask
        # 限制在i+1的目的是上面说的掩盖问题
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def decode_pred_masks(proto: torch.Tensor, pred_bboxes: torch.Tensor, pred_masks_vec: torch.Tensor, target_shape, bin=True, retina_masks=False):
    """
    解码出最终实例 mask。

    Args:
        proto:        原型 mask，形状 [C, mh, mw]  (如 32×160×160)
        pred_bboxes:  网络输出的 bbox，已解码到target_shape，形状 [n, 4]  (x1,y1,x2,y2)
        pred_masks_vec:   每个实例对 C 个原型的系数，形状 [n, C]
        target_shape: 最终希望输出的 (W, H) 尺寸（目标图大小）
        retina_masks: 是否使用高分辨率 mask。True→直接上采到原图尺寸再裁；False→在原型尺寸上裁完再上采。

    Returns:
        masks:        二值 mask，形状 [n, H, W]，dtype=torch.bool
    """
    c, mh, mw = proto.shape  # 原型通道数、高、宽
    iw, ih = target_shape  # 目标宽、高

    # 用系数与原型做线性组合，得到 n 张 mask（仍在原型分辨率）
    masks = (pred_masks_vec @ proto.float().view(c, -1)).view(-1, mh, mw)  # [n, mh, mw]

    if not retina_masks:
        # 低分辨率路线：在原型尺寸上裁剪，最后再上采
        width_ratio = mw / iw
        height_ratio = mh / ih

        # pred_boxes放缩到pred_mask尺寸
        downsampled_bboxes = pred_bboxes.clone()
        downsampled_bboxes[:, [0, 2]] *= width_ratio  # x1,x2
        downsampled_bboxes[:, [1, 3]] *= height_ratio  # y1,y2

        masks = crop_mask(masks, downsampled_bboxes)  # CHW
    else:
        # 高分辨率路线：先把原型上采到原图尺寸，再用原图 bbox 裁剪
        # 把原型上采到 (ih, iw)
        proto_up = F.interpolate(proto.unsqueeze(0),
                                 size=(ih, iw),
                                 mode='bilinear',
                                 align_corners=False).squeeze(0)  # [C, ih, iw]

        # 重新组合得到高分辨率 mask
        masks = (pred_masks_vec @ proto_up.view(c, -1)).view(-1, ih, iw)  # [n, ih, iw]

        # 用目标图尺寸的 bbox 裁剪
        abs_bboxes = pred_bboxes.clone()
        masks = crop_mask(masks, abs_bboxes)  # [n, ih, iw]

    return masks.gt_(0.0) if bin else masks
