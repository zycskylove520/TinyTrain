import cv2
import numpy as np


def align_face_from_landmarks(img: np.ndarray, pred_pts: np.ndarray, gt_pts: np.ndarray, align_size: int | tuple[int, int] = 112):
    """
    归一化坐标 -> 归一化仿射矩阵 -> 一次性乘回目标分辨率
    :param img:        np.ndarray, shape=(H,W,3)
    :param pred_pts:   np.ndarray, shape=(N,2), 归一化坐标 [0,1]
    :param gt_pts:     np.ndarray, shape=(N,2), 归一化坐标 [0,1]
    :param align_size: int, 输出人
    :return: aligned_img
    """
    # 1. 直接用归一化坐标算仿射矩阵（0~1 -> 0~1）
    M_norm, _ = cv2.estimateAffinePartial2D(pred_pts, gt_pts, method=cv2.LMEDS)

    # 2. 把矩阵一次性放大到目标分辨率
    # 只需把输出平移/缩放部分乘 align_size
    M = M_norm.copy()
    w, h = align_size if isinstance(align_size, tuple) else (align_size, align_size)
    M[:, 2] *= [w, h]  # tx* w, ty* h
    M[0, :2] *= w  # 与 x 相关的缩放
    M[1, :2] *= h  # 与 y 相关的缩放

    # 3. warp
    aligned_img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    return aligned_img
