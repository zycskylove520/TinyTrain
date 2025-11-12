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
