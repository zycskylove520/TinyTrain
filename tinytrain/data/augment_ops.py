from __future__ import annotations

import random
import numpy as np
import cv2

from typing import TYPE_CHECKING

from .data_format import BaseDataInfo, ImgDataInfo

# 仅在类型检查阶段导入
if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# 动态填充 / 调整尺寸
# -----------------------------------------------------------------------------
class DynamicFilling:
    """
    纯 OpenCV 实现的动态填充 / 拉伸变换类，仅处理图像，
    返回的 sample 中附带 2×3 仿射矩阵（self.affine_matrix）。
    """

    def __init__(
            self,
            target_size: tuple[int, int],
            p: float,
            fill_value: int = 114,
    ):
        """
        Args:
            target_size: (width, height) 目标尺寸
            p: 触发“保持宽高比 + 填充”的概率，[0,1]
            fill_value: 填充像素值
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError("p should be in [0, 1]")
        self.target_size = target_size  # (w, h)
        self.p = p
        self.fill_value = fill_value

    def __call__(self, sample: ImgDataInfo):
        assert isinstance(sample, ImgDataInfo), "sample must be ImgDataInfo"
        assert sample.img is not None, "sample.img is None"

        img = sample.img
        h, w = img.shape[:2]  # 原图尺寸
        tw, th = self.target_size  # 目标尺寸

        # ---------------- 1. 像素级图像变换 ----------------
        if random.random() < self.p:
            img, M_pixel = self._pad_keep_ratio(img, w, h)  # 2×3
        else:
            img, M_pixel = self._resize_stretch(img, w, h)  # 2×3

        sample.img = img

        # ---------------- 2. 把 M_pixel → M_norm ----------------
        # 3×3 齐次矩阵：先像素到原图大小 → 仿射变换 → 再归一化到目标图大小
        # 于是：
        # M_norm = S_target * M_pixel_3x3 * S_origin
        S_origin = np.array([[w, 0, 0],
                             [0, h, 0],
                             [0, 0, 1]], dtype=np.float32)

        S_target = np.array([[1.0 / tw, 0, 0],
                             [0, 1.0 / th, 0],
                             [0, 0, 1]], dtype=np.float32)

        # 把 2×3 M_pixel 补成 3×3
        M_pixel_3x3 = np.vstack([M_pixel, [0, 0, 1]])
        M_norm = S_target @ M_pixel_3x3 @ S_origin

        # 去掉最后一行，保留 2×3 以便后续直接乘 [x,y,1]
        M = M_norm[:2, :]
        return sample, M

    # -------------------------------------------------
    # 内部实现
    # -------------------------------------------------
    def _pad_keep_ratio(
            self, img: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """保持宽高比，缩放后四周均匀填充"""
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        # 1. 缩放
        resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # 2. 计算填充
        pad_x = (tw - new_w) / 2  # 左右
        pad_y = (th - new_h) / 2  # 上下

        # 3. 填充
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.fill_value,) * img.shape[2]
            if img.ndim == 3
            else self.fill_value,
        )

        # 4. 构造仿射矩阵 (2×3)：先缩放，再平移
        M = np.array([
            [scale, 0, left],
            [0, scale, top]
        ], dtype=np.float32,
        )
        return padded, M

    def _resize_stretch(
            self, img: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """直接拉伸到目标尺寸"""
        tw, th = self.target_size
        stretched = cv2.resize(
            img, (tw, th), interpolation=cv2.INTER_LINEAR
        )

        # 仿射矩阵：Sx = tw/w, Sy = th/h
        M = np.array([
            [tw / w, 0, 0],
            [0, th / h, 0]
        ], dtype=np.float32,
        )
        return stretched, M

    @staticmethod
    def transform_yolo_bboxes_norm(bboxes: np.ndarray,
                                   M: np.ndarray) -> np.ndarray:
        """
        bboxes : [N,4]  (cx,cy,w,h) 归一化
        M      : [2,3]  DynamicFilling 返回的归一化→归一化仿射矩阵
        return : [N,4]  变换后仍归一化
        """
        # 1. 归一化坐标 → 四角点
        cx, cy, w, h = bboxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # corners: [N,4,2]
        # 先拼成 [N*4, 2]，再 reshape 回 [N,4,2]
        pts = np.column_stack((x1, y1, x2, y1, x2, y2, x1, y2))  # [N, 8]
        corners = pts.reshape(-1, 4, 2)  # [N, 4, 2]

        # 2. 应用仿射矩阵
        ones = np.ones((*corners.shape[:-1], 1))
        corners_h = np.concatenate([corners, ones], axis=-1)  # [N,4,3]
        new_corners = corners_h @ M.T  # [N,4,2]

        # 3. 计算新的外接框
        x_new = new_corners[..., 0]
        y_new = new_corners[..., 1]
        x_min = x_new.min(axis=1)
        y_min = y_new.min(axis=1)
        x_max = x_new.max(axis=1)
        y_max = y_new.max(axis=1)

        # 4. 组装回 yolo 格式
        new_bboxes = np.stack([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            x_max - x_min,
            y_max - y_min
        ], axis=1)
        return new_bboxes


class Mosaic:
    """
    Mosaic 增强：将多张图像拼接成一张大图，用于提升小目标检测性能。

    目前仅预留接口，实际逻辑待实现。
    """

    def __init__(self,
                 task: str = "detect",
                 layout: str = "3x3"
                 ):
        assert task in ["classify", "detect", "segment", "pose"]
        assert layout in ["2x2", "3x3", ]
        self.task = task
        self.layout = layout

    def __call__(self, sample: BaseDataInfo):
        pass

    def mosaic_2x2(self, sample: ImgDataInfo):
        pass

    def mosaic_3x3(self, sample: ImgDataInfo):
        pass
