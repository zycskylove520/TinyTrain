from __future__ import annotations

import random
import numpy as np
import cv2

from typing import TYPE_CHECKING

from tinytrain.utils import LOGGER

from .data_format import BaseDataInfo, ImgDataInfo

# 仅在类型检查阶段导入
if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# 动态填充 / 调整尺寸
# -----------------------------------------------------------------------------
class DynamicFilling:
    """
    纯 OpenCV 实现的动态填充 / 拉伸变换类，仅处理图像.
    """

    def __init__(self, target_size: tuple[int, int], p: float, fill_value: int = 114, ):
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
        """
        对输入样本执行动态填充 / 拉伸变换。

        职责
        ----
        1. 以概率 `p` 决定「保持宽高比 + 填充」或「直接拉伸」。
        2. 更新 `sample.img` 为变换后的图像。
        3. 计算并返回归一化仿射矩阵 `M`，用于后续同步变换标注（如 bbox、关键点等）。

        Args:
            sample (ImgDataInfo): 输入样本，必须包含非空 `img` 字段，且为hwc格式

        Returns:
            tuple[ImgDataInfo, np.ndarray]:
                - 第 0 个元素：已更新的 `ImgDataInfo`，其 `img` 字段被替换为目标尺寸的新图像。
                - 第 1 个元素：形状为 (2, 3) 的 `np.float32` 仿射矩阵 `M`，
                  可将原图归一化坐标映射到目标图归一化坐标，供 `transform_yolo_bboxes_norm` 等函数使用。
        """
        assert isinstance(sample, ImgDataInfo), "sample must be ImgDataInfo"
        assert sample.img is not None, "sample.img is None"
        assert sample.img.size > 0, "Empty image array"

        img = sample.img
        h, w = sample.img.shape[:2]  # 原图尺寸
        tw, th = self.target_size  # 目标尺寸

        # ---------- 短路：尺寸没变 ----------
        if (w, h) == (tw, th):
            # 单位仿射矩阵：归一化→归一化仍是自身
            M = np.array([[1., 0., 0.],
                          [0., 1., 0.]], dtype=np.float32)
            return img, M
        # ------------------------------------

        # ---------------- 1. 像素级图像变换 ----------------
        if random.random() < self.p:
            img, M_pixel = self._pad_keep_ratio(img, w, h)  # 2×3
        else:
            img, M_pixel = self._resize_stretch(img, w, h)  # 2×3

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
        M = M_norm[:2, :].astype(np.float32)
        if np.abs(np.linalg.det(M[:, :2])) < 1e-6:
            LOGGER.warning("Near-singular affine matrix detected.")

        return img, M

    @staticmethod
    def map_norm_coord(M: np.ndarray, x_norm: float, y_norm: float):
        """
        将**单个**归一化点坐标 (x, y) 映射到经过 DynamicFilling 仿射变换后的归一化坐标。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        x_norm : float
            原图中点的归一化 x 坐标，范围通常 [0, 1]
        y_norm : float
            原图中点的归一化 y 坐标，范围通常 [0, 1]

        返回
        ----
        tuple[float, float]
            变换后图像上的归一化 (x_out, y_out)
        """
        vec = np.array([x_norm, y_norm, 1.0], dtype=np.float32)  # 齐次坐标
        xy_new = M @ vec  # 2×3 · 3×1 → 2×1
        return float(np.clip(xy_new[0], 0., 1.)), float(np.clip(xy_new[1], 0., 1.))

    @staticmethod
    def map_norm_cxcywh(M: np.ndarray,cx: float, cy: float, w: float, h: float):
        """
        将**单个**归一化边界框 (cx, cy, w, h) 映射到变换后图像的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        cx, cy : float
            原框中心归一化坐标
        w, h : float
            原框归一化宽高

        返回
        ----
        tuple[float, float, float, float]
            变换后的归一化 (cx_new, cy_new, w_new, h_new)
        """
        # 1. 中心点映射（含平移）
        pt = np.array([cx, cy, 1.0], dtype=np.float32)
        cx_new, cy_new = (M @ pt).tolist()

        # 2. 宽高只受缩放
        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        w_new, h_new = w * sx, h * sy

        # clamp
        cx_new = float(np.clip(cx_new, 0., 1.))
        cy_new = float(np.clip(cy_new, 0., 1.))
        w_new = float(np.clip(w_new, 0., 1.))
        h_new = float(np.clip(h_new, 0., 1.))
        return cx_new, cy_new, w_new, h_new

    @staticmethod
    def map_norm_lxlyrxry(M: np.ndarray,lx: float, ly: float, rx: float, ry: float):
        """
        将**单个**归一化对角框 (lx, ly, rx, ry) 映射到经过 DynamicFilling 仿射变换后的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        lx, ly : float
            原图左上角归一化坐标，范围 [0,1]
        rx, ry : float
            原图右下角归一化坐标，范围 [0,1]

        返回
        ----
        tuple[float, float, float, float]
            变换后的归一化 (new_lx, new_ly, new_rx, new_ry)
        """
        # 两个角点分别映射
        pts = np.array([[lx, ly], [rx, ry]], dtype=np.float32)  # [2,2]
        homo = np.concatenate([pts, np.ones((2, 1))], axis=1)  # [2,3]
        new_pts = homo @ M.T  # [2,2]
        xs, ys = new_pts[:, 0], new_pts[:, 1]
        # clamp
        xs = np.clip(xs, 0., 1.)
        ys = np.clip(ys, 0., 1.)
        return float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1])

    @staticmethod
    def map_norm_lxlywh(M: np.ndarray,lx: float, ly: float, w: float, h: float):
        """
        将**单个**归一化左上角-宽高框 (lx, ly, w, h) 映射到经过 DynamicFilling 仿射变换后的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        lx, ly : float
            原图左上角归一化坐标，范围 [0,1]
        w, h   : float
            原图归一化宽高，范围 [0,1]

        返回
        ----
        tuple[float, float, float, float]
            变换后的归一化 (new_lx, new_ly, new_w, new_h)
        """
        # 左上角点映射
        pt = np.array([lx, ly, 1.0], dtype=np.float32)
        new_lx, new_ly = (M @ pt).tolist()

        # 宽高仅受缩放
        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w, new_h = w * sx, h * sy
        # clamp
        new_lx = float(np.clip(new_lx, 0., 1.))
        new_ly = float(np.clip(new_ly, 0., 1.))
        new_w = float(np.clip(new_w, 0., 1.))
        new_h = float(np.clip(new_h, 0., 1.))
        return new_lx, new_ly, new_w, new_h

    @staticmethod
    def map_norm_points_batched(M: np.ndarray,pts: np.ndarray) -> np.ndarray:
        """
        批量将归一化点坐标数组映射到变换后的归一化坐标。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        pts : np.ndarray
            任意维度，最后一维必须为 2，例如 [N, 2] 或 [N, K, 2]

        返回
        ----
        np.ndarray
            与输入相同形状的数组，最后一维为映射后的 (x, y)
        """
        *lead, two = pts.shape
        assert two == 2
        flat = pts.reshape(-1, 2)  # [N*K, 2]
        homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)  # [N*K, 3]
        new_flat = (homo @ M.T).reshape(*lead, 2)  # 广播乘完再 reshape
        return np.clip(new_flat, 0., 1.)

    @staticmethod
    def map_norm_cxcywh_batched(M: np.ndarray,bboxes: np.ndarray) -> np.ndarray:
        """
        批量将归一化边界框数组 [N, 4] 映射到变换后的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        bboxes : np.ndarray, shape=(N, 4), dtype=np.float32
            每行对应 (cx, cy, w, h) 且已归一化

        返回
        ----
        np.ndarray, shape=(N, 4)
            每行为变换后的归一化 (cx_new, cy_new, w_new, h_new)
        """
        cx, cy, w, h = bboxes.T  # 4 个 (N,)
        pts = np.stack([cx, cy, np.ones_like(cx)], axis=0)  # [3, N]
        new_cx, new_cy = (M @ pts)  # [2, N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w = w * sx
        new_h = h * sy

        out = np.stack([new_cx, new_cy, new_w, new_h], axis=1)
        return np.clip(out, 0., 1.)

    @staticmethod
    def map_norm_lxlyrxry_batched(M: np.ndarray,bboxes: np.ndarray) -> np.ndarray:
        """
        批量将 [N,4] 归一化对角框数组映射到变换后的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        bboxes : np.ndarray, shape=(N, 4), dtype=np.float32
            每行对应 (lx, ly, rx, ry)，已归一化

        返回
        ----
        np.ndarray, shape=(N, 4)
            每行对应变换后的归一化 (new_lx, new_ly, new_rx, new_ry)
        """
        flat = bboxes.reshape(-1, 2)  # [2N, 2]
        homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)  # [2N,3]
        new_flat = homo @ M.T  # [2N,2]
        xs = new_flat[:, [0, 2]]
        ys = new_flat[:, [1, 3]]
        # clamp
        new_flat[:, [0, 2]] = np.clip(xs, 0., 1.)
        new_flat[:, [1, 3]] = np.clip(ys, 0., 1.)
        return new_flat

    @staticmethod
    def map_norm_lxlywh_batched(M: np.ndarray,bboxes: np.ndarray) -> np.ndarray:
        """
        批量将 [N,4] 归一化左上角-宽高框数组映射到变换后的归一化框。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        bboxes : np.ndarray, shape=(N, 4), dtype=np.float32
            每行对应 (lx, ly, w, h)，已归一化

        返回
        ----
        np.ndarray, shape=(N, 4)
            每行对应变换后的归一化 (new_lx, new_ly, new_w, new_h)
        """
        lx, ly, w, h = bboxes.T  # 4 个 (N,)
        pts = np.stack([lx, ly, np.ones_like(lx)], axis=0)  # [3,N]
        new_lx, new_ly = (M @ pts)  # [2,N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w = w * sx
        new_h = h * sy
        out = np.stack([new_lx, new_ly, new_w, new_h], axis=1)
        return np.clip(out, 0., 1.)

    @staticmethod
    def transform_yolo_bboxes_norm(M: np.ndarray,bboxes: np.ndarray) -> np.ndarray:
        """
        将归一化 YOLO 格式边界框（cx,cy,w,h）经过**含旋转/剪切**的仿射矩阵 M 映射到新的归一化框。

        原理
        ----
        1. 将 cxcywh 转成 4 个角点坐标；
        2. 用齐次坐标对 4 个角点批量做仿射变换；
        3. 取变换后角点的外接矩形（轴对齐），再转回 cxcywh。

        因此输出框**一定是轴对齐**的，且可能比原框稍大。

        参数
        ----
        M : np.ndarray
            归一化仿射矩阵，可通过调用DynamicFilling.__call__方法获得
        bboxes : np.ndarray, shape=(N, 4), dtype=np.float32
            每行对应归一化 (cx, cy, w, h)

        返回
        ----
        np.ndarray, shape=(N, 4)
            变换后的归一化 (cx, cy, w, h)
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
        xs, ys = new_corners[..., 0], new_corners[..., 1]
        # 先 clamp 再算外接框
        xs = np.clip(xs, 0., 1.)
        ys = np.clip(ys, 0., 1.)

        x_min = xs.min(axis=1)
        x_max = xs.max(axis=1)
        y_min = ys.min(axis=1)
        y_max = ys.max(axis=1)
        new_bboxes = np.stack([(x_min + x_max) / 2,
                               (y_min + y_max) / 2,
                               x_max - x_min,
                               y_max - y_min], axis=1)
        # 宽高再保险 clamp 一次
        return np.clip(new_bboxes, 0., 1.)

    # -------------------------------------------------
    # 内部实现
    # -------------------------------------------------
    def _pad_keep_ratio(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """保持宽高比，缩放后四周均匀填充"""
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        if new_w == 0 or new_h == 0:
            raise ValueError("Scaled image has zero dimension.")

        # 1. 缩放
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 2. 计算填充量（浮点）
        pad_x = (tw - new_w) / 2  # 左右总留白
        pad_y = (th - new_h) / 2  # 上下总留白

        # 3. 将留白拆成整数边界（保证居中）
        top = int(pad_y + 0.5)  # 四舍五入到整像素
        bottom = th - new_h - top  # 剩余全给下边
        left = int(pad_x + 0.5)
        right = tw - new_w - left

        # 4. 填充
        padded_img = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.fill_value,) * img.shape[2] if img.ndim == 3 else self.fill_value,
        )

        # 4. 构造仿射矩阵 (2×3)：先缩放，再平移
        M = np.array([
            [scale, 0, left],
            [0, scale, top]
        ], dtype=np.float32,
        )
        return padded_img, M

    def _resize_stretch(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """直接拉伸到目标尺寸"""
        tw, th = self.target_size
        stretched_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 仿射矩阵：Sx = tw/w, Sy = th/h
        M = np.array([
            [tw / w, 0, 0],
            [0, th / h, 0]
        ], dtype=np.float32,
        )
        return stretched_img, M


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
