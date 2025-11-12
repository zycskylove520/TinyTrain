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

from __future__ import annotations

import random
import numpy as np
import cv2

from typing import TYPE_CHECKING, Tuple, Optional

from tinytrain.data.data_format import BaseDataInfo, ImgDataInfo
from tinytrain.utils import LOGGER

# 仅在类型检查阶段导入
if TYPE_CHECKING:
    pass


class DynamicFilling:
    """
    DynamicFilling —— 「保持宽高比 + 动态填充」与「直接拉伸」二选一的数据增强工具

    设计背景
    --------
    在目标检测/关键点/分割任务中，网络通常要求固定尺寸输入。最朴素的办法是「直接拉伸」
    (resize without aspect ratio preservation)，但这会引入几何形变，导致小目标形状失真、
    关键点位移错误。另一种做法是「保持宽高比 + 零填充」，但传统实现只给出图像，不给出
    「如何把原始标注同步映射过去」的数学工具，工程师需要反复手写坐标换算，极易出错。

    本类特点
    --------
    1. 一张图进去，一张图出来，接口极简；同时额外返回一个 **2×3 归一化仿射矩阵 M**，
       把原始归一化坐标映射到目标图归一化坐标，后续 bbox、关键点、mask 都能一次性批量转换。
    2. 采用「概率 p 分支」而非固定策略，可在训练阶段 online 地让网络见到「形变 vs 填充」
       两种分布，提升鲁棒性。
    3. 填充像素值可自定义，默认 114（与 YOLO 系列保持一致），也可换成 (0,0,0) 或其他均值。
    4. 内部实现完全基于 OpenCV，无 PyTorch / NumPy 高级索引，嵌入式部署可直接移植到 C++。

    数学原理
    --------
    设原图像素坐标系为 I，目标像素坐标系为 T；归一化坐标系为 x∈[0,1], y∈[0,1]。
    先缩放后平移的像素级仿射矩阵记为 M_pixel（2×3），则
        [x', y', 1]^T = M_pixel @ [x, y, 1]^T
    把 M_pixel 升维到 3×3 后，归一化矩阵可写成
        M_norm = S_target · M_pixel · S_origin
    其中
        S_origin = diag(w, h, 1)        # 像素 → 归一化
        S_target = diag(1/tw, 1/th, 1)  # 归一化 → 目标像素
    最终取 M = M_norm[:2, :] 作为输出，后续任何点 [x_norm, y_norm, 1] 都可直接左乘 M
    得到目标图归一化坐标。

    使用示例
    --------
    >>> img = cv2.imread('xxx.jpg')
    >>> old_bboxes = np.array([[0.5,0.5,0.2,0.2]])
    >>> df = DynamicFilling((640, 640), p=0.8, fill_value=114)
    >>> img_new, M = df(img)          # img 为 HWC np.uint8
    >>> new_bboxes = df.map_norm_cxcywh(M, old_bboxes)  # 一键同步 bbox

    性能注意
    --------
    1. 由于采用 cv2.INTER_NEAREST 缩放，速度最快，但如果用于分割 mask，
       建议在外部把 `interpolation` 参数暴露成可选项。
    2. 矩阵乘法全部使用 NumPy，单次 forward 额外耗时 < 0.1 ms，可忽略。
    3. 若目标尺寸与原图一致，会短路返回单位矩阵，避免不必要的 copy。

    常见坑点
    --------
    * 归一化坐标系约定：本类输出的 M 直接作用于 **归一化** 坐标，而非像素坐标；
      如果你在外部已经用像素坐标，请自行先把 x/w、y/h 归一化后再传入 static 方法。
    * 当 p=1.0 时，极端长宽比图像可能出现「有效区域过小」问题，建议在外部再做一次
      random crop 或多尺度训练。
    * 填充值默认为灰色 114，若背景颜色与物体相近，可能影响检测头收敛，可视情况改成
      数据集均值或随机填充。
    """

    def __init__(self, target_size: tuple[int, int], p: float = 1.0, fill_value: int = 114):
        """
        Args:
            target_size (Tuple[int, int]):
                (width, height) 目标尺寸，**必须为正整数**；
                通常取 32 的倍数，如 (640, 640)、(768, 1280)。
            p (float):
                选择「保持宽高比 + 填充」策略的概率，范围 [0, 1]。
                p=1.0 表示永远填充；p=0.0 表示永远拉伸。
            fill_value (int):
                填充灰度值，范围 [0, 255]。
                对于三通道图像，会自动复制成 (B, G, R) 三元组；
                单通道图像则直接使用该标量。

        Returns:
            None

        Note:
            构造器只做参数校验与成员变量赋值，**不依赖 OpenCV/NumPy**，
            因此可在主进程里安全地实例化后序列化到子进程。

        Warning:
            target_size 内部顺序为 (W, H)，与 OpenCV 的 (W, H) 保持一致，
            但与 NumPy shape (H, W, C) 相反，使用时切勿混淆。
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got p={p}")
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"target_size must be positive integers, got {target_size}")
        self.target_size = target_size
        self.p = p
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        对单张 HWC 图像执行动态填充/拉伸，并返回图像 + 归一化仿射矩阵。

        Args:
            img (np.ndarray):
                输入图像，维度顺序 HWC，dtype 推荐 np.uint8；
                支持 1/3/4 通道，但通道数在运行期不会变。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img:
                    变换后的图像，尺寸严格等于 `target_size`，dtype 与输入相同。
                M:
                    2×3 归一化仿射矩阵，np.float32；
                    可将原图归一化坐标 [x, y, 1]^T 映射到目标图归一化坐标。

        Note:
            1. 若原图尺寸已与 target_size 一致，则直接短路返回，M 为单位矩阵。
            2. 内部采用 cv2.INTER_NEAREST 缩放，追求速度；如需更高质量，
               可在外部把 interpolation 参数化。

        Warning:
            返回的 M 作用于 **归一化** 坐标，而非像素坐标；
            若你有像素坐标，请先除以 (w, h) 完成归一化再使用本类提供的 static 方法。
        """

        h, w = img.shape[:2]  # 原图尺寸
        tw, th = self.target_size  # 目标尺寸

        # ---------- 短路：尺寸没变 ----------
        if (w, h) == (tw, th):
            # 单位仿射矩阵：归一化→归一化仍是自身
            M = np.array([
                [1., 0., 0.],
                [0., 1., 0.]], dtype=np.float32)
            return img, M

        # ---------- 1. 像素级图像变换 ----------
        if random.random() < self.p:
            out_img, M_pixel = self._pad_keep_ratio(img, w, h)  # 2×3
        else:
            out_img, M_pixel = self._resize_stretch(img, w, h)  # 2×3

        # ---------- 2. 像素矩阵 → 归一化矩阵 ----------
        S_origin = np.array([
            [w, 0, 0],
            [0, h, 0],
            [0, 0, 1]], dtype=np.float32)

        S_target = np.array([
            [1.0 / tw, 0, 0],
            [0, 1.0 / th, 0],
            [0, 0, 1]], dtype=np.float32)
        M_pixel_3x3 = np.vstack([M_pixel, [0, 0, 1]])
        M_norm = S_target @ M_pixel_3x3 @ S_origin
        M = M_norm[:2, :].astype(np.float32)

        # 奇异矩阵告警
        if np.abs(np.linalg.det(M[:, :2])) < 1e-6:
            LOGGER.warning("Near-singular affine matrix detected.")

        return out_img, M

    @staticmethod
    def map_norm_coord(M: np.ndarray, x_norm: float, y_norm: float) -> tuple[float, float]:
        """
        将单个归一化点映射到目标图归一化坐标，**并自动裁剪到 [0,1]**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵，通常由 __call__ 返回。
            x_norm (float):
                原图归一化 x，范围 [0,1]。
            y_norm (float):
                原图归一化 y，范围 [0,1]。

        Returns:
            Tuple[float, float]:
                (x_out, y_out) 映射后坐标，已裁剪到 [0,1]。
        """
        vec = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
        xy_new = (M @ vec).astype(np.float32)
        return float(np.clip(xy_new[0], 0., 1.)), float(np.clip(xy_new[1], 0., 1.))

    @staticmethod
    def map_norm_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        批量映射归一化点数组，支持任意领先维度。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            pts (np.ndarray):
                [N,2] 归一化坐标，最后一维必须为 2。

        Returns:
            np.ndarray:
                与输入相同 shape，最后一维为映射后的 (x, y)，已裁剪到 [0,1]。
        """
        assert pts.ndim == 2 and pts.shape[1] == 2, (f"Points array must be 2-D with shape (N, 2)."
                                                     f" Received shape {pts.shape}.")
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N,3)
        new_xy = (homo @ M.T).astype(np.float32)
        return np.clip(new_xy, 0., 1.)

    @staticmethod
    def map_norm_cxcywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> np.ndarray:
        """
        将 [N,4] 归一化 cxcywh 框映射到目标图，并可按面积过滤。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (cx, cy, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            np.ndarray:
                [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。

        Note:
            面积计算在归一化空间进行，因此阈值 0.01 表示「占图像 1%」。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        cx, cy, w, h = bboxes.T  # 4 个 (N,)
        pts = np.stack([cx, cy, np.ones_like(cx)], axis=0)  # [3, N]
        new_cx, new_cy = (M @ pts)  # [2, N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w = w * sx
        new_h = h * sy

        out = np.stack([new_cx, new_cy, new_w, new_h], axis=1)
        out = np.clip(out, 0.0, 1.0)

        if min_area is not None:
            areas = out[:, 2] * out[:, 3]
            out = out[areas >= min_area]

        return out

    @staticmethod
    def map_norm_lxlyrxry(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> np.ndarray:
        """
        将 [N,4] 归一化对角框 (lx,ly,rx,ry) 映射到目标图。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, rx, ry)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            np.ndarray:
                [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。

        Note:
            面积计算在归一化空间进行，因此阈值 0.01 表示「占图像 1%」。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        flat = bboxes.reshape(-1, 2)
        homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)  # [2N,3]
        new_flat = (homo @ M.T).reshape(-1, 4)  # [N,4]

        # 分别对 x 和 y 做 clip
        xs = new_flat[:, [0, 2]]
        ys = new_flat[:, [1, 3]]
        new_flat[:, [0, 2]] = np.clip(xs, 0., 1.)
        new_flat[:, [1, 3]] = np.clip(ys, 0., 1.)

        if min_area is not None:
            # 计算面积 (rx - lx) * (ry - ly)
            areas = (new_flat[:, 2] - new_flat[:, 0]) * (new_flat[:, 3] - new_flat[:, 1])
            new_flat = new_flat[areas >= min_area]
        return new_flat

    @staticmethod
    def map_norm_lxlywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> np.ndarray:
        """
        将 [N,4] 归一化 (lx,ly,w,h) 框映射到目标图。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            np.ndarray:
                [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。

        Note:
            面积计算在归一化空间进行，因此阈值 0.01 表示「占图像 1%」。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        lx, ly, w, h = bboxes.T
        pts = np.stack([lx, ly, np.ones_like(lx)], axis=0)  # [3, N]
        new_lx, new_ly = (M @ pts)  # [2, N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w = w * sx
        new_h = h * sy

        out = np.stack([new_lx, new_ly, new_w, new_h], axis=1)
        out = np.clip(out, 0.0, 1.0)

        if min_area is not None:
            areas = out[:, 2] * out[:, 3]
            out = out[areas >= min_area]

        return out

    # -------------------------------------------------
    # 内部实现
    # -------------------------------------------------
    def _pad_keep_ratio(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        """
        保持宽高比缩放后居中填充，返回 (out_img, M_pixel)。

        Args:
            img (np.ndarray): 原图 HWC。
            w (int): 原图宽。
            h (int): 原图高。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img: 填充后的图像，尺寸=target_size。
                M_pixel: 2×3 像素级仿射矩阵，先缩放再平移。

        Note:
            1. 缩放系数 scale = min(tw/w, th/h)，保证「最大边刚好贴边」。
            2. 填充量采用「四舍五入到整像素」策略，确保中心对称且总尺寸无误。
        """
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        if new_w == 0 or new_h == 0:
            raise ValueError("缩放后图像边长为 0，请检查输入尺寸与 target_size。")

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
        """
        直接拉伸到 target_size，返回 (out_img, M_pixel)。

        Args:
            img (np.ndarray): 原图 HWC。
            w (int): 原图宽。
            h (int): 原图高。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img: 拉伸后的图像。
                M_pixel: 2×3 像素级仿射矩阵，仅对角线缩放，无平移。

        Note:
            缩放系数 Sx=tw/w, Sy=th/h，二者一般不相等，因此会产生形变。
        """
        tw, th = self.target_size
        stretched_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)

        M = np.array([
            [tw / w, 0, 0],
            [0, th / h, 0]
        ], dtype=np.float32,
        )
        return stretched_img, M


class DynamicScaling:
    """
    DynamicScaling —— 以图像中心为原点、**整图等比例缩放**的数据增强工具。

    设计背景
    --------
    在实际业务中，我们往往希望：
    - **放大**时，网络看到「细节特写」—— 但图像尺寸不能变，于是必须 **中心裁剪**；
    - **缩小**时，网络看到「全局上下文」—— 但图像尺寸不能变，于是必须 **外圈填充**。

    传统做法需要手写两套分支（crop vs pad），还要各自计算 bbox / 关键点偏移，极易出错。
    本类把「缩放 + 裁剪/填充」两步合并为 **一个 2×3 仿射矩阵**，图像与标注一次性同步完成。

    数学原理
    --------
    1. 设原图宽高为 (w, h)，目标尺寸同样为 (w, h)（**本类不改变图像分辨率**）。
    2. 随机采样缩放系数 scale∈[scale_range[0], scale_range[1]]：
       - scale > 1 ：放大 → 先 resize 到 (w*scale, h*scale)，再 **中心裁剪** 回原尺寸；
       - scale < 1 ：缩小 → resize 后尺寸变小，四周 **对称填充** 灰度值回到原尺寸。
    3. 像素级仿射矩阵 M_pixel 仅含「缩放 + 平移」两项：
       - 放大：M_pixel = [[scale, 0, -dx], [0, scale, -dy]]
         其中 dx, dy 为裁剪左上角坐标（负值表示裁剪）。
       - 缩小：M_pixel = [[scale, 0, left], [0, scale, top]]
         其中 left, top 为填充量（正值表示填充）。
    4. 归一化矩阵 M_norm = S · M_pixel · S^{-1}，
       其中 S = diag(1/w, 1/h, 1) 把像素坐标系变换到归一化 [0,1]×[0,1]。
       于是任何归一化点 [x,y,1] 都可直接左乘 M_norm 得到目标图归一化坐标。

    本类仅依赖 OpenCV 与 NumPy，**C++ 端可直接移植**，且返回的 M 矩阵与后续 bbox、关键点

    使用示例
    --------
    >>> img = cv2.imread('xxx.jpg')
    >>> old_bboxes = np.array([[0.5,0.5,0.2,0.2]])
    >>> ds = DynamicScaling(scale_range=(0.5, 2.0), p=0.8, fill_value=114)
    >>> img_new, M = ds(img)          # img 为 HWC np.uint8
    >>> new_bboxes, keep = ds.map_norm_cxcywh(M, old_bboxes, min_area=0.001)
    >>> # keep 为有效框索引，可直接 new_bboxes[keep] 取出
    """

    def __init__(self, scale_range: tuple[float, float] = (1.0, 1.0), p: float = 1.0, fill_value: int = 114):
        """
        Args:
            scale_range (Tuple[float, float]):
                缩放系数上下界，**必须满足 0 < scale_range[0] ≤ scale_range[1]**。
                推荐值：
                - 目标检测：(0.5, 2.0) 可覆盖「全局上下文」与「细节特写」；
                - 人脸识别：(0.7, 1.5) 避免过大变形；
                - 工业质检：(0.8, 1.2) 极端尺度变化少，重视小目标。
            p (float):
                本次增强被触发的概率，范围 [0, 1]。
                p=0 表示完全跳过，返回单位矩阵；p=1 表示必做。
            fill_value (int):
                当 scale < 1 时需要外圈填充的像素值，范围 [0, 255]。
                三通道会自动复制成 (B, G, R) 三元组；单通道直接使用标量。
                默认 114 与 YOLO 系列保持一致，也可换成数据集均值。

        Returns:
            None

        Note:
            构造器只做参数校验与成员变量赋值，**不依赖 OpenCV/NumPy**，
            因此可在主进程里安全地实例化后序列化到子进程。

        Warning:
            scale_range 两端越宽，增强越强，但训练时间也会线性增加；
            在边缘计算设备上部署时，建议把范围缩窄到 (0.8, 1.2) 以内，
            否则极端缩小会导致有效目标过小，检测头召回下降。
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be in [0, 1], got p={p}")
        if not (0.0 < scale_range[0] <= scale_range[1]):
            raise ValueError(f"scale_range must satisfy 0 < min ≤ max, got {scale_range}")
        self.scale_range = scale_range
        self.p = p
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        对单张 HWC 图像执行「中心裁剪 / 外圈填充」式等比例缩放，并返回图像 + 归一化仿射矩阵。

        Args:
            img (np.ndarray):
                输入图像，维度顺序 HWC，dtype 推荐 np.uint8；
                支持 1/3/4 通道，但通道数在运行期不会变。
                **宽高必须与构造时 target_size 一致**（本类不改变分辨率），
                若需要同时改变分辨率，请先使用DynamicFilling类。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img:
                    变换后的图像，尺寸与输入完全相同，dtype 与输入相同。
                M:
                    2×3 归一化仿射矩阵，np.float32；
                    可将原图归一化坐标 [x, y, 1]^T 映射到目标图归一化坐标。

        Note:
            1. 若本次随机跳过（概率 1-p），则返回原图与单位矩阵，无任何拷贝开销。
            2. 内部采用 cv2.INTER_LINEAR 缩放，兼顾速度与质量；
               如需更高质量可改为 cv2.INTER_CUBIC，但耗时翻倍。
            3. 放大分支采用「中心裁剪」，裁剪量始终对称，保证不会引入偏移 bias；
               缩小分支采用「对称填充」，填充量始终居中，保证不会引入偏移 bias。

        Warning:
            返回的 M 作用于 **归一化** 坐标，而非像素坐标；
            若你有像素坐标，请先除以 (w, h) 完成归一化再使用本类提供的 static 方法。
        """
        h, w = img.shape[:2]

        # 短路
        if random.random() > self.p:
            M = np.array([
                [1., 0., 0.],
                [0., 1., 0.]], dtype=np.float32)
            return img, M

        scale = random.uniform(*self.scale_range)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w == 0 or new_h == 0:
            raise ValueError("Scaled image has zero dimension.")

        # 1. 等比例缩放整张图
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 2. 计算中心对齐的偏移
        dx = (new_w - w) // 2
        dy = (new_h - h) // 2

        if scale >= 1.0:  # 放大：中心裁剪，图像尺寸不变
            # 1. 先放大
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 2. 计算裁剪起点（中心对齐）
            dx = (new_w - w) // 2
            dy = (new_h - h) // 2
            out_img = scaled[dy:dy + h, dx:dx + w]  # 裁成原图大小

            # 3. 像素级仿射矩阵：缩放 + 平移（负值）
            M_pixel = np.array([[scale, 0., -dx],
                                [0., scale, -dy]], dtype=np.float32)
        else:  # 缩小：外圈填充
            out_img = cv2.copyMakeBorder(
                scaled,
                top=-dy if dy < 0 else 0,
                bottom=-dy if dy < 0 else 0,
                left=-dx if dx < 0 else 0,
                right=-dx if dx < 0 else 0,
                borderType=cv2.BORDER_CONSTANT,
                value=(self.fill_value,) * img.shape[2] if img.ndim == 3 else self.fill_value,
            )
            # 实际填充量
            top = max(0, -dy)
            left = max(0, -dx)
            M_pixel = np.array([
                [scale, 0., left],
                [0., scale, top]], dtype=np.float32)

        # 3. 把 M_pixel → M_norm
        S = np.array([
            [1 / w, 0, 0],
            [0, 1 / h, 0],
            [0, 0, 1]], dtype=np.float32)

        # 把 2×3 M_pixel 补成 3×3
        M_pixel_3x3 = np.vstack([M_pixel, [0, 0, 1]])
        M_norm = S @ M_pixel_3x3 @ np.linalg.inv(S)

        # 去掉最后一行，保留 2×3 以便后续直接乘 [x,y,1]
        M = M_norm[:2, :].astype(np.float32)

        # 奇异矩阵告警
        if np.abs(np.linalg.det(M[:, :2])) < 1e-6:
            LOGGER.warning("Near-singular affine matrix detected.")

        return out_img, M

    @staticmethod
    def map_norm_coord(M: np.ndarray, x_norm: float, y_norm: float) -> Tuple[Tuple[float, float], bool]:
        """
        将单个归一化点映射到目标图归一化坐标，并返回 **是否落在合法区间** 的标志。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵，通常由 __call__ 返回。
            x_norm (float):
                原图归一化 x，范围 [0,1]。
            y_norm (float):
                原图归一化 y，范围 [0,1]。

        Returns:
            Tuple[Tuple[float, float], bool]:
                (clip_x, clip_y):
                    映射后坐标，已裁剪到 [0,1]。
                valid:
                    若 **原始映射坐标** 已落在 [0,1]×[0,1] 矩形内则为 True，
                    否则为 False（表示该点在目标图外部，需要裁剪）。

        Note:
            本函数适用于「需要知道是否超出边界」的场景，例如：
            - 关键点检测：若 valid=False，可给该点降权或丢弃；
            -  bbox 过滤：若角点全部 valid=False，可直接丢弃该框。
        """
        vec = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
        xy_new = M @ vec
        x_new, y_new = float(xy_new[0]), float(xy_new[1])
        valid = 0. <= x_new <= 1. and 0. <= y_new <= 1.

        clip_x = float(np.clip(x_new, 0., 1.))
        clip_y = float(np.clip(y_new, 0., 1.))
        return (clip_x, clip_y), valid

    @staticmethod
    def map_norm_points(M: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射归一化点数组，并返回 **有效点索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            pts (np.ndarray):
                [N,2] 归一化坐标，最后一维必须为 2。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_pts:
                    [N,2] 映射后坐标，已裁剪到 [0,1]。
                keep_idx:
                    1-D 数组，**完整落在 [0,1] 内**的点的行索引；
                    长度即有效点数量，可直接 new_pts[keep_idx] 取出。

        Note:
            1. 本函数只适用于 **纯缩放 + 平移** 仿射矩阵（即 M[0,1]=M[1,0]=0），
               若含旋转/斜切，请改用更通用的 polygon 裁剪逻辑。
            2. 若你不需要 keep_idx，可直接取 new_pts 使用，所有坐标已保证∈[0,1]。
        """
        assert pts.ndim == 2 and pts.shape[1] == 2, (f"Points array must be 2-D with shape (N, 2)."
                                                     f" Received shape {pts.shape}.")

        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1, dtype=np.float32)
        new_pts = (homo @ M.T).astype(np.float32)

        valid_mask = (
                (new_pts[:, 0] >= 0) & (new_pts[:, 0] <= 1) &
                (new_pts[:, 1] >= 0) & (new_pts[:, 1] <= 1)
        )  # [N]
        new_pts = np.clip(new_pts, 0., 1.)

        keep_idx = np.where(valid_mask)[0]
        return new_pts, keep_idx

    @staticmethod
    def map_norm_cxcywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 cxcywh 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (cx, cy, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引（先几何判有效，再面积过滤）。

        Note:
            1. 几何判有效规则：若裁剪后框与图像仍有交集（x2>0 && y2>0 && x1<1 && y1<1）则保留。
            2. 面积计算在归一化空间进行，因此阈值 0.001 表示「占图像 0.1%」。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        cx, cy, w, h = bboxes.T  # 4×[N]
        pts = np.stack([cx, cy, np.ones_like(cx)], axis=0)  # [3,N]
        new_cx, new_cy = (M @ pts)  # [N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w, new_h = w * sx, h * sy

        x1 = new_cx - new_w / 2
        y1 = new_cy - new_h / 2
        x2 = new_cx + new_w / 2
        y2 = new_cy + new_h / 2

        # 1. 几何判有效（与图像有交集）
        valid_mask = (x2 > 0) & (y2 > 0) & (x1 < 1) & (y1 < 1)

        # 2. 裁剪
        x1_clip = np.clip(x1, 0., 1.)
        y1_clip = np.clip(y1, 0., 1.)
        x2_clip = np.clip(x2, 0., 1.)
        y2_clip = np.clip(y2, 0., 1.)

        new_bboxes = np.stack([(x1_clip + x2_clip) / 2,
                               (y1_clip + y2_clip) / 2,
                               x2_clip - x1_clip,
                               y2_clip - y1_clip], axis=1)

        # 3. 面积过滤
        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            area_mask = areas >= min_area
            valid_mask = valid_mask & area_mask

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx

    @staticmethod
    def map_norm_lxlyrxry(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化对角框 (lx,ly,rx,ry)，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, rx, ry)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先拆成两个角点，分别映射后再重新计算对角线。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        flat = bboxes.reshape(-1, 2)
        homo = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
        new_flat = homo @ M.T  # [2N, 2]

        xs = new_flat[:, 0].reshape(-1, 2)
        ys = new_flat[:, 1].reshape(-1, 2)
        x1, x2 = xs.min(axis=1), xs.max(axis=1)
        y1, y2 = ys.min(axis=1), ys.max(axis=1)

        # 1. 几何判有效
        valid_mask = (x2 > 0) & (y2 > 0) & (x1 < 1) & (y1 < 1)

        # 2. 裁剪
        new_flat = np.clip(new_flat, 0., 1.)
        new_bboxes = new_flat.reshape(-1, 4)

        # 3. 面积过滤
        if min_area is not None:
            areas = (new_bboxes[:, 2] - new_bboxes[:, 0]) * \
                    (new_bboxes[:, 3] - new_bboxes[:, 1])
            area_mask = areas >= min_area
            valid_mask = valid_mask & area_mask

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx

    @staticmethod
    def map_norm_lxlywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 (lx,ly,w,h) 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先映射左上角，再映射宽高（乘以缩放系数）。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        lx, ly, w, h = bboxes.T
        pts = np.stack([lx, ly, np.ones_like(lx)], axis=0)  # [3, N]
        new_lx, new_ly = (M @ pts)  # [N]

        sx, sy = np.abs(M[0, 0]), np.abs(M[1, 1])
        new_w, new_h = w * sx, h * sy

        x1, y1 = new_lx, new_ly
        x2, y2 = new_lx + new_w, new_ly + new_h

        # 1. 几何判有效
        valid_mask = (x2 > 0) & (y2 > 0) & (x1 < 1) & (y1 < 1)

        # 2. 裁剪
        x1_clip = np.clip(x1, 0., 1.)
        y1_clip = np.clip(y1, 0., 1.)
        x2_clip = np.clip(x2, 0., 1.)
        y2_clip = np.clip(y2, 0., 1.)

        new_bboxes = np.stack([x1_clip, y1_clip,
                               x2_clip - x1_clip, y2_clip - y1_clip], axis=1)

        # 3. 面积过滤
        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            area_mask = areas >= min_area
            valid_mask = valid_mask & area_mask

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx


class DynamicRotating:
    """
    DynamicRotating —— 以图像中心为原点、**整图旋转**的数据增强工具。

    设计背景
    --------
    在检测/分割/关键点等任务中，适度引入**平面内旋转**可显著提升模型对「歪斜目标」的鲁棒性。
    传统做法往往直接调用 cv2.warpAffine，但后续还需要手工对齐 bbox、关键点等标注，极易出错。
    本类把「旋转 + 边界填充」两步合并为 **一个 2×3 仿射矩阵**，图像与标注一次性同步完成，
    且返回的归一化矩阵可直接用于 cxcywh、lxlywh、对角框、关键点等格式，无需再算一次坐标偏移。

    数学原理
    --------
    1. 设原图宽高为 (w, h)，目标尺寸同样为 (w, h)（**本类不改变图像分辨率**）。
    2. 随机采样旋转角度 angle∈[angle_range[0], angle_range[1]]，**逆时针为正方向**。
    3. 像素级仿射矩阵 M_pixel 由 cv2.getRotationMatrix2D 生成：
       - 旋转中心 = (w/2, h/2)
       - 缩放系数 = 1.0（只做纯旋转）
    4. 归一化矩阵 M_norm = S · M_pixel · S^{-1}，
       其中 S = diag(1/w, 1/h, 1) 把像素坐标系变换到归一化 [0,1]×[0,1]。
       于是任何归一化点 [x,y,1] 都可直接左乘 M_norm 得到目标图归一化坐标。

    本类仅依赖 OpenCV 与 NumPy，**C++ 端可直接移植**，且返回的 M 矩阵与后续 bbox、关键点
    变换函数完全复用，**无需再写一套旋转逻辑**。

    使用示例
    --------
    >>> img = cv2.imread('xxx.jpg')
    >>> old_bboxes = np.array([[0.5,0.5,0.2,0.2]])
    >>> dr = DynamicRotating(angle_range=(-15, 15), p=0.8, fill_value=114)
    >>> img_new, M = dr(img)          # img 为 HWC np.uint8
    >>> new_bboxes, keep_idx = dr.map_norm_cxcywh(M, old_bboxes, min_area=0.001)
    >>> # keep_idx 为有效框索引，可直接 new_bboxes[keep_idx] 取出
    """

    def __init__(self, angle_range: Tuple[float, float] = (-0., 0.), p: float = 1.0, fill_value: int = 114):
        """
        Args:
            angle_range (Tuple[float, float]):
                角度上下界（单位：度），**必须满足 -180 ≤ min ≤ max ≤ 180**。
                推荐值：
                - 目标检测：(-15, 15) 足以覆盖日常拍摄倾斜；
                - 文档/文字：(-90, 90) 可应对横竖排混排；
                - 工业质检：(-5, 5) 避免过度旋转引入伪影。
                **角度为正则顺时针旋转**，与数学定义相反，但符合用户直觉。
            p (float):
                本次增强被触发的概率，范围 [0, 1]。
                p=0 表示完全跳过，返回单位矩阵；p=1 表示必做。
            fill_value (int):
                旋转后外圈填充的像素值，范围 [0, 255]。
                三通道会自动复制成 (B, G, R) 三元组；单通道直接使用标量。
                默认 114 与 YOLO 系列保持一致，也可换成数据集均值。

        Returns:
            None

        Note:
            构造器只做参数校验与成员变量赋值，**不依赖 OpenCV/NumPy**，
            因此可在主进程里安全地实例化后序列化到子进程。

        Warning:
            angle_range 两端越宽，增强越强，但训练时间也会略微增加；
            在边缘计算设备上部署时，建议把范围缩窄到 (-10, 10) 以内，
            否则极端旋转会导致 bbox 面积急剧缩小，检测头召回下降。
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be ∈ [0, 1], but got p={p}")
        if not (-180. <= angle_range[0] <= angle_range[1] <= 180.):
            raise ValueError(f"angle_range must satisfy -180 ≤ min ≤ max ≤ 180, but got {angle_range}")
        self.angle_range = angle_range
        self.p = p
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单张 HWC 图像执行「中心旋转 + 外圈填充」，并返回图像 + 归一化仿射矩阵。

        Args:
            img (np.ndarray):
                输入图像，维度顺序 HWC，dtype 推荐 np.uint8；
                支持 1/3/4 通道，但通道数在运行期不会变。
                **宽高必须与构造时 target_size 一致**（本类不改变分辨率），
                若需要同时改变分辨率，请先使用 DynamicFilling 类。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img:
                    变换后的图像，尺寸与输入完全相同，dtype 与输入相同。
                M:
                    2×3 归一化仿射矩阵，np.float32；
                    可将原图归一化坐标 [x, y, 1]^T 映射到目标图归一化坐标。

        Note:
            1. 若本次随机跳过（概率 1-p），则返回原图与单位矩阵，无任何拷贝开销。
            2. 内部采用 cv2.INTER_LINEAR 旋转，兼顾速度与质量；
               如需更高质量可改为 cv2.INTER_CUBIC，但耗时翻倍。
            3. 旋转中心始终为图像几何中心，保证不会引入平移 bias；
               外圈填充采用 cv2.BORDER_CONSTANT，像素值由 fill_value 指定。

        Warning:
            返回的 M 作用于 **归一化** 坐标，而非像素坐标；
            若你有像素坐标，请先除以 (w, h) 完成归一化再使用本类提供的 static 方法。
        """

        h, w = img.shape[:2]
        if np.random.rand() > self.p:
            M = np.array([[1., 0., 0.],
                          [0., 1., 0.]], dtype=np.float32)
            return img, M

        # 1. 采样角度（用户定义：正 = 逆时针）
        angle_ccw = np.random.uniform(*self.angle_range)

        # 2. OpenCV 需要“负值”才能逆时针 → 取反
        angle_cv = -angle_ccw

        # 3. 像素级 2×3 矩阵（逆时针旋转）
        M_pixel = cv2.getRotationMatrix2D((w / 2, h / 2), angle_cv, 1.0)

        # 4. 归一化变换矩阵  S * M_px * S^-1
        S = np.array([[1 / w, 0, 0],
                      [0, 1 / h, 0],
                      [0, 0, 1]], dtype=np.float32)
        S_inv = np.linalg.inv(S)
        M_pixel_3x3 = np.vstack([M_pixel, [0, 0, 1]])
        M_norm_3x3 = S @ M_pixel_3x3 @ S_inv
        M = M_norm_3x3[:2, :].astype(np.float32)

        # 5. 图像 warp
        fill = (self.fill_value,) * img.shape[2] if img.ndim == 3 else self.fill_value
        out_img = cv2.warpAffine(img, M_pixel, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=fill)

        # 奇异矩阵告警
        if np.abs(np.linalg.det(M[:, :2])) < 1e-6:
            LOGGER.warning("Near-singular affine matrix detected.")
        return out_img, M

    @staticmethod
    def map_norm_coord(M: np.ndarray, x_norm: float, y_norm: float) -> Tuple[Tuple[float, float], bool]:
        """
        将单个归一化点映射到目标图归一化坐标，并返回 **是否落在合法区间** 的标志。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵，通常由 __call__ 返回。
            x_norm (float):
                原图归一化 x，范围 [0,1]。
            y_norm (float):
                原图归一化 y，范围 [0,1]。

        Returns:
            Tuple[Tuple[float, float], bool]:
                (clip_x, clip_y):
                    映射后坐标，已裁剪到 [0,1]。
                valid:
                    若 **原始映射坐标** 已落在 [0,1]×[0,1] 矩形内则为 True，
                    否则为 False（表示该点在目标图外部，需要裁剪）。

        Note:
            本函数适用于「需要知道是否超出边界」的场景，例如：
            - 关键点检测：若 valid=False，可给该点降权或丢弃；
            - bbox 过滤：若角点全部 valid=False，可直接丢弃该框。
        """
        vec = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
        xy_new = M @ vec
        x_new, y_new = float(xy_new[0]), float(xy_new[1])
        valid = 0. <= x_new <= 1. and 0. <= y_new <= 1.
        x_new = float(np.clip(x_new, 0., 1.))
        y_new = float(np.clip(y_new, 0., 1.))
        return (x_new, y_new), valid

    @staticmethod
    def map_norm_points(M: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射归一化点数组，并返回 **有效点索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            pts (np.ndarray):
                [N,2] 归一化坐标，最后一维必须为 2。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_pts:
                    [N,2] 映射后坐标，已裁剪到 [0,1]。
                keep_idx:
                    1-D 数组，**完整落在 [0,1] 内**的点的行索引；
                    长度即有效点数量，可直接 new_pts[keep_idx] 取出。

        Note:
            1. 本函数只适用于 **纯旋转 + 平移** 仿射矩阵（即无缩放/斜切），
               若含缩放，请改用更通用的 polygon 裁剪逻辑。
            2. 若你不需要 keep_idx，可直接取 new_pts 使用，所有坐标已保证∈[0,1]。
        """
        assert pts.ndim == 2 and pts.shape[1] == 2, (f"Points array must be 2-D with shape (N, 2)."
                                                     f" Received shape {pts.shape}.")
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1, dtype=np.float32)
        new_pts = (homo @ M.T).astype(np.float32)
        valid_mask = ((new_pts[:, 0] >= 0) & (new_pts[:, 0] <= 1) &
                      (new_pts[:, 1] >= 0) & (new_pts[:, 1] <= 1))
        new_pts = np.clip(new_pts, 0., 1.)
        keep_idx = np.where(valid_mask)[0]
        return new_pts, keep_idx

    @staticmethod
    def map_norm_cxcywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 cxcywh 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (cx, cy, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引（先几何判有效，再面积过滤）。

        Note:
            1. 先将原框四顶点分别旋转，再求轴对齐最小外接矩形；
               因此输出框 **不一定与原始框方向相同**，但保证包含全部旋转后顶点。
            2. 几何判有效规则：若裁剪后框与图像仍有交集（x2>0 && y2>0 && x1<1 && y1<1）则保留。
            3. 面积计算在归一化空间进行，因此阈值 0.001 表示「占图像 0.1%」。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        cx, cy, w, h = bboxes.T

        # 1. 原框四顶点（归一化）
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy - h / 2
        x3, y3 = cx + w / 2, cy + h / 2
        x4, y4 = cx - w / 2, cy + h / 2
        quad = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=1).reshape(-1, 2)

        # 2. 仿射变换 → [N,4,2]
        homo = np.concatenate([quad, np.ones((quad.shape[0], 1))], axis=1, dtype=np.float32)
        new_quad = np.array(homo @ M.T)[:, :2].reshape(-1, 4, 2)

        # 原地计算最小外接矩形（仅取轴对齐部分）
        final = np.zeros((new_quad.shape[0], 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            x_min = q[:, 0].min()
            y_min = q[:, 1].min()
            x_max = q[:, 0].max()
            y_max = q[:, 1].max()
            final[i] = ((x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min)

        # 几何有效
        x1n = final[:, 0] - final[:, 2] / 2
        y1n = final[:, 1] - final[:, 3] / 2
        x2n = final[:, 0] + final[:, 2] / 2
        y2n = final[:, 1] + final[:, 3] / 2
        valid_mask = (x2n > 0) & (y2n > 0) & (x1n < 1) & (y1n < 1)

        # 裁剪
        x1c = np.clip(x1n, 0., 1.)
        y1c = np.clip(y1n, 0., 1.)
        x2c = np.clip(x2n, 0., 1.)
        y2c = np.clip(y2n, 0., 1.)
        new_bboxes = np.stack([(x1c + x2c) / 2, (y1c + y2c) / 2,
                               x2c - x1c, y2c - y1c], axis=1)

        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            valid_mask &= areas >= min_area

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx

    @staticmethod
    def map_norm_lxlyrxry(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化对角框 (lx,ly,rx,ry)，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行格式 [xmin, ymin, xmax, ymax]，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [K,4] 映射后的框，格式不变，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先拆成四个角点，分别旋转后再重新计算对角线。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        N = bboxes.shape[0]

        # 1. 还原 4 个角点  [N,4,2]
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        quad = np.stack([
            x1, y1,  # lt
            x2, y1,  # rt
            x2, y2,  # rb
            x1, y2  # lb
        ], axis=1).reshape(-1, 2)  # [4N, 2]

        # 2. 仿射变换
        homo = np.concatenate([quad, np.ones((4 * N, 1))], axis=1, dtype=np.float32)  # [4N, 3]
        new_quad = (homo @ M.T)[:, :2].reshape(N, 4, 2)  # [N,4,2]

        # 3. 最小外接轴对齐框
        new_bboxes = np.zeros((N, 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            xmin = q[:, 0].min()
            ymin = q[:, 1].min()
            xmax = q[:, 0].max()
            ymax = q[:, 1].max()
            new_bboxes[i] = (xmin, ymin, xmax, ymax)

        # 4. 几何有效性
        valid_mask = (new_bboxes[:, 2] > 0) & (new_bboxes[:, 3] > 0) & (new_bboxes[:, 0] < 1) & (new_bboxes[:, 1] < 1)

        # 5. clip 到 [0,1]
        new_bboxes = np.clip(new_bboxes, 0., 1.)

        # 6. 面积过滤
        if min_area is not None:
            areas = (new_bboxes[:, 2] - new_bboxes[:, 0]) * (new_bboxes[:, 3] - new_bboxes[:, 1])
            valid_mask &= areas >= min_area

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx

    @staticmethod
    def map_norm_lxlywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 (lx,ly,w,h) 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out:
                    [M,4] 映射后的框，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先映射左上角，再映射宽高（乘以缩放系数）。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        lx, ly, w, h = bboxes.T

        # 1. 仿射变换
        quad = np.stack([lx, ly, lx + w, ly, lx + w, ly + h, lx, ly + h], axis=1).reshape(-1, 2)
        homo = np.concatenate([quad, np.ones((quad.shape[0], 1))], axis=1, dtype=np.float32)
        new_quad = (homo @ M.T)[:, :2].reshape(-1, 4, 2)

        # 2. 最小外接轴对齐框
        new_bboxes = np.zeros((new_quad.shape[0], 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            x_min = q[:, 0].min()
            y_min = q[:, 1].min()
            x_max = q[:, 0].max()
            y_max = q[:, 1].max()
            new_bboxes[i] = (x_min, y_min, x_max - x_min, y_max - y_min)

        # 3. 几何有效性
        x2n = new_bboxes[:, 0] + new_bboxes[:, 2]
        y2n = new_bboxes[:, 1] + new_bboxes[:, 3]
        valid_mask = (x2n > 0) & (y2n > 0) & (new_bboxes[:, 0] < 1) & (new_bboxes[:, 1] < 1)

        # 4. clip 到 [0,1]
        x1c = np.clip(new_bboxes[:, 0], 0., 1.)
        y1c = np.clip(new_bboxes[:, 1], 0., 1.)
        x2c = np.clip(x2n, 0., 1.)
        y2c = np.clip(y2n, 0., 1.)
        new_bboxes = np.stack([x1c, y1c, x2c - x1c, y2c - y1c], axis=1)

        # 5. 面积过滤
        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            valid_mask &= areas >= min_area

        keep_idx = np.where(valid_mask)[0]
        return new_bboxes, keep_idx


class DynamicShearing:
    """
    DynamicShearing —— 以图像中心为原点、**整图剪切**的数据增强工具。

    设计背景
    --------
    在实际业务中，我们往往希望：
    - 引入**轻微的剪切（shear）**来模拟拍摄角度变化、物体非刚性形变；
    - 但图像尺寸不能变，于是必须 **外圈填充**；
    - 同时，bbox / 关键点要**同步变换**，否则标注会错位。

    传统做法需要手写 shear → warp → 计算偏移，极易出错。
    本类把「剪切 + 填充」两步合并为 **一个 2×3 仿射矩阵**，图像与标注一次性同步完成。

    数学原理
    --------
    1. 设原图宽高为 (w, h)，目标尺寸同样为 (w, h)（**本类不改变图像分辨率**）。
    2. 随机采样剪切系数 sx∈[shear_range[0], shear_range[1]]，
       sy 同理（若希望仅水平剪切，可在构造时把 sy 范围设为 0）。
    3. 像素级仿射矩阵 M_pixel 为：
       [[1, sx, -sx * cy],
        [sy, 1, -sy * cx]]
       其中 (cx, cy) = (w/2, h/2) 为图像中心，保证剪切后中心不动。
    4. 归一化矩阵 M_norm = S · M_pixel · S^{-1}，
       其中 S = diag(1/w, 1/h, 1) 把像素坐标系变换到归一化 [0,1]×[0,1]。
       于是任何归一化点 [x,y,1] 都可直接左乘 M_norm 得到目标图归一化坐标。

    本类仅依赖 OpenCV 与 NumPy，**C++ 端可直接移植**，且返回的 M 矩阵与后续 bbox、关键点
    计算逻辑与 DynamicScaling 完全一致，方便组合使用。

    使用示例
    --------
    >>> img = cv2.imread('xxx.jpg')
    >>> old_bboxes = np.array([[0.5,0.5,0.2,0.2]])
    >>> dshear = DynamicShearing(shear_range=(-0.2, 0.2), p=0.5, fill_value=114)
    >>> img_new, M = dshear(img)          # img 为 HWC np.uint8
    >>> new_bboxes, keep = dshear.map_norm_cxcywh(M, old_bboxes, min_area=0.001)
    >>> # keep 为有效框索引，可直接 new_bboxes[keep] 取出
    """

    def __init__(self, shear_range: Tuple[float, float] = (-0.0, 0.0), p: float = 1.0, fill_value: int = 114):
        """
        Args:
            shear_range (Tuple[float, float]):
                剪切系数上下界，**必须满足 -1 ≤ shear_range[0] ≤ shear_range[1] ≤ 1**。
                推荐值：
                - 目标检测：(-0.2, 0.2) 可模拟轻微视角变化；
                - 人脸识别：(-0.1, 0.1) 避免面部扭曲；
                - 工业质检：(-0.05, 0.05) 几乎无感知，仅增加鲁棒性。
                注：sx=0.2 表示图像顶部向右偏移 20% 宽度。
            p (float):
                本次增强被触发的概率，范围 [0, 1]。
                p=0 表示完全跳过，返回单位矩阵；p=1 表示必做。
            fill_value (int):
                外圈填充的像素值，范围 [0, 255]。
                三通道会自动复制成 (B, G, R) 三元组；单通道直接使用标量。
                默认 114 与 YOLO 系列保持一致，也可换成数据集均值。

        Returns:
            None

        Note:
            构造器只做参数校验与成员变量赋值，**不依赖 OpenCV/NumPy**，
            因此可在主进程里安全地实例化后序列化到子进程。

        Warning:
            shear_range 两端越宽，增强越强，但训练时间也会线性增加；
            在边缘计算设备上部署时，建议把范围缩窄到 (-0.1, 0.1) 以内，
            否则极端剪切会导致目标长宽比失真，检测头召回下降。
        """
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be ∈ [0, 1], but got p={p}")
        if not (-1. <= shear_range[0] <= shear_range[1] <= 1.):
            raise ValueError(f"shear_range must satisfy -1 ≤ min ≤ max ≤ 1, got {shear_range}")
        self.shear_range = shear_range
        self.p = p
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单张 HWC 图像执行「中心剪切 + 外圈填充」，并返回图像 + 归一化仿射矩阵。

        Args:
            img (np.ndarray):
                输入图像，维度顺序 HWC，dtype 推荐 np.uint8；
                支持 1/3/4 通道，但通道数在运行期不会变。
                **宽高必须与构造时 target_size 一致**（本类不改变分辨率），
                若需要同时改变分辨率，请先使用 DynamicFilling 类。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                out_img:
                    变换后的图像，尺寸与输入完全相同，dtype 与输入相同。
                M:
                    2×3 归一化仿射矩阵，np.float32；
                    可将原图归一化坐标 [x, y, 1]^T 映射到目标图归一化坐标。

        Note:
            1. 若本次随机跳过（概率 1-p），则返回原图与单位矩阵，无任何拷贝开销。
            2. 内部采用 cv2.INTER_LINEAR 缩放，兼顾速度与质量；
               如需更高质量可改为 cv2.INTER_CUBIC，但耗时翻倍。
            3. 剪切矩阵以图像中心为原点，保证不会引入偏移 bias；
               外圈填充始终居中，不会引入偏移 bias。

        Warning:
            返回的 M 作用于 **归一化** 坐标，而非像素坐标；
            若你有像素坐标，请先除以 (w, h) 完成归一化再使用本类提供的 static 方法。
        """
        h, w = img.shape[:2]
        if np.random.rand() > self.p:
            M = np.array([[1., 0., 0.],
                          [0., 1., 0.]], dtype=np.float32)
            return img, M

        # 1. 采样剪切系数
        sx = np.random.uniform(*self.shear_range)
        sy = np.random.uniform(*self.shear_range)

        # 2. 构造像素级 2×3 剪切矩阵（中心保持不动）
        cx, cy = w / 2., h / 2.
        M_pixel = np.array([[1., sx, -sx * cy],
                            [sy, 1., -sy * cx]], dtype=np.float32)

        # 3. 归一化矩阵  S * M_px * S^-1
        S = np.array([
            [1 / w, 0, 0],
            [0, 1 / h, 0],
            [0, 0, 1]], dtype=np.float32)
        M_pixel_3x3 = np.vstack([M_pixel, [0, 0, 1]])
        M = (S @ M_pixel_3x3 @ np.linalg.inv(S))[:2, :].astype(np.float32)

        # 4. 图像 warp
        fill = (self.fill_value,) * img.shape[2] if img.ndim == 3 else self.fill_value
        out_img = cv2.warpAffine(img, M_pixel, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=fill)
        return out_img, M

    @staticmethod
    def map_norm_coord(M: np.ndarray, x_norm: float, y_norm: float) -> Tuple[Tuple[float, float], bool]:
        """
        将单个归一化点映射到目标图归一化坐标，并返回 **是否落在合法区间** 的标志。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵，通常由 __call__ 返回。
            x_norm (float):
                原图归一化 x，范围 [0,1]。
            y_norm (float):
                原图归一化 y，范围 [0,1]。

        Returns:
            Tuple[Tuple[float, float], bool]:
                (clip_x, clip_y):
                    映射后坐标，已裁剪到 [0,1]。
                valid:
                    若 **原始映射坐标** 已落在 [0,1]×[0,1] 矩形内则为 True，
                    否则为 False（表示该点在目标图外部，需要裁剪）。

        Note:
            本函数适用于「需要知道是否超出边界」的场景，例如：
            - 关键点检测：若 valid=False，可给该点降权或丢弃；
            -  bbox 过滤：若角点全部 valid=False，可直接丢弃该框。
        """
        vec = np.array([x_norm, y_norm, 1.0], dtype=np.float32)
        xy_new = M @ vec
        x_new, y_new = float(xy_new[0]), float(xy_new[1])
        valid = 0. <= x_new <= 1. and 0. <= y_new <= 1.
        x_new = float(np.clip(x_new, 0., 1.))
        y_new = float(np.clip(y_new, 0., 1.))
        return (x_new, y_new), valid

    @staticmethod
    def map_norm_points(M: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射归一化点数组，并返回 **有效点索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            pts (np.ndarray):
                [N,2] 归一化坐标，最后一维必须为 2。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_pts:
                    [N,2] 映射后坐标，已裁剪到 [0,1]。
                keep_idx:
                    1-D 数组，**完整落在 [0,1] 内**的点的行索引；
                    长度即有效点数量，可直接 new_pts[keep_idx] 取出。

        Note:
            1. 本函数只适用于 **纯剪切 + 平移** 仿射矩阵（即无旋转/斜切），
               若含旋转/斜切，请改用更通用的 polygon 裁剪逻辑。
            2. 若你不需要 keep_idx，可直接取 new_pts 使用，所有坐标已保证∈[0,1]。
        """
        assert pts.ndim == 2 and pts.shape[1] == 2, (f"Points array must be 2-D with shape (N, 2)."
                                                     f" Received shape {pts.shape}.")

        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1, dtype=np.float32)
        new_pts = (homo @ M.T).astype(np.float32)

        valid_mask = ((new_pts[:, 0] >= 0) & (new_pts[:, 0] <= 1) &
                      (new_pts[:, 1] >= 0) & (new_pts[:, 1] <= 1))

        new_pts = np.clip(new_pts, 0., 1.)
        keep_idx = np.where(valid_mask)[0]
        return new_pts, keep_idx

    @staticmethod
    def map_norm_cxcywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 cxcywh 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (cx, cy, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，M ≤ N，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引（先几何判有效，再面积过滤）。

        Note:
            1. 几何判有效规则：若裁剪后框与图像仍有交集（x2>0 && y2>0 && x1<1 && y1<1）则保留。
            2. 面积计算在归一化空间进行，因此阈值 0.001 表示「占图像 0.1%」。
            3. 由于剪切会改变矩形为平行四边形，本函数先拆成 4 个角点分别映射，
               再求新轴对齐包围盒，因此比缩放版略慢。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        cx, cy, w, h = bboxes.T

        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy - h / 2
        x3, y3 = cx + w / 2, cy + h / 2
        x4, y4 = cx - w / 2, cy + h / 2
        quad = np.stack([x1, y1, x2, y2, x3, y3, x4, y4], axis=1).reshape(-1, 2)
        homo = np.concatenate([quad, np.ones((quad.shape[0], 1))], axis=1, dtype=np.float32)
        new_quad = (homo @ M.T)[:, :2].reshape(-1, 4, 2)

        final = np.zeros((new_quad.shape[0], 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            xmin, ymin = q[:, 0].min(), q[:, 1].min()
            xmax, ymax = q[:, 0].max(), q[:, 1].max()
            final[i] = ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin)

        x1n, y1n = final[:, 0] - final[:, 2] / 2, final[:, 1] - final[:, 3] / 2
        x2n, y2n = final[:, 0] + final[:, 2] / 2, final[:, 1] + final[:, 3] / 2
        valid_mask = (x2n > 0) & (y2n > 0) & (x1n < 1) & (y1n < 1)
        x1c, y1c = np.clip(x1n, 0., 1.), np.clip(y1n, 0., 1.)
        x2c, y2c = np.clip(x2n, 0., 1.), np.clip(y2n, 0., 1.)
        new_bboxes = np.stack([(x1c + x2c) / 2, (y1c + y2c) / 2,
                               x2c - x1c, y2c - y1c], axis=1)

        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            valid_mask &= areas >= min_area
        return new_bboxes, np.where(valid_mask)[0]

    @staticmethod
    def map_norm_lxlyrxry(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化对角框 (lx,ly,rx,ry)，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, rx, ry)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先拆成两个角点，分别映射后再重新计算对角线。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        N = bboxes.shape[0]
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        quad = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1).reshape(-1, 2)
        homo = np.concatenate([quad, np.ones((4 * N, 1))], axis=1, dtype=np.float32)
        new_quad = (homo @ M.T)[:, :2].reshape(N, 4, 2)

        new_bboxes = np.zeros((N, 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            new_bboxes[i] = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())

        valid_mask = (new_bboxes[:, 2] > 0) & (new_bboxes[:, 3] > 0) & \
                     (new_bboxes[:, 0] < 1) & (new_bboxes[:, 1] < 1)
        new_bboxes = np.clip(new_bboxes, 0., 1.)

        if min_area is not None:
            areas = (new_bboxes[:, 2] - new_bboxes[:, 0]) * (new_bboxes[:, 3] - new_bboxes[:, 1])
            valid_mask &= areas >= min_area
        return new_bboxes, np.where(valid_mask)[0]

    @staticmethod
    def map_norm_lxlywh(M: np.ndarray, bboxes: np.ndarray, min_area: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量映射 [N,4] 归一化 (lx,ly,w,h) 框，并返回 **有效框索引**。

        Args:
            M (np.ndarray):
                2×3 归一化仿射矩阵。
            bboxes (np.ndarray):
                [N,4] 每行 (lx, ly, w, h)，已归一化。
            min_area (Optional[float]):
                最小保留面积阈值，范围 [0,1]；为 None 时跳过过滤。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                new_bboxes:
                    [M,4] 映射后的框，已裁剪到 [0,1]。
                keep_idx:
                    最终有效框索引。

        Note:
            1. 先映射左上角，再映射宽高（乘以缩放系数）。
            2. 几何判有效规则同上。
        """
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4, (f"Bounding-box array must be 2-D with shape (N, 4)."
                                                           f" Received shape {bboxes.shape}.")
        lx, ly, w, h = bboxes.T

        quad = np.stack([lx, ly, lx + w, ly, lx + w, ly + h, lx, ly + h], axis=1).reshape(-1, 2)
        homo = np.concatenate([quad, np.ones((quad.shape[0], 1))], axis=1, dtype=np.float32)
        new_quad = (homo @ M.T)[:, :2].reshape(-1, 4, 2)

        new_bboxes = np.zeros((new_quad.shape[0], 4), dtype=np.float32)
        for i, q in enumerate(new_quad):  # type: np.ndarray
            xmin, ymin = q[:, 0].min(), q[:, 1].min()
            xmax, ymax = q[:, 0].max(), q[:, 1].max()
            new_bboxes[i] = (xmin, ymin, xmax - xmin, ymax - ymin)

        x2n, y2n = new_bboxes[:, 0] + new_bboxes[:, 2], new_bboxes[:, 1] + new_bboxes[:, 3]
        valid_mask = (x2n > 0) & (y2n > 0) & (new_bboxes[:, 0] < 1) & (new_bboxes[:, 1] < 1)
        x1c, y1c = np.clip(new_bboxes[:, 0], 0., 1.), np.clip(new_bboxes[:, 1], 0., 1.)
        x2c, y2c = np.clip(x2n, 0., 1.), np.clip(y2n, 0., 1.)
        new_bboxes = np.stack([x1c, y1c, x2c - x1c, y2c - y1c], axis=1)

        if min_area is not None:
            areas = new_bboxes[:, 2] * new_bboxes[:, 3]
            valid_mask &= areas >= min_area
        return new_bboxes, np.where(valid_mask)[0]


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
