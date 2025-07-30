from __future__ import annotations

import random
import numpy as np

from typing import TYPE_CHECKING

from .data_format import BaseDataInfo, ImgDataInfo, ClassifyDataInfo, DetectDataInfo

# 仅在类型检查阶段导入
if TYPE_CHECKING:
    import albumentations as A
    import cv2


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def normalize(img, mean, std, max_pixel_value=255.) -> np.ndarray:
    """
    将图像像素值归一化到 [0, 1] 后再标准化。

    Args:
        img (np.ndarray): HWC 格式的 uint8 或 float32 图像。
        mean: 每个通道的均值。
        std : 每个通道的标准差。
        max_pixel_value: 原始像素最大值，用于将数据先缩放到 [0, 1]。

    Returns:
        np.ndarray: 归一化并标准化后的图像（float32）。
    """

    import albucore
    denominator = np.reciprocal(np.array(std, dtype=np.float32) * max_pixel_value)
    return albucore.normalize(img, mean, denominator)


# -----------------------------------------------------------------------------
# 动态填充 / 调整尺寸
# -----------------------------------------------------------------------------
class DynamicFilling:
    """
    支持 **保持宽高比 + 填充** 与 **直接拉伸** 两种策略的实时变换类，
    用于分类 / 检测任务。

    策略选择
    --------
    - 概率 `p` 触发 **保持宽高比 + 填充**（PadIfNeeded）。
    - 否则 **直接拉伸**（Resize）。

    注意
    ----
    1. 检测任务会自动同步 bbox 与 label。
    2. 填充值默认 114（YOLO 系列常用灰色）。
    """

    def __init__(
            self,
            target_size: tuple[int, int],
            p: float,
            task: str = "classify",
            fill_value=114,
    ):
        """
        Args:
            target_size (tuple[int, int]): 目标输出尺寸 (width, height)。
            p (float): 0~1 之间，触发“保持宽高比 + 填充”的概率。
            task (str): 任务类型，目前支持 "classify" / "detect"。
            fill_value (int): 填充像素值，默认 114。
        """
        assert 0.0 <= p <= 1.0, "p should be in [0, 1]"
        assert task in {"classify", "detect"}, "only classify / detect supported now"
        self.target_size = target_size  # (w, h)
        self.p = p
        self.task = task
        self.fill_value = fill_value

    def __call__(self, sample: BaseDataInfo) -> BaseDataInfo:
        """
        根据概率选择策略并执行变换，同步更新 sample.current_shape。

        Args:
            sample (BaseDataInfo): 输入样本，需与 task 类型匹配。

        Returns:
            BaseDataInfo: 同一样本实例，字段已更新。
        """
        if random.random() < self.p:
            transform = self._pad_branch()
        else:
            transform = self._resize_branch()

        if self.task == "classify":
            assert isinstance(sample, ClassifyDataInfo)
            out = transform(image=sample.img)
            sample.img = out["image"]

        elif self.task == "detect":
            assert isinstance(sample, DetectDataInfo)
            out = transform(
                image=sample.img,
                bboxes=sample.bboxes,
                class_labels=sample.label,
            )
            sample.img = out["image"]
            sample.bboxes = out["bboxes"]
            sample.label = out["class_labels"]

        else:
            raise NotImplementedError(f"task {self.task} not implemented")

        sample.current_shape = sample.img.shape[:2][::-1]
        return sample

    def _pad_branch(self):
        """构建“保持宽高比 + 填充”的 albumentations 流水线。"""
        import albumentations as A
        import cv2

        tf = [
            A.LongestMaxSize(max_size_hw=(self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.target_size[1],
                min_width=self.target_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill_value,
            ),
        ]
        if self.task == "detect":
            return A.Compose(
                tf,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_area=100,
                    min_visibility=0.1,
                    filter_invalid_bboxes=True,
                ),
                p=1.0,
            )
        else:  # classify
            return A.Compose(tf, p=1.0)

    def _resize_branch(self):
        """构建“直接拉伸”的 albumentations 流水线。"""
        import albumentations as A
        import cv2

        tf = [A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LINEAR)]
        if self.task == "detect":
            return A.Compose(
                tf,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_area=100,
                    min_visibility=0.1,
                    filter_invalid_bboxes=True,
                ),
                p=1.0,
            )
        else:  # classify
            return A.Compose(tf, p=1.0)


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
