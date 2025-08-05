from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from PIL import Image

from .data_format import BaseDataInfo, DetectDataInfo, ClassifyDataInfo, PoseDataInfo

# 仅在类型检查阶段 import，运行时不会加载
if TYPE_CHECKING:
    import torchvision.transforms as T
    import albumentations as A


# base augmentations class -----------------------------------------------------------------------------------------
class BaseAugmentation:
    """
    所有数据增强/变换策略的抽象基类，定义统一接口：
    - augment : 通常返回一个 **离线增强流水线**（一次性生成多图，训练用）。
    - transform : 返回一个 **在线变换流水线**（每次读取样本时实时执行，验证/预测用）。
    子类按需实现其中之一或两者。
    """

    def augment(self, *args, **kwargs):
        """离线增强接口，子类返回 TTCompose 或自定义流水线，默认空实现。"""
        pass

    def transform(self, *args, **kwargs):
        """在线变换接口，子类返回 TTCompose 或自定义流水线，默认空实现。"""
        pass


class TorchvisionAdapter:
    """
    torchvision.transforms → TinyTrain 统一接口的适配器。

    限制：
    1. 仅支持 **像素级** 操作（如 Resize、ColorJitter、Normalize）。
    2. **不处理** bbox、mask、keypoints 等空间标注。

    使用示例
    --------
    >>> tv = TorchvisionAdapter(
    ...     T.Compose([T.Resize(224), T.ToTensor()])
    ... )
    >>> sample = tv(sample)   # sample.img 已更新
    """

    def __init__(self, pixel_transforms: T.Compose = None):
        """
        Args:
            pixel_transforms (T.Compose | None):
                torchvision 变换序列；为 None 时等效恒等映射。
        """
        self.pixel_transforms = pixel_transforms

    def __call__(self, sample: BaseDataInfo):
        """
        对样本图像应用 torchvision 变换，并同步更新 current_shape（W, H）。

        Args:
            sample (BaseDataInfo): 输入样本，img 为 HWC np.uint8。

        Returns:
            BaseDataInfo: 同一样本实例，img 已被替换为 HWC np.float32。
        """
        if self.pixel_transforms is not None:
            image = Image.fromarray(sample.img)
            image = self.pixel_transforms(image)
            sample.img = image.permute(1, 2, 0).numpy()  # CHW->HWC

        sample.current_shape = sample.img.shape[:2][::-1]

        return sample


class AlbumentationsAdapter:
    """
    albumentations → TinyTrain 统一接口的适配器。

    支持任务
    --------
    - "classify" : 仅图像像素增强。
    - "detect"   : 同步增强图像、bboxes、labels，自动过滤无效 bbox。
    """

    def __init__(self, transforms: A.Compose = None, task="detect"):
        """
        Args:
            transforms (A.Compose | None):
                albumentations 流水线；None 时等效恒等映射。
            task (str):
                任务类型，目前仅支持 "classify" 或 "detect"。
        """
        self.transforms = transforms
        self.task = task

    def __call__(self, sample: BaseDataInfo):
        """
        根据 task 调用对应增强函数，并更新 current_shape（W, H）。

        Args:
            sample (BaseDataInfo): 输入样本，类型必须与 task 匹配。

        Returns:
            BaseDataInfo: 同一样本实例，字段已同步增强。
        """
        if self.task == "classify":
            assert isinstance(sample, ClassifyDataInfo), "task=classify 需传入 ClassifyDataInfo"
            self.classify(sample)
        elif self.task == "detect":
            assert isinstance(sample, DetectDataInfo), "task=detect 需传入 DetectDataInfo"
            sample = self.detect(sample)
        elif self.task == "pose":
            assert isinstance(sample, DetectDataInfo), "task=detect 需传入 DetectDataInfo"
            sample = self.detect(sample)

        sample.current_shape = sample.img.shape[:2][::-1]

        return sample

    def classify(self, sample: ClassifyDataInfo):
        """
        对分类任务仅做像素级增强。

        Args:
            sample (ClassifyDataInfo): 输入样本。

        Returns:
            ClassifyDataInfo: 同一样本，字段已同步更新。
        """
        if self.transforms is not None:
            sample.img = self.transforms(image=sample.img)['image']

    def detect(self, sample: DetectDataInfo):
        """
        对检测任务同步增强图像、bboxes、labels。

        Args:
            sample (DetectDataInfo): 输入样本。

        Returns:
            DetectDataInfo: 同一样本，字段已同步更新。
        """
        if self.transforms is not None:
            transformed = self.transforms(image=sample.img, bboxes=sample.bboxes, class_labels=sample.label)
            sample.img = transformed['image']
            sample.bboxes = transformed['bboxes']
            sample.label = np.array(transformed['class_labels'])

        return sample

    def pose(self,sample: PoseDataInfo):
        """
        对姿态估计任务同步增强图像、bboxes、labels。

        Args:
            sample (DetectDataInfo): 输入样本。

        Returns:
            DetectDataInfo: 同一样本，字段已同步更新。
        """
        if self.transforms is not None:
            transformed = self.transforms(image=sample.img, bboxes=sample.bboxes, class_labels=sample.label, keypoints=sample.key_points)
            sample.img = transformed['image']
            sample.bboxes = transformed['bboxes']
            sample.label = np.array(transformed['class_labels'])
            sample.key_points = np.array(transformed['key_points'])

        return sample

class TTCompose:
    """
    将多个增强/变换适配器顺序组合成一条流水线。

    使用示例
    --------
    >>> pipeline = TTCompose([
    ...     DynamicFilling(...),
    ...     AlbumentationsAdapter(...)
    ... ])
    >>> sample = pipeline(sample)
    """
    def __init__(self, transform_adapters: list):
        """
        Args:
            transform_adapters (list[Callable[[BaseDataInfo], BaseDataInfo]]):
                增强/变换适配器列表，按顺序执行。
        """
        self.transform_adapters = transform_adapters

    def __call__(self, sample: BaseDataInfo):
        """
        顺序应用所有适配器。

        Args:
            sample (BaseDataInfo): 输入样本。

        Returns:
            BaseDataInfo: 经过所有适配器处理后的同一样本实例。
        """
        for transform_adapter in self.transform_adapters:
            sample = transform_adapter(sample)
        return sample
