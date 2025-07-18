from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from PIL import Image

from .data_format import BaseDataInfo, DetectDataInfo, ClassifyDataInfo

# 仅在类型检查阶段 import，运行时不会加载
if TYPE_CHECKING:
    import torchvision.transforms as T
    import albumentations as A


# base augmentations class -----------------------------------------------------------------------------------------
class BaseAugmentation:
    def augment(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        pass


class TorchvisionAdapter:
    """
    适配torchvision库的适配器类。
    """

    def __init__(self, pixel_transforms: T.Compose = None):
        """
        @param pixel_transforms: torchvision只支持pixel-level变换。
        """
        self.pixel_transforms = pixel_transforms

    def __call__(self, sample: BaseDataInfo):
        if self.pixel_transforms is not None:
            image = Image.fromarray(sample.img)
            image = self.pixel_transforms(image)
            sample.img = image.permute(1, 2, 0).numpy()  # CHW->HWC

        sample.current_shape = sample.img.shape[:2][::-1]

        return sample


class AlbumentationsAdapter:
    """
    适配albumentations库的适配器类。
    """

    def __init__(self, transforms: A.Compose = None, task="detect"):
        self.transforms = transforms
        self.task = task

    def __call__(self, sample: BaseDataInfo):
        if self.task == "classify":
            assert isinstance(sample, ClassifyDataInfo)
            self.classify(sample)
        if self.task == "detect":
            assert isinstance(sample, DetectDataInfo)
            sample = self.detect(sample)

        sample.current_shape = sample.img.shape[:2][::-1]

        return sample

    def classify(self, sample: ClassifyDataInfo):
        if self.transforms is not None:
            sample.img = self.transforms(image=sample.img)['image']

    def detect(self, sample: DetectDataInfo):
        """
        用于目标检测算法的Albumentations增强
        @param sample:
        @return:
        """
        if self.transforms is not None:
            transformed = self.transforms(image=sample.img, bboxes=sample.bboxes, class_labels=sample.label)
            sample.img = transformed['image']
            sample.bboxes = transformed['bboxes']
            sample.label = np.array(transformed['class_labels'])

        return sample


class TTCompose:
    def __init__(self, transform_adapters: list):
        self.transform_adapters = transform_adapters

    def __call__(self, sample: BaseDataInfo):
        for transform_adapter in self.transform_adapters:
            sample = transform_adapter(sample)
        return sample
