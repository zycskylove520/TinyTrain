from __future__ import annotations

from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.any_utils import make2tuple

from .data_format import ClassifyDataInfo
from tinytrain.data.base.base_augment import BaseAugmentation

if TYPE_CHECKING:
    pass


class ClassificationAugmentation(BaseAugmentation):
    """
    为图像分类任务构建 **训练/验证** 数据增强流水线。

    特性
    ----
    1. **训练增强**：随机裁剪、翻转、颜色抖动、随机擦除、Resize。
    2. **验证增强**：中心 Resize。
    3. **均值/方差 归一化延后**：与检测保持一致，放在 device 端执行。

    使用示例
    --------
    >>> aug = ClassificationAugmentation(cfg)
    >>> train_pipe = aug.augment()
    >>> val_pipe   = aug.transform()
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self._load_cfg()

    def _load_cfg(self) -> None:
        augment_cfg = self.config_manager.augment
        self.target_size = make2tuple(self.config_manager.dataset["img_size"])
        self.mean = augment_cfg["mean"]
        self.std = augment_cfg["std"]
        self.scale = augment_cfg["scale"]
        self.ratio = augment_cfg["ratio"]
        self.hflip = augment_cfg["hflip"]
        self.vflip = augment_cfg["vflip"]
        self.hsv_h = augment_cfg["hsv_h"]
        self.hsv_s = augment_cfg["hsv_s"]
        self.hsv_v = augment_cfg["hsv_v"]
        self.color_jitter = augment_cfg["color_jitter"]
        self.erasing = augment_cfg["erasing"]

    def set_augment(self):
        """
        构建 **训练阶段** 增强流水线。
        """

        import albumentations as A

        self.augment = A.Compose([
            A.HorizontalFlip(p=self.hflip),
            A.VerticalFlip(p=self.vflip),
            A.ColorJitter(brightness=self.hsv_v, contrast=self.hsv_v, saturation=self.hsv_s, hue=self.hsv_h, p=self.color_jitter),
            A.Erasing(p=self.erasing),
            A.RandomResizedCrop(size=self.target_size, scale=self.scale, ratio=self.ratio),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ])

    def set_transform(self):
        """
        构建 **验证/预测阶段** 轻量增强流水线（仅 Resize）。
        """

        import albumentations as A

        self.transform = A.Compose([
            A.Resize(width=self.target_size[0], height=self.target_size[1]),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ])

    def do_augment(self, sample: ClassifyDataInfo):
        assert isinstance(sample, ClassifyDataInfo)
        if self.augment is not None:
            sample.img = self.augment(image=sample.img)['image']
        return sample

    def do_transform(self, sample: ClassifyDataInfo):
        assert isinstance(sample, ClassifyDataInfo)
        if self.transform is not None:
            sample.img = self.transform(image=sample.img)['image']
        return sample
