from __future__ import annotations

from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from .augment_base import BaseAugmentation, TTCompose
from tinytrain.utils.any_utils import make2tuple

if TYPE_CHECKING:
    import albumentations as A
    from .augment_ops import DynamicFilling


# custom augmentations class -----------------------------------------------------------------------------------------

class YOLODetectionAugmentation(BaseAugmentation):
    """
    A class for creating YOLO detection data augmentation pipelines.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._load_cfg()

    def _load_cfg(self) -> None:
        augment_cfg = self.config_manager.augment
        self.target_size = make2tuple(self.config_manager.dataset["img_size"])
        self.mean = augment_cfg["mean"]
        self.std = augment_cfg["std"]
        self.scale = augment_cfg["scale"]
        self.translate = augment_cfg["translate"]
        self.rotate = augment_cfg["rotate"]
        self.shear = augment_cfg["shear"]
        self.hflip = augment_cfg["hflip"]
        self.vflip = augment_cfg["vflip"]
        self.hsv_h = augment_cfg["hsv_h"]
        self.hsv_s = augment_cfg["hsv_s"]
        self.hsv_v = augment_cfg["hsv_v"]
        self.color_jitter = augment_cfg["color_jitter"]

    def augment(self) -> TTCompose:
        """
        Creates a data augmentation pipeline for YOLO detection tasks.

        Returns:
            TTCompose: A composed augmentation pipeline.
        """
        import albumentations as A
        from .augment_base import AlbumentationsAdapter, TTCompose
        from .augment_ops import DynamicFilling

        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5, task="detect")

        albumentations_compose = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.1),
            A.RandomGamma(p=0.1),
            A.ImageCompression(quality_range=(75, 100), p=0.1),
            A.ColorJitter(brightness=self.hsv_v, contrast=self.hsv_v, saturation=self.hsv_s, hue=self.hsv_h, p=self.color_jitter),
            A.HorizontalFlip(p=self.hflip),
            A.VerticalFlip(p=self.vflip),
            A.Affine(scale=self.scale, shear=self.shear, translate_percent=self.translate, rotate=self.rotate),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=100, min_visibility=0.1, filter_invalid_bboxes=True))

        albumentations_adapter = AlbumentationsAdapter(transforms=albumentations_compose, task="detect")

        # Combine the augmentations into a single pipeline
        final_compose = TTCompose([dynamic_filling, albumentations_adapter])
        return final_compose

    def transform(self) -> TTCompose:
        """
        Creates a transformation pipeline for YOLO detection tasks (similar to augment but named differently).

        Returns:
            TTCompose: A composed transformation pipeline.
        """
        import albumentations as A
        from .augment_base import AlbumentationsAdapter, TTCompose
        from .augment_ops import DynamicFilling

        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5, task="detect")

        albumentations_compose = A.Compose([
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=100, min_visibility=0.1, filter_invalid_bboxes=True))

        albumentations_adapter = AlbumentationsAdapter(transforms=albumentations_compose, task="detect")

        # Combine the transformations into a single pipeline
        final_compose = TTCompose([dynamic_filling, albumentations_adapter])
        return final_compose


class ClassificationAugmentation(BaseAugmentation):
    """
    A class for creating image classification data augmentation pipelines.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
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

    def augment(self) -> TTCompose:
        """
        Creates a data augmentation pipeline for image classification tasks.

        Returns:
            TTCompose: A composed augmentation pipeline.
        """
        import albumentations as A
        from .augment_base import AlbumentationsAdapter, TTCompose

        albumentations_pixel_compose = A.Compose([
            A.HorizontalFlip(p=self.hflip),
            A.VerticalFlip(p=self.vflip),
            A.ColorJitter(brightness=self.hsv_v, contrast=self.hsv_v, saturation=self.hsv_s, hue=self.hsv_h, p=self.color_jitter),
            A.Erasing(p=self.erasing),
            A.RandomResizedCrop(size=self.target_size, scale=self.scale, ratio=self.ratio),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ])

        albumentations_adapter = AlbumentationsAdapter(
            transforms=albumentations_pixel_compose,
            task="classify")

        # combine all augmentations
        final_compose = TTCompose([albumentations_adapter])

        return final_compose

    def transform(self) -> TTCompose:
        """
        Creates a transformation pipeline for validation or prediction.

        Returns:
            TTCompose: A composed transformation pipeline.
        """
        import albumentations as A
        from .augment_base import AlbumentationsAdapter, TTCompose

        albumentations_pixel_compose = A.Compose([
            A.Resize(width=self.target_size[0], height=self.target_size[1]),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ])

        albumentations_adapter = AlbumentationsAdapter(
            transforms=albumentations_pixel_compose,
            task="classify")

        # combine all augmentations
        final_compose = TTCompose([albumentations_adapter])

        return final_compose
