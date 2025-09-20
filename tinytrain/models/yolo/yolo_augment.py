from __future__ import annotations

import numpy as np
import albumentations as A

from tinytrain.cfg import ConfigManager
from tinytrain.data.data_format import DetectDataInfo, PoseDataInfo
from tinytrain.data.base import BaseAugmentation
from tinytrain.utils.any_utils import make2tuple
from tinytrain.data.augment_ops import DynamicFilling


class YOLODetectionAugmentation(BaseAugmentation):
    """
    为 YOLO 系列检测任务构建 **训练/验证** 数据增强流水线。

    特性
    ----
    1. **动态填充**（DynamicFilling）：先随机填充到统一尺寸，再进入 Albumentations。
    2. **Albumentations 全量增强**：模糊、灰度、亮度对比、HSV、翻转、仿射、压缩等。
    3. **Yolo 格式 bbox 同步**：Albumentations 自动保证 bbox 与标签同步变换，并过滤无效框。
    4. **均值/方差 归一化延后**：为了加快 CPU→GPU 传输，归一化放在 device 端执行。

    使用示例
    --------
    >>> aug = YOLODetectionAugmentation(cfg)
    >>> pipeline = aug.augment()          # 训练增强
    >>> val_pipeline = aug.transform()    # 验证增强
    >>> sample = pipeline(sample)
    """

    def __init__(self, config_manager: ConfigManager):
        """
        从配置文件中读取增强超参数。

        Args:
            config_manager (ConfigManager): 包含 augment / dataset 配置段。
        """
        super().__init__(config_manager)
        self._load_cfg()

    def _load_cfg(self) -> None:
        """
        一次性把增强相关配置读入成员变量，方便后续构造 pipeline。
        """
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

    def set_augment(self):
        """
        构建 **训练阶段** 增强流水线。
        """
        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5)

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
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=100, min_visibility=0.1, filter_invalid_bboxes=True))

        self.augment = [dynamic_filling, albumentations_compose]

    def set_transform(self):
        """
        构建 **验证/预测阶段** 轻量增强流水线（仅 resize 与 bbox 同步）。
        """
        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5)

        albumentations_compose = A.Compose([
            # 以下两步在移到对应的device后在做,可以提速
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.ToTensorV2(),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=100, min_visibility=0.1, filter_invalid_bboxes=True))

        # Combine the transformations into a single pipeline
        self.transform = [dynamic_filling, albumentations_compose]

    def do_augment(self, sample: DetectDataInfo):
        if self.augment is None:
            return sample

        assert isinstance(sample, DetectDataInfo)
        df = self.augment[0]
        df(sample)
        sample.bboxes = df.transform_yolo_bboxes_norm(sample.bboxes)

        a_compose = self.augment[1]
        transformed = a_compose(image=sample.img, bboxes=sample.bboxes, class_labels=sample.label)
        sample.img = transformed['image']
        sample.bboxes = transformed['bboxes']
        sample.label = np.array(transformed['class_labels'])
        return sample

    def do_transform(self, sample: DetectDataInfo):
        if self.transform is None:
            return sample

        assert isinstance(sample, DetectDataInfo)
        df = self.transform[0]
        df(sample)
        sample.bboxes = df.transform_yolo_bboxes_norm(sample.bboxes)

        a_compose = self.transform[1]
        transformed = a_compose(image=sample.img, bboxes=sample.bboxes, class_labels=sample.label)
        sample.img = transformed['image']
        sample.bboxes = transformed['bboxes']
        sample.label = np.array(transformed['class_labels'])
        return sample


class YOLOPoseAugmentation(BaseAugmentation):
    """
    为 YOLO 系列检测任务构建 **训练/验证** 数据增强流水线。

    特性
    ----
    1. **动态填充**（DynamicFilling）：先随机填充到统一尺寸，再进入 Albumentations。
    2. **Albumentations 全量增强**：模糊、灰度、亮度对比、HSV、翻转、仿射、压缩等。
    3. **Yolo 格式 bbox 同步**：Albumentations 自动保证 bbox 与标签同步变换，并过滤无效框。
    4. **均值/方差 归一化延后**：为了加快 CPU→GPU 传输，归一化放在 device 端执行。

    使用示例
    --------
    >>> aug = YOLOPoseAugmentation(cfg)
    >>> pipeline = aug.augment()          # 训练增强
    >>> val_pipeline = aug.transform()    # 验证增强
    >>> sample = pipeline(sample)
    """

    def __init__(self, config_manager: ConfigManager):
        """
        从配置文件中读取增强超参数。

        Args:
            config_manager (ConfigManager): 包含 augment / dataset 配置段。
        """
        super().__init__(config_manager)
        self._load_cfg()

    def _load_cfg(self) -> None:
        """
        一次性把增强相关配置读入成员变量，方便后续构造 pipeline。
        """
        augment_cfg = self.config_manager.augment
        self.target_size = make2tuple(self.config_manager.dataset["img_size"])
        self.mean = augment_cfg["mean"]
        self.std = augment_cfg["std"]
        self.hflip = augment_cfg["hflip"]
        self.vflip = augment_cfg["vflip"]
        self.hsv_h = augment_cfg["hsv_h"]
        self.hsv_s = augment_cfg["hsv_s"]
        self.hsv_v = augment_cfg["hsv_v"]
        self.color_jitter = augment_cfg["color_jitter"]

    def set_augment(self):
        """
        构建 **训练阶段** 增强流水线。

        Returns:
            TTCompose: 组合了 DynamicFilling + Albumentations 的增强器。
        """

        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5)

        albumentations_compose = A.Compose([
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.1),
            A.RandomGamma(p=0.1),
            A.ImageCompression(quality_range=(75, 100), p=0.1),
            A.ColorJitter(brightness=self.hsv_v, contrast=self.hsv_v, saturation=self.hsv_s, hue=self.hsv_h, p=self.color_jitter),
            # 以下两步在移到对应的device后在做,可以提速
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.ToTensorV2(),
        ])

        self.augment = [dynamic_filling, albumentations_compose]

    def set_transform(self):
        """
        构建 **验证/预测阶段** 轻量增强流水线（仅 resize 与 bbox 同步）。

        Returns:
            TTCompose: 组合了 DynamicFilling + 轻量 Albumentations 的增强器。
        """

        # Dynamic filling augmentation
        dynamic_filling = DynamicFilling(target_size=self.target_size, p=0.5)

        albumentations_compose = A.Compose([
            # 以下两步在移到对应的device后在做,可以提速
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # A.ToTensorV2(),
        ])
        self.augment = [dynamic_filling, albumentations_compose]

    def do_augment(self, sample: PoseDataInfo):
        assert isinstance(sample, PoseDataInfo)
        if self.augment is None:
            return sample

        # dynamic resize
        df = self.augment[0]
        df(sample)
        sample.bboxes = df.transform_yolo_bboxes_norm(sample.bboxes)
        sample.keypoints[..., :2] = df.map_norm_points_batched(sample.keypoints[..., :2])

        a_compose = self.augment[1]
        transformed = a_compose(image=sample.img)
        sample.img = transformed['image']

        return sample

    def do_transform(self, sample: PoseDataInfo):
        assert isinstance(sample, PoseDataInfo)
        if self.augment is None:
            return sample

        df = self.augment[0]
        df(sample)
        sample.bboxes = df.transform_yolo_bboxes_norm(sample.bboxes)
        sample.keypoints[..., :2] = df.map_norm_points_batched(sample.keypoints[..., :2])
        return sample
