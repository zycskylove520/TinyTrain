import albumentations as A

from tinytrain.cfg import TTConfigManager
from tinytrain.data.augment_ops import DynamicFilling
from tinytrain.data.base import TTBaseAugmentation
from tinytrain.models.ocr.ocr_data_format import LPRDataInfo


class LPRAugmentation(TTBaseAugmentation):

    def __init__(self, config_manager: TTConfigManager, target_size: tuple[int, int]):
        super().__init__(config_manager)
        self.target_size = target_size

        self._load_cfg()

    def _load_cfg(self) -> None:
        augment_cfg = self.config_manager.augment
        self.mean = augment_cfg["mean"]
        self.std = augment_cfg["std"]
        self.hsv_h = augment_cfg["hsv_h"]
        self.hsv_s = augment_cfg["hsv_s"]
        self.hsv_v = augment_cfg["hsv_v"]
        self.color_jitter = augment_cfg["color_jitter"]

    def set_augment(self):
        """
        构建 **训练阶段** 增强流水线。
        """
        # Dynamic filling augmentation
        df = DynamicFilling(target_size=self.target_size, p=0.5)

        albumentations_compose = A.Compose([
            A.ColorJitter(brightness=self.hsv_v, contrast=self.hsv_v, saturation=self.hsv_s, hue=self.hsv_h, p=self.color_jitter),
            # 以下两步在移到对应的device后在做,可以提速
            # A.ToFloat(),
            # A.Normalize(mean=self.mean, std=self.std),
        ])

        self.augment = [df, albumentations_compose]

    def set_transform(self):
        """
        构建 **验证/预测阶段** 轻量增强流水线（仅 Resize）。
        """

        # Dynamic filling augmentation
        df = DynamicFilling(target_size=self.target_size, p=0.5)

        self.transform = df

    def do_augment(self, sample: LPRDataInfo):
        if self.augment is None:
            return sample

        assert isinstance(sample, LPRDataInfo)
        df = self.augment[0]
        sample.img, _ = df(sample.img)

        a_compose = self.augment[1]
        sample.img = a_compose(image=sample.img)['image']

        return sample

    def do_transform(self, sample: LPRDataInfo):
        if self.transform is None:
            return sample

        assert isinstance(sample, LPRDataInfo)
        sample.img, _ = self.transform(sample.img)
        return sample
