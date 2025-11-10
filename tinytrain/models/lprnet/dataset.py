import cv2
import numpy as np
import torch

from pathlib import Path

from tinytrain.data.dataset import TTYOLOVisionDataset
from tinytrain.models.lprnet.data_format import LPRDataInfo, LPRBatchDataInfo
from tinytrain.utils.data_utils import cv_imread, load_image_cache_file


class LPRNetDataset(TTYOLOVisionDataset):
    def __init__(self, config_manager, img_path: Path | list[Path], mode: str = "train"):
        super().__init__(config_manager, img_path, mode)
        self.lpr_max_len = self.config_manager.dataset["lpr_max_len"]
        self.rgb: bool = self.config_manager.augment["rgb"]
        self.chars_dict = {char: int(idx) for idx, char in self.config_manager.dataset["names"].items()}

        self.plates = self.prepare_data()

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, index):
        # 车牌识别只考虑有标签对应的图片
        if self.cache:
            img = load_image_cache_file(self.npy_files[index])  # BGR 走内存映射（省 RAM + 多进程共享）
        else:
            img = cv_imread(self.img_files[index])  # BGR  读取图片存在IO瓶颈

        # Convert color format
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        origin_shape = img.shape[:2][::-1]  # w,h

        # transform
        if origin_shape[0] != self.img_size[0] or origin_shape[1] != self.img_size[1]:
            img = cv2.resize(img, self.img_size)
        img = self.transforms(img)

        plate_name = self.plates[index]
        label = list()
        for c in plate_name:
            label.append(self.chars_dict[c])

        # 新能源车牌规则检查
        if len(label) == 8:
            assert self.check_relu(label), f"Error plate rule:{plate_name}."

        if len(label) > self.lpr_max_len:
            raise ValueError(f"Plate {plate_name} length={len(label)} exceeds max={self.lpr_max_len}")

        sample = LPRDataInfo(
            img=img,
            label=label,
            length=len(label),
            origin_shape=origin_shape,
            target_shape=self.img_size,
        )
        return sample

    def transforms(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        return img

    def prepare_data(self):
        plates = []
        # 读取txt文件中的车牌
        for label_file in self.label_files:
            with open(label_file, "r", encoding="utf-8") as f:
                # 只读第一行
                plate = f.readline().strip()
                plates.append(plate)
        return plates

    @staticmethod
    def check_relu(char):
        # 车牌字符排列规则
        return True

    def collate_fn(self, batch_samples: list[LPRDataInfo]) -> LPRBatchDataInfo:
        imgs = []
        labels = []
        lengths = []

        for sample in batch_samples:
            imgs.append(torch.from_numpy(sample.img.transpose(2, 0, 1)))
            labels.extend(sample.label)
            lengths.append(sample.length)
        labels = np.asarray(labels).flatten().astype(np.int32)

        return LPRBatchDataInfo(
            data=torch.stack(imgs, 0),
            target=torch.from_numpy(labels),
            lengths=lengths
        )
