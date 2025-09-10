import os
import random
import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import transforms

from tinytrain.data.base import TTBaseMapDataset
from tinytrain.data.data_format import ClassifyDataInfo, ClassifyBatchDataInfo, FaceRecognitionValidBatchDataInfo
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import make2tuple
from tinytrain.utils.data_utils import cv_imread


class FaceRecognitionTrainDataset(ImageFolder):
    def __init__(self, config_manager, root: list[Path]):
        if isinstance(root, list):
            if len(root) > 1:
                LOGGER.warning(f"Face recognition datasets do not support multiple directories! Only loaded: {root[0]}")
            root = root[0]
        super().__init__(root=root)
        self.config_manager = config_manager
        self.img_size = make2tuple(self.config_manager.dataset["train_img_size"])
        self.crop_fraction = self.config_manager.augment["img_crop_fraction"]
        self.rgb: bool = self.config_manager.augment["rgb"]

        self.crop_samples()

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> ClassifyDataInfo:
        filename, label = self.samples[index]  # filename, label
        label = np.array(label)
        image = cv_imread(filename)  # BGR

        if image is None or image.size == 0:
            raise RuntimeError(f"Empty image at {filename}")

        # Convert NumPy array
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.img_size)

        origin_shape = image.shape[:2][::-1]
        sample = ClassifyDataInfo(
            img_file=filename,
            img=image,
            origin_shape=origin_shape,
            target_shape=self.img_size,
            label=label
        )

        # use transform
        sample.img = Image.fromarray(sample.img)
        sample.img = self.transform(sample.img)  # torch.tensor and shape=[c,h,w]
        return sample

    def crop_samples(self):
        """训练阶段按 crop_fraction 裁剪数据集。"""
        if self.crop_fraction < 1.0:
            origin_len = len(self.samples)
            # 打乱数据集
            random.shuffle(self.samples)
            self.samples = self.samples[: round(len(self.samples) * self.crop_fraction)]
            LOGGER.info(f"Perform datasets clipping, current train dataset size is:{origin_len}x{self.crop_fraction}={len(self.samples)}")

    def collate_fn(self, batch: list[ClassifyDataInfo]) -> ClassifyBatchDataInfo:
        B = len(batch)
        if B == 0:
            raise ValueError("Empty batch!")

        # 预分配张量，避免 Python list → tensor 拷贝
        first: torch.Tensor = batch[0].img  # (C, H, W) numpy
        C, H, W = first.shape
        dtype = first.dtype  # 保持原 dtype

        images = torch.empty((B, C, H, W), dtype=dtype)
        labels = torch.empty(B, dtype=torch.int64)

        # 直接填充
        for i, sample in enumerate(batch):
            images[i] = sample.img
            labels[i] = torch.from_numpy(sample.label).long().item()

        return ClassifyBatchDataInfo(
            data=images,
            target=labels
        )


class FaceRecognitionValidDataset(TTBaseMapDataset):
    def __init__(self, config_manager, txt_files: list[Path]):
        super().__init__(config_manager)
        self.txt_files = txt_files
        self.img_size = make2tuple(self.config_manager.dataset["val_img_size"])
        self.rgb: bool = self.config_manager.augment["rgb"]

        self.samples = self.make_pair_dataset()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def make_pair_dataset(self) -> list:
        samples = []
        for pairs_file in self.txt_files:
            root = Path(pairs_file).parent
            assert pairs_file.suffix == ".txt", "valid dataset should have .txt file"
            with open(pairs_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 3:
                        raise ValueError(f"Unexpected line format: {line}")
                    img1 = os.path.join(root, parts[0])
                    img2 = os.path.join(root, parts[1])
                    is_pair = int(parts[2])
                    samples.append(((img1, img2), is_pair))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> tuple[ClassifyDataInfo, ClassifyDataInfo, int]:
        (filename1, filename2), is_pair = self.samples[index]
        image1 = cv_imread(filename1)  # BGR
        image2 = cv_imread(filename2)

        if image1 is None or image1.size == 0:
            raise RuntimeError(f"Empty image at {filename1}")
        if image2 is None or image2.size == 0:
            raise RuntimeError(f"Empty image at {filename2}")

        # Convert NumPy array
        if self.rgb:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, self.img_size)
        image2 = cv2.resize(image2, self.img_size)

        origin_shape1 = image1.shape[:2][::-1]
        sample1 = ClassifyDataInfo(
            img_file=filename1,
            img=image1,
            origin_shape=origin_shape1,
            target_shape=self.img_size
        )

        origin_shape2 = image2.shape[:2][::-1]
        sample2 = ClassifyDataInfo(
            img_file=filename2,
            img=image2,
            origin_shape=origin_shape2,
            target_shape=self.img_size
        )

        sample1.img = Image.fromarray(sample1.img)
        sample1.img = self.transform(sample1.img)  # torch.tensor and shape=[c,h,w]

        sample2.img = Image.fromarray(sample2.img)
        sample2.img = self.transform(sample2.img)  # torch.tensor and shape=[c,h,w]

        return sample1, sample2, is_pair

    def collate_fn(self, batch: list[tuple[ClassifyDataInfo, ClassifyDataInfo, int]]) -> FaceRecognitionValidBatchDataInfo:
        B = len(batch)
        if B == 0:
            raise ValueError("Empty batch!")

        # 预分配张量，避免 Python list → tensor 拷贝
        first: torch.Tensor = batch[0][0].img  # (C, H, W) numpy
        C, H, W = first.shape
        dtype = first.dtype  # 保持原 dtype

        images1 = torch.empty((B, C, H, W), dtype=dtype)
        images2 = torch.empty((B, C, H, W), dtype=dtype)
        match_tensor = torch.empty(B, dtype=torch.int64)

        for i, (sample1, sample2, is_pair) in enumerate(batch):
            images1[i] = sample1.img
            images2[i] = sample2.img
            match_tensor[i] = is_pair

        return FaceRecognitionValidBatchDataInfo(
            data=[images1, images2],
            match_tensor=match_tensor,
        )
