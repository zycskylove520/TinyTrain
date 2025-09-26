"""
人脸识别训练 & 验证数据集

FaceRecognitionTrainDataset
    继承自 torchvision.datasets.ImageFolder，用于“分类”式训练。
    支持：
    1. 单目录加载（root 为 list 时仅取第 0 项）
    2. 按比例随机裁剪训练集（crop_fraction）
    3. 随机水平翻转 + 归一化（mean=0.5, std=0.5）
    4. 返回 ClassifyDataInfo 结构体，collate 后得到 ClassifyBatchDataInfo

FaceRecognitionValidDataset
    继承自 TTBaseMapDataset，用于“配对”验证（verification）。
    支持：
    1. 读取 *.txt 列表，每行格式：img1_path img2_path is_same(0/1)
    2. 返回 (sample1, sample2, is_pair) 三元组
    3. collate 后得到 FaceRecognitionValidBatchDataInfo
"""

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
from tinytrain.data.data_format import ClassifyDataInfo, ClassifyBatchDataInfo, FaceRecognitionValidBatchDataInfo, BaseBatchDataInfo
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import make2tuple
from tinytrain.utils.data_utils import cv_imread


class FaceRecognitionTrainDataset(ImageFolder):
    """
    人脸识别训练数据集（分类任务）
    目录结构须符合 ImageFolder 约定：
    root/
    ├── person1/
    │   ├── 0001.jpg
    │   └── 0002.jpg
    ├── person2/
    │   └── 0001.jpg
    """

    def __init__(self, config_manager, root: list[Path]):
        # 1. 仅支持单目录，多目录时给出警告并取第 0 个
        if isinstance(root, list):
            if len(root) > 1:
                LOGGER.warning(f"Face recognition datasets do not support multiple directories! Only loaded: {root[0]}")
            root = root[0]

        # 2. 父类 ImageFolder 完成样本扫描：self.samples = [(path, class_id), ...]
        super().__init__(root=root)

        self.config_manager = config_manager
        self.img_size = make2tuple(self.config_manager.dataset["train_img_size"])
        self.crop_fraction = self.config_manager.augment["img_crop_fraction"]
        self.rgb: bool = self.config_manager.augment["rgb"]

        # 3. 按比例裁剪训练集
        self.crop_samples()

        # 4. 定义数据增强
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> ClassifyDataInfo:
        """
        返回单个样本
        流程：
        1. 读取图片（BGR）
        2. 可选 BGR→RGB
        3. resize 到目标尺寸
        4. 封装成 ClassifyDataInfo
        5. transform → tensor
        """
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
        """训练阶段按 crop_fraction 随机裁剪数据集（减少样本量）"""
        if self.crop_fraction < 1.0:
            origin_len = len(self.samples)
            # 打乱数据集
            random.shuffle(self.samples)
            self.samples = self.samples[: round(len(self.samples) * self.crop_fraction)]
            LOGGER.info(f"Perform datasets clipping, current train dataset size is:{origin_len}x{self.crop_fraction}={len(self.samples)}")

    def collate_fn(self, batch: list[ClassifyDataInfo]) -> ClassifyBatchDataInfo:
        """
       将 list[ClassifyDataInfo] 合并成批张量
       采用预分配策略，避免 Python list → tensor 的额外拷贝
       """
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
    """
    人脸识别验证数据集（配对任务）
    列表文件格式（txt）：
        img1 img2 is_same
        0001.jpg 0002.jpg 1
        0003.jpg 0004.jpg 0
    路径均为相对于 txt 所在目录的相对路径
    """

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
        """读取全部 txt 文件，生成列表：samples = [((img1_path, img2_path), is_pair), ...]"""
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

    def __getitem__(self, index) -> tuple[tuple, tuple, int]:
        """
        返回：
        sample1: ClassifyDataInfo  第一张图
        sample2: ClassifyDataInfo  第二张图
        is_pair: int               是否同一人（0/1）
        """
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
        flip_image1 = cv2.flip(image1, 1)
        image2 = cv2.resize(image2, self.img_size)
        flip_image2 = cv2.flip(image2, 1)

        origin_shape1 = image1.shape[:2][::-1]
        sample1 = ClassifyDataInfo(
            img_file=filename1,
            img=image1,
            origin_shape=origin_shape1,
            target_shape=self.img_size
        )
        flip_sample1 = ClassifyDataInfo(
            img_file=filename1,
            img=flip_image1,
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
        flip_sample2 = ClassifyDataInfo(
            img_file=filename2,
            img=flip_image2,
            origin_shape=origin_shape2,
            target_shape=self.img_size
        )

        sample1.img = self.transform(Image.fromarray(sample1.img))  # torch.tensor and shape=[c,h,w]
        flip_sample1.img = self.transform(Image.fromarray(flip_sample1.img))

        sample2.img = self.transform(Image.fromarray(sample2.img))  # torch.tensor and shape=[c,h,w]
        flip_sample2.img = self.transform(Image.fromarray(flip_sample2.img))

        return (sample1, sample2), (flip_sample1, flip_sample2), is_pair

    def collate_fn(self, batch: list[tuple[tuple, tuple, int]]) -> BaseBatchDataInfo:
        """
        合并成批张量
        返回：
        data = [images1, images2]  # 两个 (B,C,H,W) 张量
        match_tensor = (B,)  # 0/1 标签
        """
        B = len(batch)
        if B == 0:
            raise ValueError("Empty batch!")

        # 预分配张量，避免 Python list → tensor 拷贝
        first: torch.Tensor = batch[0][0][0].img  # (C, H, W) numpy
        C, H, W = first.shape
        dtype = first.dtype  # 保持原 dtype

        images1 = torch.empty((B, C, H, W), dtype=dtype)
        images2 = torch.empty((B, C, H, W), dtype=dtype)
        flip_images1 = torch.empty((B, C, H, W), dtype=dtype)
        flip_images2 = torch.empty((B, C, H, W), dtype=dtype)
        match_tensor = torch.empty(B, dtype=torch.int64)

        for i, ((sample1, sample2), (flip_sample1, flip_sample2), is_pair) in enumerate(batch):
            images1[i] = sample1.img
            images2[i] = sample2.img
            flip_images1[i] = flip_sample1.img
            flip_images2[i] = flip_sample2.img
            match_tensor[i] = is_pair

        return BaseBatchDataInfo(
            data=((images1, images2), (flip_images1, flip_images2), match_tensor),
        )
