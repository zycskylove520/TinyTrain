import glob
import random
import shutil
from typing import Any, AnyStr

import numpy as np
import torch
import cv2
import torch.distributed as dist

from multiprocessing.pool import ThreadPool
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from tinytrain.global_var import NUM_THREADS, RANK
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.utils.checks import check_image_and_label, check_image, IMG_FORMATS, check_img_size
from tinytrain.utils.data_utils import save_dict_cache_file, cv_imread, get_hash, load_dict_cache_file, save_image_cache_file

from .augment import ClassificationAugmentation
from .data_format import ClassifyDataInfo, ClassifyBatchDataInfo


class TTBaseDataset(Dataset):
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        return batch


class TTBaseVisionDataset(TTBaseDataset):
    """
    BaseDataset类只负责检查图片和标签txt文件是否有问题，并根据用户需求做图片缓存。
    """

    def __init__(
            self,
            config_manager,
            img_path: Path | list[Path],
            mode: str = "train"
    ):
        super().__init__(config_manager=config_manager)
        self.img_path = img_path
        self.img_size = self.config_manager.dataset["img_size"]
        self.mode = mode
        self.crop_fraction = self.config_manager.augment["img_crop_fraction"]
        self.cache = self.config_manager.dataset["cache"]

        self.bg_img_files = []  # 图片路径为绝对路径，都为背景图片
        self.bg_npy_files = []  # npy路径为绝对路径，都为背景图片的缓存

        self.img_files = []  # 图片路径为绝对路径，都为有标签对应的图片
        self.label_files = []  # 标签文件路径为绝对路径，都为有图片对应的标签文件
        self.npy_files = []  # npy路径为绝对路径，都为有标签对应的图片的缓存

        self.init()
        self.crop_samples()
        # 用户增加自定义检查
        self.custom_check()

        self.transform = self.set_transform()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.cache:
            return len(self.npy_files + self.bg_npy_files)
        return len(self.img_files + self.bg_img_files)

    def crop_samples(self):
        # samples clip, reduce training fraction
        if self.mode == "train" and self.crop_fraction < 1.0:
            self.img_files = self.img_files[: round(len(self.img_files) * self.crop_fraction)]
            self.label_files = self.label_files[: round(len(self.label_files) * self.crop_fraction)]
            if self.cache:
                self.npy_files = self.npy_files[: round(len(self.npy_files) * self.crop_fraction)]

    def init(self):
        self.img_size = self.config_manager.dataset["img_size"] = check_img_size(self.img_size, mode=self.mode)
        self.check_class_names()
        self.check_images_and_labels()

        if self.cache:
            # 1) 全局哈希缓存文件
            global_cache_file = None
            for p in (self.img_path if isinstance(self.img_path, list) else [self.img_path]):
                global_cache_file = p.parent / f"{self.mode}.npy"
                break  # 只需拿一个路径即可

            # 2) 生成缓存：仅 rank-0 执行
            if RANK in {-1, 0}:
                # 2-a) 全局哈希校验
                if global_cache_file and global_cache_file.exists():
                    hash_1 = get_hash(sorted([str(img_file) for img_file in self.img_files + self.bg_img_files]))
                    hash_2 = load_dict_cache_file(global_cache_file)
                    need_rebuild = (hash_1 != hash_2)
                else:
                    need_rebuild = True

                # 2-b) 单文件缓存目录校验
                for p in (self.img_path if isinstance(self.img_path, list) else [self.img_path]):
                    img_path_parts = p.parts
                    cache_path_parts = [part if part != "images" else "caches" for part in img_path_parts]
                    cache_path = Path(*cache_path_parts)
                    npy_files = glob.glob(str(cache_path / "*.npy"), recursive=True)
                    if len(npy_files) != len(self.img_files + self.bg_img_files):
                        need_rebuild = True
                        break

                # 2-c) 真正重建
                if need_rebuild:
                    self.make_disk_cache()

            # 3) 所有进程等待 rank-0 完成
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            # 4) 重新收集缓存文件列表
            self.npy_files.clear()
            self.bg_npy_files.clear()
            for img_file in self.img_files:
                img_file_parts = img_file.parts
                cache_file_parts = [part if part != "images" else "caches" for part in img_file_parts]
                cache_file = Path(*cache_file_parts).with_suffix(".npy")
                self.npy_files.append(cache_file)

            for bg_img_file in self.bg_img_files:
                bg_img_file_parts = bg_img_file.parts
                cache_file_parts = [part if part != "images" else "caches" for part in bg_img_file_parts]
                cache_file = Path(*cache_file_parts).with_suffix(".npy")
                self.bg_npy_files.append(cache_file)

            LOGGER.info(f"use cache loading...")

    def check_class_names(self):
        classes_num = len(self.config_manager.dataset.get("names", dict()).values())
        nc = self.config_manager.dataset["nc"]
        if classes_num != nc:
            LOGGER.warning(f"The names in the dataset.toml do not match the current dataset names. You can find the updated dataset.toml in the 'args' directory of the trained save path.")
            self.config_manager.dataset["names"] = {i: i for i in range(nc)}

    def check_images_and_labels(self):
        for p in self.img_path if isinstance(self.img_path, list) else [self.img_path]:
            # 获取标签路径
            label_path_parts = p.parts
            new_label_path_parts = [part if part != "images" else "labels" for part in label_path_parts]
            label_path = Path(*new_label_path_parts)
            # 获取所有文件
            _files = list(p.rglob("*.*"))
            # 过滤出图片文件
            img_files = [str(file) for file in _files if file.suffix.lower().lstrip('.') in IMG_FORMATS]

            with ThreadPool(NUM_THREADS) as pool:
                def wrapper(args):
                    return check_image_and_label(*args)

                label_paths = [label_path] * len(img_files)
                results = pool.imap(
                    func=wrapper,
                    iterable=zip(img_files, label_paths))

                desc = f"{self.mode}: Checking image and label files"
                messages = []

                pbar = TTProgressBar(results, desc=desc, total=len(img_files))

                for i, (bg_img_file, img_file, label_file, is_bg, good, bad, message) in enumerate(pbar):
                    # 图片没问题
                    if good:
                        if is_bg:
                            self.bg_img_files.append(Path(bg_img_file))
                        else:
                            self.img_files.append(Path(img_file))
                            self.label_files.append(Path(label_file))

                    if bad:
                        messages.append(message)

                for msg in messages:
                    LOGGER.warning(msg)

    def make_disk_cache(self):
        # 制作全局缓存，用于校验文件是否增删
        for p in self.img_path if isinstance(self.img_path, list) else [self.img_path]:
            hash_ = get_hash(sorted([str(img_file) for img_file in self.img_files + self.bg_img_files]))
            global_cache_file = p.parent / f"{self.mode}.npy"
            save_dict_cache_file(global_cache_file, hash_, allow_pickle=True)

        # 对每张图片单独制作缓存
        for p in self.img_path if isinstance(self.img_path, list) else [self.img_path]:
            # 获取缓存路径
            img_path_parts = p.parts
            cache_path_parts = [part if part != "images" else "caches" for part in img_path_parts]
            cache_path = Path(*cache_path_parts)
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(
                    func=self.thread_disk_cache,
                    iterable=self.img_files + self.bg_img_files)

                desc = f"Making {self.mode} data disk cache"
                # 下面的for循环不能删!
                for i, _ in TTProgressBar(enumerate(results), desc=desc, total=len(self.img_files + self.bg_img_files)):
                    continue

    def thread_disk_cache(self, img_file):
        # 获取缓存路径
        img_file_parts = img_file.parts
        cache_file_parts = [part if part != "images" else "caches" for part in img_file_parts]
        cache_file = Path(*cache_file_parts).with_suffix(".npy")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_image_cache_file(cache_file, cv_imread(img_file))  # 加载时走内存映射（省 RAM + 多进程共享）

    def custom_check(self):
        raise NotImplementedError("custom_check is not implemented.")

    def set_transform(self):
        raise NotImplementedError(f"set_transform is not implemented.")

    def collate_fn(self, batch):
        raise NotImplementedError(f"collate_fn is not implemented.")


class TTClassificationDataset(ImageFolder):
    def __init__(
            self,
            config_manager,
            root: str | Path,
            mode: str = "train"
    ):
        """
        Initialize object with root, image size, augmentations, and cache settings, etc.
        """
        if isinstance(root, list):
            root = root[0]
            LOGGER.warning(f"Classify datasets do not support multiple directories!Only loaded: {root}")
        super().__init__(root)
        self.config_manager = config_manager
        self.mode = mode
        self.cache = self.config_manager.dataset["cache"]
        self.img_size = self.config_manager.dataset["img_size"]
        self.crop_fraction = self.config_manager.augment["img_crop_fraction"]
        self.rgb: bool = self.config_manager.augment["rgb"]

        self.init()
        self.crop_samples()
        self.classification_augmentation = ClassificationAugmentation(self.config_manager)
        self.set_transform()

    def __getitem__(self, index):
        if self.cache:
            cache_file = self.samples[index]
            sample = load_dict_cache_file(cache_file)
            image = sample["img"]
            label = np.array(sample["label"])
        else:
            filename, label = self.samples[index]  # filename, label
            label = np.array(label)
            image = cv_imread(filename)  # BGR  读取图片存在IO瓶颈

            if image is None or image.size == 0:
                raise RuntimeError(f"Empty image at {filename}")

        # Convert NumPy array
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        origin_shape = image.shape[:2][::-1]
        sample = ClassifyDataInfo(
            img_file=self.samples[index] if self.cache else self.samples[index][0],
            img=image,
            origin_shape=origin_shape,
            current_shape=origin_shape,
            target_shape=self.img_size,
            label=label
        )

        # use transform
        sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """Return the total number of samples in the dataset_config."""
        return len(self.samples)

    def init(self):
        self.check_class_names()
        self.img_size = self.config_manager.dataset["img_size"] = check_img_size(self.img_size)
        self.samples = self.check_images()

        if self.cache:
            global_cache_file = self.root.parent / f"{self.mode}.npy"
            cache_path = self.root.parent / f"{self.mode}_cache/"

            if RANK in {-1, 0}:
                need_rebuild = False

                # 全局哈希校验
                if global_cache_file.exists():
                    hash_1 = get_hash(sorted([str(x[0]) for x in self.samples]))
                    hash_2 = load_dict_cache_file(global_cache_file)
                    if hash_1 != hash_2:
                        need_rebuild = True
                else:
                    need_rebuild = True

                # 单文件缓存数量校验
                if cache_path.exists():
                    samples = glob.glob(str(cache_path / "**" / "*.npy"), recursive=True)
                    if len(samples) != len(self.samples):
                        need_rebuild = True
                else:
                    need_rebuild = True

                if need_rebuild:
                    self.make_disk_cache()

            # 等待 rank-0 完成
            if RANK != -1 and dist.is_available() and dist.is_initialized():
                dist.barrier()

            # 重新收集缓存文件列表
            self.samples = glob.glob(str(cache_path / "**" / "*.npy"), recursive=True)

    def crop_samples(self):
        # samples clip, reduce training fraction
        if self.mode == "train" and self.crop_fraction < 1.0:
            origin_len = len(self.samples)
            # 打乱数据集
            random.shuffle(self.samples)
            self.samples = self.samples[: round(len(self.samples) * self.crop_fraction)]
            LOGGER.info(f"Perform datasets clipping, current {self.mode} dataset size is:{origin_len}x{self.crop_fraction}={len(self.samples)}")

    def set_transform(self):
        if self.mode == "train":
            self.transform = self.classification_augmentation.augment()
        else:
            self.transform = self.classification_augmentation.transform()

    def check_class_names(self):
        # 根据class_to_idx设置类别索引
        # self.names = self.config_manager.dataset["names"]
        names = {value: key for key, value in self.class_to_idx.items()}
        if self.config_manager.dataset.get("names", dict()) != names:
            LOGGER.warning(f"The names in the dataset.toml do not match the current dataset names. You can find the updated dataset.toml in the 'args' directory of the trained save path.")
            self.config_manager.dataset["names"] = names

    def check_images(self):
        with ThreadPool(NUM_THREADS) as pool:
            def wrapper(args):
                return check_image(*args)

            results = pool.imap(
                func=wrapper,
                iterable=self.samples)

            desc = f"Checking {self.mode} image files"
            messages = []
            samples = []

            pbar = TTProgressBar(results, desc=desc, total=len(self.samples))

            for i, (img_file, good, bad, message, label) in enumerate(pbar):
                if good:
                    samples.append((Path(img_file), label))
                if bad:
                    messages.append(message)

        for msg in messages:
            LOGGER.warning(msg)
        return samples

    def make_disk_cache(self):
        # 制作全局缓存，用于校验文件是否增删
        global_cache_file = self.root.parent / f"{self.mode}.npy"
        rel_paths = [str(Path(p[0]).relative_to(Path(p[0]).anchor)) for p in self.samples]
        hash_ = get_hash(sorted(rel_paths))
        save_dict_cache_file(global_cache_file, hash_, allow_pickle=True)

        # 对每张图片单独制作缓存
        cache_path = self.root.parent / f"{self.mode}_cache/"
        shutil.rmtree(cache_path, ignore_errors=True)  # 先清空
        with ThreadPool(NUM_THREADS) as pool:
            def wrapper(args):
                return self.thread_disk_cache(*args)

            results = pool.imap(
                func=wrapper,
                iterable=self.samples)

            desc = f"Making {self.mode} data disk cache"
            # 下面的for循环不能删!
            for i, _ in TTProgressBar(enumerate(results), desc=desc, total=len(self.samples)):
                continue

    def thread_disk_cache(self, img_file, label):
        sample = {}
        cache_file = (self.root.parent / f"{self.mode}_cache/" / img_file.relative_to(self.root)).with_suffix(".npy")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        img = cv_imread(img_file)
        sample["img"] = img
        sample["label"] = label
        save_dict_cache_file(cache_file, sample, allow_pickle=True)

    def collate_fn(self, batch: list[ClassifyDataInfo]):
        B = len(batch)
        if B == 0:
            raise ValueError("Empty batch!")

        # 预分配张量，避免 Python list → tensor 拷贝
        first = batch[0].img  # (H, W, C) numpy
        C, H, W = first.transpose(2, 0, 1).shape
        dtype = torch.from_numpy(first).dtype  # 保持原 dtype

        images = torch.empty((B, C, H, W), dtype=dtype)
        labels = torch.empty(B, dtype=torch.int64)

        # 直接填充
        for i, sample in enumerate(batch):
            images[i] = torch.from_numpy(sample.img.transpose(2, 0, 1))
            labels[i] = torch.from_numpy(sample.label).item()

        return ClassifyBatchDataInfo(
            origin_shapes=None,
            target_shapes=None,
            data=images,
            target=labels
        )
