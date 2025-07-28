import time

import cv2
import torch

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

from tinytrain.global_var import NUM_THREADS, RANK
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.utils.checks import check_detect_yolo_label
from tinytrain.utils.data_utils import cv_imread, load_image_cache_file

from tinytrain.data.augment import YOLODetectionAugmentation
from tinytrain.data.data_format import DetectDataInfo, DetectBatchDataInfo
from tinytrain.data.dataset import TTBaseVisionDataset


class YOLODetectionDataset(TTBaseVisionDataset):
    """
    YOLO detection dataset.采用YOLO格式作为标签。
    """

    def __init__(self,
                 config_manager,
                 img_path: Path | list[Path],
                 mode: str = "train",
                 ):
        self.rgb: bool = config_manager.augment["rgb"]
        self.samples: list[DetectDataInfo] = []
        self.detect_augmentation = YOLODetectionAugmentation(config_manager)
        super().__init__(config_manager=config_manager, img_path=img_path, mode=mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # from copy import deepcopy
        # sample = deepcopy(self.samples[index]) # 做深拷贝，避免内存常驻img数据
        sample = self.samples[index]  # 浅拷贝
        sample = DetectDataInfo(**vars(sample))  # 仅新建外壳，img 字段后面覆盖

        if self.cache:
            sample.img = load_image_cache_file(sample.img_file)  # BGR 走内存映射（省 RAM + 多进程共享）
        else:
            sample.img = cv_imread(sample.img_file)  # BGR  读取图片存在IO瓶颈

        # Convert NumPy array
        if self.rgb:
            sample.img = cv2.cvtColor(sample.img, cv2.COLOR_BGR2RGB)

        origin_shape = sample.img.shape[:2][::-1]  # w,h
        sample.origin_shape = origin_shape
        sample.current_shape = origin_shape
        sample.target_shape = self.img_size

        # use transform
        sample = self.transform(sample)

        return sample

    def custom_check(self):
        messages = []
        with ThreadPool(NUM_THREADS) as pool:
            def wrapper(args):
                return check_detect_yolo_label(*args)

            # 检查非背景图片
            results1 = pool.imap(
                func=wrapper,
                iterable=zip(self.img_files, self.npy_files if len(self.npy_files) else repeat(None), self.label_files))

            pbar = TTProgressBar(results1, total=len(self.img_files), desc="check nor background images")

            for i, (img_file, npy_file, message, cls, boxes) in enumerate(pbar):
                sample = DetectDataInfo(
                    img_file=npy_file if self.cache else img_file,
                    label=cls,
                    bboxes=boxes,
                    bbox_format="cxcywh",
                    normalized=True
                )
                self.samples.append(sample)

                if message:
                    messages.append(message)

            # 检查背景图片
            results2 = pool.imap(
                func=wrapper,
                iterable=zip(self.bg_img_files, self.bg_npy_files if len(self.bg_npy_files) else repeat(None)))

            pbar = TTProgressBar(results2, total=len(self.bg_img_files), desc="check background images")

            for i, (bg_img_file, bg_npy_file, message, cls, boxes) in enumerate(pbar):
                sample = DetectDataInfo(
                    img_file=bg_npy_file if self.cache else bg_img_file,
                    label=cls,
                    bboxes=boxes,
                    bbox_format="cxcywh",
                    normalized=True
                )
                self.samples.append(sample)

                if message:
                    messages.append(message)

        for msg in messages:
            LOGGER.warning(msg)

        # 添加last DetectDataInfo，用于多图像融合增强等操作
        samples_len = len(self.samples)  # 提前计算长度
        for i, sample in enumerate(self.samples):
            # 计算下一个元素的索引，使用模运算实现循环
            next_index = (i + 1) % samples_len
            # sample.next_ImgDataInfo = self.samples[next_index]   # dataloader多进程拷贝存在问题，暂时不开放
            sample.next_ImgDataInfo = None

    def set_transform(self):
        if self.mode == "train":
            return self.detect_augmentation.augment()
        else:
            return self.detect_augmentation.transform()

    def collate_fn(self, batch_samples: list[DetectDataInfo]):
        train_mode = self.mode == "train"

        # ---- 预分配 ----
        B = len(batch_samples)
        # 获取 dtype 和通道顺序
        first = batch_samples[0].img  # (H, W, C) numpy array
        C, H, W = first.transpose(2, 0, 1).shape
        dtype_torch = torch.from_numpy(first).dtype  # 自动匹配 numpy dtype
        images = torch.empty((B, C, H, W), dtype=dtype_torch)

        bboxes_list = []
        labels_list = []
        bbox_idx_list = []

        origin_shapes = None
        target_shapes = None

        if not train_mode:
            origin_shapes = torch.empty((B, 2), dtype=torch.int64)
            target_shapes = torch.empty((B, 2), dtype=torch.int64)

        for i, sample in enumerate(batch_samples):
            # 直接拷贝 numpy -> torch，无额外内存
            images[i] = torch.from_numpy(sample.img.transpose(2, 0, 1))

            bboxes_list.append(torch.from_numpy(sample.bboxes))
            labels_list.append(torch.from_numpy(sample.label))
            bbox_idx_list.append(torch.full((sample.bboxes.shape[0],), i, dtype=torch.int32))

            if not train_mode:
                origin_shapes[i] = torch.tensor(sample.origin_shape, dtype=torch.int64)
                target_shapes[i] = torch.tensor(sample.target_shape, dtype=torch.int64)

        # ---- 拼接 ----
        bboxes = torch.cat(bboxes_list, dim=0).float()
        labels = torch.cat(labels_list, dim=0).long()
        bboxes_idx = torch.cat(bbox_idx_list, dim=0)

        return DetectBatchDataInfo(
            origin_shapes=origin_shapes if not train_mode else None,
            target_shapes=target_shapes if not train_mode else None,
            data=images,
            bboxes=bboxes,
            target=labels,
            bboxes_idx=bboxes_idx
        )
