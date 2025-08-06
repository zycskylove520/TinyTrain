"""
YOLODetectionDataset
==================
YOLO 通用检测数据集封装类，支持 YOLO 格式（class cx cy w h，归一化坐标）标签。
设计目标：极致性能 + 零冗余内存占用 + 训练/验证/推理一体化。

主要特性
--------
1. 内存映射缓存
   通过 `load_image_cache_file` 将图片以只读内存映射方式缓存，
   多进程 DataLoader 共享物理内存，显著降低 RAM 占用。
2. 异步标签校验
   使用 `ThreadPool` 并行检查所有图片与标签合法性，
   实时汇总异常信息并统一 `LOGGER.warning` 输出。
3. 动态增强管道
   训练/验证模式分别绑定 `YOLODetectionAugmentation` 的不同策略，
   在 `__getitem__` 中按需执行。
4. 训练友好 collate_fn
   预分配 GPU 连续内存、拼接 batch，返回 `DetectBatchDataInfo`，
   可直接送入模型，无需额外后处理。
5. 背景图自动混入
   支持空标签背景图（background images），在 `custom_check` 中与普通样本
   一起校验、拼接，增强模型鲁棒性。

示例
----
>>> cfg = ConfigManager("cfg/yolov8-detect.toml")
>>> ds = YOLODetectionDataset(cfg, img_path=Path("datasets/coco128/images/train"), mode="train")
>>> dl = InfiniteDataLoader(ds, batch_size=32, num_workers=8)
>>> for batch in dl:
...     # batch: DetectBatchDataInfo
...     preds = model(batch.data, batch.bboxes, batch.target)
"""

import cv2
import torch

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

from tinytrain.data import PoseBatchDataInfo
from tinytrain.global_var import NUM_THREADS, RANK
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.utils.checks import check_detect_yolo_label, check_pose_yolo_label
from tinytrain.utils.data_utils import cv_imread, load_image_cache_file

from tinytrain.models.yolo.yolo_augment import YOLODetectionAugmentation, YOLOPoseAugmentation
from tinytrain.data.data_format import DetectDataInfo, DetectBatchDataInfo, PoseDataInfo
from tinytrain.data.dataset import TTBaseVisionDataset


class YOLODetectionDataset(TTBaseVisionDataset):
    """
    YOLO 检测数据集封装，支持 YOLO 格式标签（class cx cy w h，已归一化）。
    """

    def __init__(self,
                 config_manager,
                 img_path: Path | list[Path],
                 mode: str = "train",
                 ):
        """
        参数
        ----
        config_manager : ConfigManager
            全局配置管理器，读取 augment / dataset / core 等配置段。
        img_path : Path | list[Path]
            图片所在目录或目录列表，支持混合背景图。
        mode : {"train", "val", "test"}
            数据集模式，决定增强策略及 collate 行为。
        """
        self.rgb: bool = config_manager.augment["rgb"]
        self.samples: list[DetectDataInfo] = []
        super().__init__(config_manager=config_manager, img_path=img_path, mode=mode)

    def __len__(self):
        """返回样本总数（含背景图）。"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        根据索引加载单张图片与标签，应用增强后返回 DetectDataInfo。

        注意
        ----
        - 若启用 cache，则通过内存映射读取，避免重复 IO。
        - 返回对象仅浅拷贝外壳，内部 img 字段会被新的 ndarray 覆盖。
        """

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
        sample.target_shape = self.img_size

        # use transform
        if self.mode == "train":
            sample = self.transform.do_augment(sample)
        else:
            sample = self.transform.do_transform(sample)

        return sample

    def custom_check(self):
        """
        多线程并行校验所有图片与标签，并构建 DetectDataInfo 列表。
        背景图（无标签）与普通图共用同一校验函数，异常信息统一收集。
        """
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
        """根据模式返回训练增强或验证/测试转换。"""
        detect_augmentation = YOLODetectionAugmentation(self.config_manager)
        if self.mode == "train":
            detect_augmentation.set_augment()
        else:
            detect_augmentation.set_transform()
        return detect_augmentation

    def collate_fn(self, batch_samples: list[DetectDataInfo]):
        """
        将 DetectDataInfo 列表打包为 DetectBatchDataInfo。

        实现细节
        --------
        - 预分配连续 GPU 内存，避免拼接时额外拷贝。
        - 训练模式不返回 origin/target shape，节省显存。
        - 返回的 `bboxes_idx` 用于 scatter 到正确 GPU 卡。
        """
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

class YOLOPoseDataset(TTBaseVisionDataset):
    """
    YOLO 检测数据集封装，支持 YOLO 格式标签（class cx cy w h，已归一化）。
    """

    def __init__(self,
                 config_manager,
                 img_path: Path | list[Path],
                 mode: str = "train",
                 ):
        """
        参数
        ----
        config_manager : ConfigManager
            全局配置管理器，读取 augment / dataset / core 等配置段。
        img_path : Path | list[Path]
            图片所在目录或目录列表，支持混合背景图。
        mode : {"train", "val", "test"}
            数据集模式，决定增强策略及 collate 行为。
        """
        self.rgb: bool = config_manager.augment["rgb"]
        self.keypoint_shape= config_manager.dataset["keypoint_shape"]
        self.samples: list[PoseDataInfo] = []
        super().__init__(config_manager=config_manager, img_path=img_path, mode=mode)

    def __len__(self):
        """返回样本总数（含背景图）。"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        根据索引加载单张图片与标签，应用增强后返回 PoseDataInfo。

        注意
        ----
        - 若启用 cache，则通过内存映射读取，避免重复 IO。
        - 返回对象仅浅拷贝外壳，内部 img 字段会被新的 ndarray 覆盖。
        """

        # from copy import deepcopy
        # sample = deepcopy(self.samples[index]) # 做深拷贝，避免内存常驻img数据
        sample = self.samples[index]  # 浅拷贝
        sample = PoseDataInfo(**vars(sample))  # 仅新建外壳，img 字段后面覆盖

        if self.cache:
            sample.img = load_image_cache_file(sample.img_file)  # BGR 走内存映射（省 RAM + 多进程共享）
        else:
            sample.img = cv_imread(sample.img_file)  # BGR  读取图片存在IO瓶颈

        # Convert NumPy array
        if self.rgb:
            sample.img = cv2.cvtColor(sample.img, cv2.COLOR_BGR2RGB)

        origin_shape = sample.img.shape[:2][::-1]  # w,h
        sample.origin_shape = origin_shape
        sample.target_shape = self.img_size

        # use transform
        sample.img = cv2.resize(sample.img, self.img_size, interpolation=cv2.INTER_LINEAR)
        # sample = self.transform(sample)

        return sample

    def custom_check(self):
        """
        多线程并行校验所有图片与标签，并构建 DetectDataInfo 列表。
        背景图（无标签）与普通图共用同一校验函数，异常信息统一收集。
        """
        messages = []
        with ThreadPool(NUM_THREADS) as pool:
            def wrapper(args):
                return check_pose_yolo_label(*args)

            # 检查非背景图片
            results1 = pool.imap(
                func=wrapper,
                iterable=zip(self.img_files, repeat(self.keypoint_shape), self.npy_files if len(self.npy_files) else repeat(None), self.label_files))

            pbar = TTProgressBar(results1, total=len(self.img_files), desc="check nor background images")

            for i, (img_file, npy_file, message, cls, boxes, keypoints) in enumerate(pbar):
                sample = PoseDataInfo(
                    img_file=npy_file if self.cache else img_file,
                    label=cls,
                    bboxes=boxes,
                    bbox_format="cxcywh",
                    key_points=keypoints,
                    kpt_shape=self.keypoint_shape,
                    normalized=True
                )
                self.samples.append(sample)

                if message:
                    messages.append(message)

            # 检查背景图片
            results2 = pool.imap(
                func=wrapper,
                iterable=zip(self.bg_img_files, repeat(self.keypoint_shape), self.bg_npy_files if len(self.bg_npy_files) else repeat(None)))

            pbar = TTProgressBar(results2, total=len(self.bg_img_files), desc="check background images")

            for i, (bg_img_file, bg_npy_file, message, cls, boxes, keypoints) in enumerate(pbar):
                sample = PoseDataInfo(
                    img_file=bg_npy_file if self.cache else bg_img_file,
                    label=cls,
                    bboxes=boxes,
                    bbox_format="cxcywh",
                    key_points=keypoints,
                    kpt_shape=self.keypoint_shape,
                    normalized=True
                )
                self.samples.append(sample)

                if message:
                    messages.append(message)

        for msg in messages:
            LOGGER.warning(msg)

        # 添加last PoseDataInfo，用于多图像融合增强等操作
        samples_len = len(self.samples)  # 提前计算长度
        for i, sample in enumerate(self.samples):
            # 计算下一个元素的索引，使用模运算实现循环
            next_index = (i + 1) % samples_len
            # sample.next_ImgDataInfo = self.samples[next_index]   # dataloader多进程拷贝存在问题，暂时不开放
            sample.next_ImgDataInfo = None

    def set_transform(self):
        """根据模式返回训练增强或验证/测试转换。"""
        pose_augmentation = YOLOPoseAugmentation(self.config_manager)
        if self.mode == "train":
            pose_augmentation.set_augment()
        else:
            pose_augmentation.set_transform()
        return pose_augmentation

    def collate_fn(self, batch_samples: list[PoseDataInfo]):
        """
        将 PoseDataInfo 列表打包为 DetectBatchDataInfo。

        实现细节
        --------
        - 预分配连续 GPU 内存，避免拼接时额外拷贝。
        - 训练模式不返回 origin/target shape，节省显存。
        - 返回的 `bboxes_idx` 用于 scatter 到正确 GPU 卡。
        """
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
        keypoints_list = []
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
            keypoints_list.append(torch.from_numpy(sample.key_points))
            bbox_idx_list.append(torch.full((sample.bboxes.shape[0],), i, dtype=torch.int32))

            if not train_mode:
                origin_shapes[i] = torch.tensor(sample.origin_shape, dtype=torch.int64)
                target_shapes[i] = torch.tensor(sample.target_shape, dtype=torch.int64)

        # ---- 拼接 ----
        bboxes = torch.cat(bboxes_list, dim=0).float()
        labels = torch.cat(labels_list, dim=0).long()
        keypoints = torch.cat(keypoints_list, dim=0).float()
        bboxes_idx = torch.cat(bbox_idx_list, dim=0)

        return PoseBatchDataInfo(
            origin_shapes=origin_shapes if not train_mode else None,
            target_shapes=target_shapes if not train_mode else None,
            data=images,
            bboxes=bboxes,
            target=labels,
            batch_key_points=keypoints,
            bboxes_idx=bboxes_idx
        )