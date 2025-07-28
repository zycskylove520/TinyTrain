import time

import numpy as np
from torch import memory_format

from tinytrain.data import DetectBatchDataInfo
from tinytrain.data.data_format import DetectDataInfo
from tinytrain.models.yolo.yolo_dataset import YOLODetectionDataset
from tinytrain.engine.trainer import BaseTrainer
from tinytrain.metrics.label_info import LabelInfo
from tinytrain.utils import LOGGER


class YOLODetectionTrainer(BaseTrainer):
    def build_dataset(self, mode="train"):
        if mode == "train":
            return YOLODetectionDataset(config_manager=self.config_manager,
                                        img_path=self.train_dir,
                                        mode="train"
                                        )
        elif mode == "val":
            return YOLODetectionDataset(config_manager=self.config_manager,
                                        img_path=self.val_dir,
                                        mode="val"
                                        )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: DetectBatchDataInfo) -> DetectBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std

        return batch_samples

    # def build_dataloader(self, world_size: int, mode: str = "train"):
    #     """
    #     这里可以选择不重写该函数。实测InfiniteDataLoader和默认DataLoader效率基本一致，速度略高于默认DataLoader一点点，真的就一点点
    #     缺点：使用InfiniteDataLoader会失去数据多样性分布
    #     """
    #     # 计算每个进程的批量大小
    #     batch_size = self.batch_size // max(world_size, 1)
    #
    #     # 构建数据集
    #     dataset = self.build_dataset(mode=mode)
    #
    #     # 配置数据加载器参数
    #     shuffle = mode == "train"
    #     batch_size = min(batch_size, len(dataset))  # 确保批量大小不超过数据集大小
    #     num_devices = torch.cuda.device_count()  # CUDA 设备数量
    #     num_workers = min(os.cpu_count() // max(num_devices, 1), self.config_manager.core["workers"])  # 工作进程数量
    #     sampler = DistributedSampler(dataset, shuffle=shuffle) if world_size > 1 else None
    #     generator = torch.Generator()
    #     generator.manual_seed(self.config_manager.core["seed"] + RANK)
    #
    #     # 根据模式调整批量大小和是否打乱数据
    #     if mode == "train":
    #         batch_size = batch_size
    #         shuffle = shuffle and sampler is None
    #     else:  # val 或 test 模式
    #         batch_size = batch_size // 2
    #         shuffle = self.config_manager.core["shuffle_val_dataloader"]
    #
    #     # 创建 DataLoader
    #     dataloader = InfiniteDataLoader(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         num_workers=num_workers,
    #         sampler=sampler,
    #         collate_fn=getattr(dataset, "collate_fn", None),
    #         generator=generator,
    #         persistent_workers=True if num_workers > 0 else False,
    #         drop_last=False,
    #         pin_memory=True,
    #         prefetch_factor=2
    #     )
    #
    #     return dataloader

    def plot_something_before_train(self):
        """
        绘制标签统计信息图
        """
        LOGGER.info(f"Start plotting label Statistics information before training...")
        train_samples: list[DetectDataInfo] = self.train_dataloader.dataset.samples
        labels = []
        bboxes = []
        for sample in train_samples:
            labels.append(sample.label)
            bboxes.append(sample.bboxes)
        labels = np.concatenate(labels, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        class_names = list(self.config_manager.dataset["names"].values())

        label_info = LabelInfo(num_classes=self.config_manager.dataset["nc"], class_names=class_names, labels=labels, bboxes=bboxes)
        label_info.plot(self.save_dir)
