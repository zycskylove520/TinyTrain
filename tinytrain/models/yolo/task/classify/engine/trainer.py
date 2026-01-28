"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch

from tinytrain.cfg import TTConfigManager
from tinytrain.data.data_format import ClassifyBatchDataInfo, ClassifyDataInfo
from tinytrain.data.dataset import TTClassificationDataset
from tinytrain.engine import TTBaseModel
from tinytrain.global_var import RANK
from tinytrain.metrics.classify_metrics import ClassesLabelHistogram
from tinytrain.models.yolo.yolo_trainer import YOLOTrainer
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback


class YOLOClassificationTrainer(YOLOTrainer):

    def __init__(self, config_manager: TTConfigManager, device: torch.device, model: TTBaseModel, callback: Callback):
        super().__init__(config_manager, device, model, callback)

        # 获取配置参数
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"]

        # 转换为 tensor 并确保在正确的设备上
        if isinstance(mean, (int, float)):
            self.mean_tensor = torch.tensor([mean], device=self.device)
        else:
            self.mean_tensor = torch.tensor(mean, device=self.device)

        if isinstance(std, (int, float)):
            self.std_tensor = torch.tensor([std], device=self.device) + 1e-8
        else:
            self.std_tensor = torch.tensor(std, device=self.device) + 1e-8

    def build_dataset(self, mode="train"):
        if mode == "train":
            return TTClassificationDataset(config_manager=self.config_manager,
                                           root=self.train_dir,
                                           mode="train"
                                           )
        elif mode == "val":
            return TTClassificationDataset(config_manager=self.config_manager,
                                           root=self.val_dir,
                                           mode="val"
                                           )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: ClassifyBatchDataInfo) -> ClassifyBatchDataInfo:
        # 调整维度以匹配输入数据
        if batch_samples.data.dim() == 4:  # [B, C, H, W]
            # 为 mean/std 添加合适的维度以便广播
            mean_tensor = self.mean_tensor.view(1, -1, 1, 1)
            std_tensor = self.std_tensor.view(1, -1, 1, 1)
        else:
            mean_tensor = self.mean_tensor
            std_tensor = self.std_tensor

        # 执行归一化
        batch_samples.data = batch_samples.data.to(self.device, non_blocking=True).float()
        batch_samples.data = (batch_samples.data / 255.0 - mean_tensor) / std_tensor

        return batch_samples

    def plot_something_before_train(self):
        """
        绘制标签统计信息图
        """
        class_names = list(self.config_manager.dataset["names"].values())
        nc = self.config_manager.dataset["nc"]

        if RANK in {-1, 0}:
            # train dataset plot
            LOGGER.info(f"Start plotting label Statistics information before training...")
            train_dataset: list[ClassifyDataInfo] = self.train_dataloader.dataset
            labels = []
            for sample in train_dataset:
                labels.append(sample.label)
            labels = np.array(labels)

            label_histogram = ClassesLabelHistogram(num_classes=nc, class_names=class_names, labels=labels, prefix="train")
            label_histogram.plot(self.save_dir)

        if RANK in {-1, 0}:
            # validation dataset plot
            val_dataset: list[ClassifyDataInfo] = self.val_dataloader.dataset
            labels = []
            for sample in val_dataset:
                labels.append(sample.label)
            labels = np.array(labels)

            label_histogram = ClassesLabelHistogram(num_classes=nc, class_names=class_names, labels=labels, prefix="val")
            label_histogram.plot(self.save_dir)
