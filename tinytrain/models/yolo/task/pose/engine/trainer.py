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

from tinytrain.data.data_format import PoseBatchDataInfo, PoseDataInfo
from tinytrain.global_var import RANK
from tinytrain.metrics.classify_metrics import ClassesLabelHistogram
from tinytrain.metrics.detect_metrics import DetectLabelInfo
from tinytrain.models.yolo.yolo_dataset import YOLOPoseDataset
from tinytrain.models.yolo.yolo_trainer import YOLOTrainer
from tinytrain.utils import LOGGER


class YOLOPoseTrainer(YOLOTrainer):
    def build_dataset(self, mode="train"):
        if mode == "train":
            return YOLOPoseDataset(config_manager=self.config_manager,
                                   img_path=self.train_dir,
                                   mode="train"
                                   )
        elif mode == "val":
            return YOLOPoseDataset(config_manager=self.config_manager,
                                   img_path=self.val_dir,
                                   mode="val"
                                   )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: PoseBatchDataInfo) -> PoseBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        batch_samples.bboxes = batch_samples.bboxes.to(self.device, non_blocking=True)
        batch_samples.bboxes_idx = batch_samples.bboxes_idx.to(self.device, non_blocking=True)
        batch_samples.batch_keypoints = batch_samples.batch_keypoints.to(self.device, non_blocking=True)
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
            train_samples: list[PoseDataInfo] = self.train_dataloader.dataset.samples
            labels = []
            bboxes = []
            for sample in train_samples:
                labels.append(sample.label)
                bboxes.append(sample.bboxes)
            labels = np.concatenate(labels, axis=0)
            bboxes = np.concatenate(bboxes, axis=0)

            label_histogram = ClassesLabelHistogram(num_classes=nc, class_names=class_names, labels=labels, prefix="train")
            label_histogram.plot(self.save_dir)

            label_info = DetectLabelInfo(num_classes=nc, class_names=class_names, labels=labels, bboxes=bboxes, max_samples=300, prefix="train")
            label_info.plot(self.save_dir)

        if RANK in {-1, 0}:
            # validation dataset plot
            val_samples: list[PoseDataInfo] = self.val_dataloader.dataset.samples
            labels = []
            bboxes = []
            for sample in val_samples:
                labels.append(sample.label)
                bboxes.append(sample.bboxes)
            labels = np.concatenate(labels, axis=0)
            bboxes = np.concatenate(bboxes, axis=0)

            label_histogram = ClassesLabelHistogram(num_classes=nc, class_names=class_names, labels=labels, prefix="val")
            label_histogram.plot(self.save_dir)

            label_info = DetectLabelInfo(num_classes=nc, class_names=class_names, labels=labels, bboxes=bboxes, max_samples=300, prefix="val")
            label_info.plot(self.save_dir)