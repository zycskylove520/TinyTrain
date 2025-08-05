import numpy as np

from tinytrain.data import PoseBatchDataInfo
from tinytrain.data.data_format import PoseDataInfo
from tinytrain.models.yolo.yolo_dataset import YOLOPoseDataset
from tinytrain.engine.trainer import BaseTrainer
from tinytrain.metrics.label_info import LabelInfo
from tinytrain.utils import LOGGER


class YOLOPoseTrainer(BaseTrainer):
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

        return batch_samples

    def plot_something_before_train(self):
        """
        绘制标签统计信息图
        """
        LOGGER.info(f"Start plotting label Statistics information before training...")
        train_samples: list[PoseDataInfo] = self.train_dataloader.dataset.samples
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
