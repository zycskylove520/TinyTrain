import numpy as np

from tinytrain.data.data_format import DetectDataInfo, DetectBatchDataInfo
from tinytrain.global_var import RANK
from tinytrain.metrics.detect_metrics import LabelInfo
from tinytrain.models.yolo.yolo_dataset import YOLODetectionDataset
from tinytrain.models.yolo.yolo_trainer import YOLOTrainer
from tinytrain.utils import LOGGER


class YOLODetectionTrainer(YOLOTrainer):
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

    def plot_something_before_train(self):
        """
        绘制标签统计信息图
        """
        if RANK in {-1, 0}:
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

            label_info = LabelInfo(num_classes=self.config_manager.dataset["nc"], class_names=class_names, labels=labels, bboxes=bboxes, max_samples=1000)
            label_info.plot(self.save_dir)