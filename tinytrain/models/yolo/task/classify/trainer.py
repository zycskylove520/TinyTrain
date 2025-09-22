import numpy as np

from tinytrain.data.data_format import ClassifyBatchDataInfo, ClassifyDataInfo
from tinytrain.data.dataset import TTClassificationDataset
from tinytrain.global_var import RANK
from tinytrain.metrics.classify_metrics import ClassesLabelHistogram
from tinytrain.models.yolo.yolo_trainer import YOLOTrainer
from tinytrain.utils import LOGGER


class YOLOClassificationTrainer(YOLOTrainer):
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
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
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
                print(f"sample: {sample.label}")
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
