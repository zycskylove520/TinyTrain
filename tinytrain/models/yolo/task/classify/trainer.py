from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.data.dataset import TTClassificationDataset
from tinytrain.models.yolo.yolo_trainer import YOLOTrainer


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
