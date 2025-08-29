import os
import torch

from torch.utils.data import DataLoader, DistributedSampler

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.data.dataset import TTClassificationDataset
from tinytrain.engine.trainer import BaseTrainer
from tinytrain.global_var import RANK


class FaceRecognitionTrainer(BaseTrainer):
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
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        self.model.head.target = batch_samples.target

        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std

        return batch_samples
