import os
import torch

from torch.utils.data import DataLoader, DistributedSampler

from TinyTrain.data.data_format import ClassifyBatchDataInfo
from TinyTrain.data.dataset import TTClassificationDataset
from TinyTrain.engine.trainer import BaseTrainer
from TinyTrain.global_var import RANK


class YOLOClassificationTrainer(BaseTrainer):
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

    def build_dataloader(self, world_size: int, mode: str = "train"):
        batch_size = self.batch_size // max(world_size, 1)

        # build dataset
        # with torch_distributed_zero_first(LOCAL_RANK):
        dataset = self.build_dataset(mode=mode)

        # build dataloader
        shuffle = True  # 分类数据集默认shuffle
        batch_size = min(batch_size, len(dataset))
        num_devices = torch.cuda.device_count()  # number of CUDA devices
        num_workers = min(os.cpu_count() // max(num_devices, 1), self.config_manager.core["workers"])  # number of workers
        sampler = None if world_size <= 1 else DistributedSampler(dataset, shuffle=shuffle)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size if mode == "train" else batch_size // 2,
            shuffle=shuffle and sampler is None,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn", None),
            generator=generator,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2
        )

    def preprocess_data(self, batch_samples: ClassifyBatchDataInfo) -> ClassifyBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        return batch_samples
