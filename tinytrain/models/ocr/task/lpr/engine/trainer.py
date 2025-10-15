from tinytrain.engine import TTBaseTrainer
from tinytrain.models.ocr.ocr_data_format import LPRBatchDataInfo
from tinytrain.models.ocr.ocr_dataset import LPRNetDataset


class LPRTrainer(TTBaseTrainer):
    def build_dataset(self, mode="train"):
        if mode == "train":
            return LPRNetDataset(config_manager=self.config_manager,
                                           img_path=self.train_dir,
                                           mode="train"
                                           )
        elif mode == "val":
            return LPRNetDataset(config_manager=self.config_manager,
                                           img_path=self.val_dir,
                                           mode="val"
                                           )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: LPRBatchDataInfo) -> LPRBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        # batch_samples.data = (batch_samples.data.to(self.device, non_blocking=True).float() - 127.5) * 0.0078125
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        return batch_samples
