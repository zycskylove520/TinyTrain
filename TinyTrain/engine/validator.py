from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from TinyTrain.data.data_format import BaseBatchDataInfo

from TinyTrain.utils.TT_progress_bar import TTProgressBar
from TinyTrain.cfg.config_manager import ConfigManager
from ..global_var import RANK

if TYPE_CHECKING:
    from .trainer import BaseTrainer


class BaseValidator:
    def __init__(self, trainer: BaseTrainer, world_size: int):
        self.trainer: BaseTrainer = trainer
        self.world_size = world_size
        self.config_manager: ConfigManager = trainer.config_manager

        # device
        self.device: torch.device = trainer.device

        # dataloader
        self.val_dataloader = trainer.val_dataloader

        # callback
        self.callbacks = trainer.callbacks

    def __call__(self, *args, **kwargs):
        self.validate(*args, **kwargs)

    @torch.inference_mode()
    def validate(self):
        # 用于判断训练何时结束
        stop = self.trainer.stop

        # 获取模型
        model = self.trainer.get_model_instance(self.world_size)
        model.eval()

        self.callbacks.run_callback(self, "on_val_start")

        pbar = TTProgressBar(self.val_dataloader, total=len(self.val_dataloader), info_color="blue")

        if stop:
            self.start_metrics_on_train_completed(pbar)
        else:
            self.start_metrics_on_training(pbar)

        for i, batch_samples in enumerate(pbar):
            self.callbacks.run_callback(self, "on_val_batch_start")

            # Preprocess
            batch_samples = self.preprocess(batch_samples)  # type: ignore[arg-type]

            # Inference
            preds = model(batch_samples.data)

            # Postprocess
            outputs = self.postprocess(preds)

            if stop:
                self.update_metrics_on_train_completed(outputs, batch_samples, pbar)
            else:
                self.update_metrics_on_training(outputs, batch_samples, pbar)

            self.callbacks.run_callback(self, "on_val_batch_end")

        if stop:
            self.end_metrics_on_train_completed(pbar)
        else:
            self.end_metrics_on_training(pbar)

        fitness = self.get_fitness()
        self.callbacks.run_callback(self, "on_val_end")

        return fitness

    def preprocess(self, batch_samples: BaseBatchDataInfo) -> BaseBatchDataInfo:
        """
        对batch_samples进行数据预处理，比如将data移到对于的device上
        @param batch_samples:
        @return:
        """
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        对预测结果preds进行数据后处理
        @param preds:
        @return:
        """
        return preds

    def start_metrics_on_training(self, pbar: TTProgressBar):
        """
        训练过程中，每个epoch结束后，在进行验证前先使用该函数做metrics初始化。
        @return:
        """
        pass

    def update_metrics_on_training(self, outputs: list[torch.Tensor], batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        """
        训练过程中，每个epoch结束后，在验证过程中循环验证数据集时，使用该函数更新metrics。
        @param outputs:
        @param batch_samples:
        @param pbar:
        @return:
        """
        pass

    def end_metrics_on_training(self, pbar: TTProgressBar):
        """
        训练过程中，每个epoch结束后，在验证过程结束时，使用该函数进行最终的metrics处理。
        @return:
        """
        pass

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        """
        训练的所有epoch结束后，在验证过程中循环验证数据集时，使用该函数更新metrics。
        @return:
        """
        pass

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        """
        训练的所有epoch结束后，每个epoch结束后，在验证过程中循环验证数据集时，使用该函数更新metrics。
        @param outputs:
        @param batch_samples:
        @param pbar:
        @return:
        """
        pass

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        """
        训练的所有epoch结束后，每个epoch结束后，在验证过程结束时，使用该函数进行最终的metrics处理。
        @return:
        """
        pass

    def get_fitness(self) -> float:
        """
        用于评估验证集好坏的评估指标，要求返回一个浮点数。
        @return:
        """
        return 0
