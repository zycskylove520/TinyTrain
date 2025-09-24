from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.engine import BaseModel
from tinytrain.engine.trainer import BaseTrainer
from tinytrain.global_var import LOCAL_RANK
from tinytrain.models.face.face_dataset import FaceRecognitionTrainDataset, FaceRecognitionValidDataset
from tinytrain.utils import LOGGER


class FaceRecognitionTrainer(BaseTrainer):
    def __init__(self, config_manager, device, model, callback):
        super(FaceRecognitionTrainer, self).__init__(config_manager, device, model, callback)
        self.best_threshold = 0.  # 人脸识别计算余弦相似度最佳阈值
        self.ddp_model_state_dict = None

    def build_dataset(self, mode="train"):
        if mode == "train":
            return FaceRecognitionTrainDataset(config_manager=self.config_manager,
                                               root=self.train_dir
                                               )
        elif mode == "val":
            return FaceRecognitionValidDataset(config_manager=self.config_manager,
                                               txt_files=self.val_dir,
                                               )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: ClassifyBatchDataInfo) -> ClassifyBatchDataInfo:
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        batch_samples.data = batch_samples.data.to(self.device, non_blocking=True)

        return batch_samples

    def convert_ddp_model(self, world_size: int):
        """
        转换模型成DDP模型，criterion部分不转换，PartialFCLoss在分布式情况不同device上的weight大小可能不一致。
        """
        if world_size <= 1:
            return

        # 多卡情况下：先转 SyncBN，再封装 DDP
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  # 把普通 BN → SyncBatchNorm

        for i in range(len(self.model.module_list)):
            self.model.module_list[i] = DDP(self.model.module_list[i], device_ids=[LOCAL_RANK], gradient_as_bucket_view=True)

    def get_model_instance(self, world_size: int) -> BaseModel:
        return self.model

    def save_model(self, world_size: int, current_epoch: int):
        """
        保存模型 checkpoint，包括 last.pt、best.pt、epoch_X.pt。

        Args:
            world_size (int): 分布式训练中的进程数量。
            current_epoch (int): 当前训练轮次。
        """
        # 保存当前模型的DDP模式下的参数
        self.ddp_model_state_dict = deepcopy(self.model.state_dict())

        if LOCAL_RANK not in {-1, 0}:
            return

        try:
            # 保存非DDP模型参数
            model: BaseModel = deepcopy(self.model)
            for i in range(len(model.module_list)):
                if isinstance(model.module_list[i], torch.nn.parallel.DistributedDataParallel):
                    model.module_list[i] = model.module_list[i].module
            model.eval()

            # 剔除criterion.weight参数
            new_state_dict = model.state_dict()
            new_state_dict.pop("criterion.weight")

            # 构建检查点
            checkpoint = {
                "current_epoch": current_epoch + 1,
                'model_name': self.config_manager.model["name"],
                "model": new_state_dict,
                "best_threshold": self.best_threshold,
                "optimizer": self.optimizer.state_dict(),
                "fitness": self.fitness,
                "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
                "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()}
            }

            # 保存最新的模型
            torch.save(checkpoint, self.last_pt.as_posix())

            # 保存最佳模型
            if self.best_fitness == self.fitness:
                torch.save(checkpoint, self.best_pt.as_posix())

            # 按周期保存模型（仅主进程）
            save_period = self.config_manager.core["save_period"]
            if save_period > 0 and (current_epoch + 1) % save_period == 0:
                epoch_checkpoint_path = Path(self.weight_dir / f"epoch_{current_epoch + 1}.pt")
                torch.save(checkpoint, epoch_checkpoint_path.as_posix())

        except Exception as e:
            LOGGER.error(f"Error occurred while saving model: {e}")
            raise e

    def simplified_model(self, world_size: int):
        """
        导出精简模型pt文件（如 fp16）用于部署。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        LOGGER.info(f"start export simplified model...")

        # 保存非DDP模型参数
        model: BaseModel = deepcopy(self.model)
        for i in range(len(model.module_list)):
            if isinstance(model.module_list[i], torch.nn.parallel.DistributedDataParallel):
                model.module_list[i] = model.module_list[i].module

        fp16_pt = self.config_manager.core["fp16_pt"]
        if fp16_pt:
            model = model.half()
            LOGGER.info("Simplified model converted to float16 (fp16) format.")

        model.eval()

        # 剔除criterion.weight参数
        new_state_dict = model.state_dict()
        new_state_dict.pop("criterion.weight")

        checkpoint = {
            'model_name': self.config_manager.model["name"],
            'model': new_state_dict,
            "best_threshold": self.best_threshold,
            "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
            "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()},
            "fp16": fp16_pt
        }

        torch.save(checkpoint, self.simplified_pt.as_posix())

    def load_best_model_state_dict(self, world_size, checkpoint):
        self.model.load_state_dict(self.ddp_model_state_dict)