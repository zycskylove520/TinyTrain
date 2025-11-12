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

import torch

from copy import deepcopy
from pathlib import Path
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.engine import TTBaseModel, TTBaseModel
from tinytrain.engine.trainer import TTBaseTrainer
from tinytrain.global_var import LOCAL_RANK, WORLD_SIZE
from tinytrain.models.face.face_dataset import FaceRecognitionTrainDataset, FaceRecognitionValidDataset
from tinytrain.utils import LOGGER


class FaceRecognitionTrainer(TTBaseTrainer):
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

    def get_model_instance(self, world_size: int) -> TTBaseModel:
        if world_size > 1:
            model: TTBaseModel = deepcopy(self.model)
            for i in range(len(model.module_list)):
                if isinstance(model.module_list[i], torch.nn.parallel.DistributedDataParallel):
                    model.module_list[i] = model.module_list[i].module

            return model
        else:
            return super().get_model_instance(world_size)

    def save_model(self, world_size: int, current_epoch: int):
        """
        保存模型 checkpoint，包括 last.pt、best.pt、epoch_X.pt。

        Args:
            world_size (int): 分布式训练中的进程数量。
            current_epoch (int): 当前训练轮次。
        """
        # 保存当前模型的DDP模式下的参数
        if world_size > 1:
            self.ddp_model_state_dict = deepcopy(self.model.state_dict())

        try:
            loss_weight_dir = self.weight_dir / "loss"
            loss_weight_dir.mkdir(parents=True, exist_ok=True)

            # 保存模型参数
            model: TTBaseModel = self.get_model_instance(world_size)
            model.eval()

            # 剔除criterion.weight参数
            new_state_dict = model.state_dict()
            criterion_state_dict = new_state_dict.pop("criterion.weight")

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

            criterion_checkpoint = {
                'criterion': criterion_state_dict,  # PartialFC下不同进程的该参数shape不一致，需单独保存
            }

            # 保存最新的模型
            if LOCAL_RANK in {-1, 0}:
                torch.save(checkpoint, self.last_pt.as_posix())

            last_criterion_pt = loss_weight_dir / f"last_criterion_{LOCAL_RANK}.pt"
            torch.save(criterion_checkpoint, last_criterion_pt.as_posix())

            # 保存最佳模型
            if self.best_fitness == self.fitness:
                if LOCAL_RANK in {-1, 0}:
                    torch.save(checkpoint, self.best_pt.as_posix())

                best_criterion_pt = loss_weight_dir / f"best_criterion_{LOCAL_RANK}.pt"
                torch.save(criterion_checkpoint, best_criterion_pt.as_posix())

            # 按周期保存模型（仅主进程）
            save_period = self.config_manager.core["save_period"]
            if save_period > 0 and (current_epoch + 1) % save_period == 0:
                if LOCAL_RANK in {-1, 0}:
                    epoch_pt = self.weight_dir / f"epoch_{current_epoch + 1}.pt"
                    torch.save(checkpoint, epoch_pt.as_posix())

                epoch_criterion_pt = loss_weight_dir / f"epoch_criterion_{current_epoch + 1}_{LOCAL_RANK}.pt"
                torch.save(criterion_checkpoint, epoch_criterion_pt.as_posix())

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
        model: TTBaseModel = self.get_model_instance(world_size)
        model.eval()

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

    def load_extra_save_params(self, model: TTBaseModel) -> None:
        """
        加载额外保存的参数，如 criterion.weight。

        Args:
            model (TTBaseModel): 模型实例。
        """

        # 验证模式下不要加载额外参数
        if self.config_manager.core["only_val"]:
            return

        model_pt: Path = self.config_manager.link["model"]
        model_pt_stem = model_pt.stem

        loss_weight_dir = model_pt.parent / "loss"
        loss_weight_dir.mkdir(parents=True, exist_ok=True)

        # 根据不同的权重文件类型加载criterion weight
        if model_pt_stem.startswith("best"):
            # 加载最佳模型对应的 criterion.weight
            name = f"best_criterion_{LOCAL_RANK}.pt"
            criterion_pt = loss_weight_dir / name

            if not criterion_pt.exists():
                LOGGER.warning(f"{name} does not exist, LOCAL_RANK: {LOCAL_RANK} skip loading extra parameters.")
                return

            ckpt = torch.load(criterion_pt.as_posix(), map_location="cpu", weights_only=False)
            criterion_state_dict = {'weight': ckpt['criterion']}
            model.criterion.load_state_dict(criterion_state_dict, strict=True)

        elif model_pt_stem.startswith("last"):
            # 加载最新模型对应的 criterion.weight
            name = f"last_criterion_{LOCAL_RANK}.pt"
            criterion_pt = loss_weight_dir / name

            if not criterion_pt.exists():
                LOGGER.warning(f"{name} does not exist, LOCAL_RANK: {LOCAL_RANK} skip loading extra parameters.")
                return

            ckpt = torch.load(criterion_pt.as_posix(), map_location="cpu", weights_only=False)
            criterion_state_dict = {'weight': ckpt['criterion']}
            model.criterion.load_state_dict(criterion_state_dict, strict=True)

        elif model_pt_stem.startswith("epoch_criterion_"):
            # 加载按周期保存的模型对应的 criterion.weight
            epoch_num = model_pt_stem.split('_')[-1]
            name = f"epoch_criterion_{epoch_num}_{LOCAL_RANK}.pt"
            criterion_pt = loss_weight_dir / name

            if not criterion_pt.exists():
                LOGGER.warning(f"{name} does not exist, LOCAL_RANK: {LOCAL_RANK} skip loading extra parameters.")
                return

            ckpt = torch.load(criterion_pt.as_posix(), map_location="cpu", weights_only=False)
            criterion_state_dict = {'weight': ckpt['criterion']}
            model.criterion.load_state_dict(criterion_state_dict, strict=True)

        else:
            # 如果权重文件类型不匹配，跳过加载
            LOGGER.warning(f"Unknown model file type: {model_pt_stem}, skip loading extra parameters.")
            return

    def load_model_to_final_eval(self, world_size, checkpoint):
        if world_size > 1:
            self.model.load_state_dict(self.ddp_model_state_dict)
        else:
            super().load_model_to_final_eval(world_size, checkpoint)
