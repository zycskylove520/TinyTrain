import torch
from torch import nn

from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.engine import TTBaseTrainer
from tinytrain.engine.validator import TTBaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.face_metrics import FaceRecognitionMetrics
from tinytrain.utils.progress_bar import TTProgressBar


class FaceRecognitionValidator(TTBaseValidator):
    def __init__(self, trainer: TTBaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.metrics = FaceRecognitionMetrics()

    def preprocess(self, batch_samples: BaseBatchDataInfo) -> BaseBatchDataInfo:
        (images1, images2), (flip_images1, flip_images2), match_tensor = batch_samples.data

        # 1. 全部搬过去
        images1 = images1.to(self.device, non_blocking=True)
        images2 = images2.to(self.device, non_blocking=True)
        flip_images1 = flip_images1.to(self.device, non_blocking=True)
        flip_images2 = flip_images2.to(self.device, non_blocking=True)
        match_tensor = match_tensor.to(self.device, non_blocking=True)

        # 2. 重新打包成原来的结构返回
        batch_samples.data = ((images1, images2), (flip_images1, flip_images2), match_tensor)
        return batch_samples

    def inference(self, model: nn.Module, batch_samples: BaseBatchDataInfo):
        (images1, images2), (flip_images1, flip_images2), match_tensor = batch_samples.data

        # 合并成一次推理，加快效率，减小抖动
        x = torch.cat([images1, images2, flip_images1, flip_images2], dim=0)  # 4b

        feat = model.inference(x)[0]

        pred1, pred2, flip_pred1, flip_pred2 = torch.chunk(feat, 4, dim=0)

        # TTA（Test-Time Augmentation）策略：降低单张图片因姿态/光照不对称带来的方差，提升稳定性。
        return (pred1 + flip_pred1), (pred2 + flip_pred2), match_tensor

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.metrics.reset()

    def update_metrics_on_training(self, outputs: torch.Tensor, batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        fuse_pred1, fuse_pred2, match_tensor = outputs

        # 归一化
        fuse_pred1 = torch.nn.functional.normalize(fuse_pred1, dim=-1)
        fuse_pred2 = torch.nn.functional.normalize(fuse_pred2, dim=-1)
        self.metrics.update(fuse_pred1.cpu().numpy(), fuse_pred2.cpu().numpy(), match_tensor.cpu().numpy())

        desc = f"{'val':^5}|{'Accuracy':^15}|{'TPR@FAR=1e-3':^15}|{'TPR@FAR=1e-4':^15}|{'Best_threshold':^15}|"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.metrics.compute()

        acc = self.metrics.avg_accuracy()
        tpr_1e3 = self.metrics.tpr_at_far1e3()
        tpr_1e4 = self.metrics.tpr_at_far1e4()
        best_threshold = self.metrics.best_threshold()
        self.trainer.best_threshold = best_threshold

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{acc:^15.4f}|{tpr_1e3:^15.4f}|{tpr_1e4:^15.4f}|{best_threshold:^15.4f}|"
            print(progress_str)

            # 写回日志
            if self.trainer.train_result is not None:
                self.trainer.train_result.add("Accuracy", acc)
                self.trainer.train_result.add("TPR@FAR=1e-3", tpr_1e3)
                self.trainer.train_result.add("TPR@FAR=1e-3", tpr_1e4)
                self.trainer.train_result.add("best_threshold", best_threshold)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.metrics.reset()

    def update_metrics_on_train_completed(self, outputs: torch.Tensor, batch_samples: BaseBatchDataInfo, pbar: TTProgressBar):
        self.update_metrics_on_training(outputs, batch_samples, pbar)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.metrics.compute()

        acc = self.metrics.avg_accuracy()
        tpr_1e3 = self.metrics.tpr_at_far1e3()
        tpr_1e4 = self.metrics.tpr_at_far1e4()
        best_threshold = self.metrics.best_threshold()
        self.trainer.best_threshold = best_threshold

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{acc:^15.4f}|{tpr_1e3:^15.4f}|{tpr_1e4:^15.4f}|{best_threshold:^15.4f}|"
            print(progress_str)

            # plot
            self.metrics.plot(self.save_dir)

    def get_fitness(self) -> float:
        """超参搜索时可返回 AUC 或 Best_ACC"""
        acc = self.metrics.avg_accuracy()
        tpr_1e3 = self.metrics.tpr_at_far1e3()
        tpr_1e4 = self.metrics.tpr_at_far1e4()

        weights = [0.1, 0.4, 0.5]
        return acc * weights[0] + tpr_1e3 * weights[1] + tpr_1e4 * weights[2]

    def get_model_instance(self):
        if self.trainer.ema:
            model = self.trainer.ema.ema_model
        else:
            model = self.trainer.model
        return model
