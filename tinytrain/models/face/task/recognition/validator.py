import torch
import torch.nn.functional as F

from torch import nn

from tinytrain.data.data_format import FaceRecognitionValidBatchDataInfo
from tinytrain.engine import BaseTrainer
from tinytrain.engine.validator import BaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.face_metrics import FaceRecognitionMetrics
from tinytrain.utils.TT_progress_bar import TTProgressBar


class FaceRecognitionValidator(BaseValidator):
    def __init__(self, trainer: BaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.metrics = FaceRecognitionMetrics()

    def preprocess(self, batch_samples: FaceRecognitionValidBatchDataInfo) -> FaceRecognitionValidBatchDataInfo:
        for i in range(2):
            batch_samples.data[i] = batch_samples.data[i].to(self.device, non_blocking=True)
        batch_samples.match_tensor = batch_samples.match_tensor.to(self.device, non_blocking=True)
        return batch_samples

    def inference(self, model: nn.Module, batch_samples: FaceRecognitionValidBatchDataInfo):
        # 返回未归一化的 BN 输出即可，后处理里再统一归一化
        preds1 = model.inference(batch_samples.data[0])[0]
        preds2 = model.inference(batch_samples.data[1])[0]
        return preds1, preds2

    def postprocess(self, preds: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        pred1 = preds[0]
        pred2 = preds[1]

        pred1 = F.normalize(pred1, p=2, dim=1)
        pred2 = F.normalize(pred2, p=2, dim=1)
        return F.cosine_similarity(pred1, pred2, dim=1)  # [B]

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.metrics.reset()

    def update_metrics_on_training(self, outputs: torch.Tensor, batch_samples: FaceRecognitionValidBatchDataInfo, pbar: TTProgressBar):
        scores = outputs.detach().cpu().numpy()
        labels = batch_samples.match_tensor.cpu().numpy()
        self.metrics.update(scores, labels)

        desc = f"{'val':^5}|{'Accuracy':^15}|{'AUC':^15}|{'TPR@FAR=1e-3':^15}|{'Best_threshold':^15}|"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.metrics.compute()
        acc = self.metrics.balanced_accuracy()
        auc = self.metrics.auc()
        tpr_1e3 = self.metrics.tpr_1e3()
        best_threshold = self.metrics.best_threshold()
        self.trainer.best_threshold = best_threshold

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{acc:^15.3f}|{auc:^15.3f}|{tpr_1e3:^15.3f}|{best_threshold:^15.3f}|"
            print(progress_str)

            # 写回日志
            if self.trainer.train_result is not None:
                self.trainer.train_result.add("Accuracy", acc)
                self.trainer.train_result.add("AUC", auc)
                self.trainer.train_result.add("TPR@FAR=1e-3", tpr_1e3)
                self.trainer.train_result.add("best_threshold", best_threshold)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.metrics.reset()

    def update_metrics_on_train_completed(self, outputs: torch.Tensor, batch_samples: FaceRecognitionValidBatchDataInfo, pbar: TTProgressBar):
        scores = outputs.detach().cpu().numpy()
        labels = batch_samples.match_tensor.cpu().numpy()
        self.metrics.update(scores, labels)

        desc = f"{'val':^5}|{'Accuracy':^15}|{'AUC':^15}|{'TPR@FAR=1e-3':^15}|{'Best_threshold':^15}|"
        pbar.set_description(desc)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.metrics.compute()
        acc = self.metrics.balanced_accuracy()
        auc = self.metrics.auc()
        tpr_1e3 = self.metrics.tpr_1e3()
        best_threshold = self.metrics.best_threshold()

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{acc:^15.3f}|{auc:^15.3f}|{tpr_1e3:^15.3f}|{best_threshold:^15.3f}|"
            print(progress_str)

            # plot
            self.metrics.plot(self.save_dir)

    def get_fitness(self) -> float:
        """超参搜索时可返回 AUC 或 Best_ACC"""
        acc = self.metrics.balanced_accuracy()
        auc = self.metrics.auc()
        tpr_1e3 = self.metrics.tpr_1e3()
        weights = [0.06, 0.04, 0.9]
        return acc * weights[0] + auc * weights[1] + tpr_1e3 * weights[2]

    def get_model_instance(self):
        if self.trainer.ema:
            model = self.trainer.ema.ema_model
        else:
            model = self.trainer.model
        return model
