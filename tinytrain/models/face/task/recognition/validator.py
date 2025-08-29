import torch

from tinytrain.data.data_format import ClassifyBatchDataInfo, BaseBatchDataInfo
from tinytrain.engine import BaseTrainer
from tinytrain.engine.validator import BaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.confusion_matrix import ClassifyConfusionMatrix
from tinytrain.metrics.img_result import ClassifyImgResult
from tinytrain.metrics.top_k_accuracy import ClassifyTopKAccuracy, ClassifySingleClassesAccuracy
from tinytrain.utils.TT_progress_bar import TTProgressBar


class FaceRecognitionValidator(BaseValidator):
    """
    人脸识别模型的验证器。
    """

    def __init__(self, trainer: BaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.loss_names = ["cls_loss"]
        self.num_classes = self.config_manager.dataset["nc"]
        self.class_names = list(self.config_manager.dataset["names"].values())
        self.save_dir = trainer.save_dir

        # top1 && topn accuracy
        self.top1 = ClassifyTopKAccuracy(k=1)
        self.n = self.num_classes if self.num_classes < 5 else 5
        self.topn = ClassifyTopKAccuracy(k=self.n)

        # img result
        self.img_result = ClassifyImgResult(class_names_dict=self.config_manager.dataset["names"], save_dir=self.save_dir, mode="val", rgb=self.config_manager.augment["rgb"])

    def preprocess(self, batch_samples: ClassifyBatchDataInfo) -> BaseBatchDataInfo:
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        model = self.trainer.get_model_instance(self.world_size)
        model.head.target = batch_samples.target

        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std

        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        return preds

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.top1.reset()
        self.topn.reset()

    def update_metrics_on_training(self, outputs: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo, pbar: TTProgressBar):
        # metrics update
        self.top1.update(outputs[0], batch_samples.target)
        self.topn.update(outputs[0], batch_samples.target)

        # metrics result
        top1_accuracy = self.top1.compute()
        topn_accuracy = self.topn.compute()

        # log update
        topn_acc = f"Top{self.n}_Acc"
        title = f"{'val':^5}|{'classes':^15}|{'Top1_Acc':^15}|{topn_acc:^15}|"
        desc = f"{'val':^5}|{self.num_classes:^15}|{top1_accuracy:^15.3f}|{topn_accuracy:^15.3f}|"
        pbar.set_title(title)
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        # metrics result
        top1_accuracy = self.top1.compute()
        topn_accuracy = self.topn.compute()

        if self.trainer.train_result is not None:
            self.trainer.train_result.add("top1_accuracy", top1_accuracy)
            topn_acc = f"top{self.n}_accuracy"
            self.trainer.train_result.add(topn_acc, topn_accuracy)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.top1.reset()

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo, pbar: TTProgressBar):
        pred = outputs[0].cpu()
        self.top1.update(outputs[0], batch_samples.target)
        accuracy = self.top1.compute()

        # log update
        title = f"{'val':^5}|{'classes':^15}|{'Top1_Acc':^15}|"
        desc = f"{'val':^5}|{self.num_classes:^15}|{accuracy:^15.3f}|"
        pbar.set_title(title)
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples, pred)

    def get_fitness(self) -> float:
        return (self.top1.compute() + self.topn.compute()) / 2
