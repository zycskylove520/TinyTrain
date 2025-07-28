import torch

from tinytrain.data.data_format import ClassifyBatchDataInfo, BaseBatchDataInfo
from tinytrain.engine import BaseTrainer
from tinytrain.engine.validator import BaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.confusion_matrix import ClassifyConfusionMatrix
from tinytrain.metrics.img_result import ClassifyImgResult
from tinytrain.metrics.top_k_accuracy import ClassifyTopKAccuracy, ClassifySingleClassesAccuracy
from tinytrain.utils.TT_progress_bar import TTProgressBar


class YOLOClassificationValidator(BaseValidator):
    """
    YOLO分类模型的验证器。
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

        # single classes accuracy
        self.single_classes_acc = ClassifySingleClassesAccuracy(num_classes=self.num_classes, classes_name=self.class_names)

        # confuse matrix
        self.confuse_matrix = ClassifyConfusionMatrix(self.num_classes, self.class_names)

        # img result
        self.img_result = ClassifyImgResult(class_names_dict=self.config_manager.dataset["names"], save_dir=self.save_dir, mode="val", rgb=self.config_manager.augment["rgb"])

    def preprocess(self, batch_samples: ClassifyBatchDataInfo) -> BaseBatchDataInfo:
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
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
        top1_accuracy = self.top1.result()
        topn_accuracy = self.topn.result()

        # log update
        topn_acc = f"Top{self.n}_Acc"
        title = f"{"val":^5}|{"classes":^15}|{"Top1_Acc":^15}|{topn_acc:^15}|"
        desc = f"{"val":^5}|{self.num_classes:^15}|{top1_accuracy:^15.3f}|{topn_accuracy:^15.3f}|"
        pbar.set_title(title)
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        # metrics result
        top1_accuracy = self.top1.result()
        topn_accuracy = self.topn.result()

        topn_acc = f"top{self.n}_accuracy"

        if self.trainer.train_result is not None:
            self.trainer.train_result.add("top1_accuracy", top1_accuracy)
            self.trainer.train_result.add(topn_acc, topn_accuracy)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.single_classes_acc.reset()
        self.confuse_matrix.reset()

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo, pbar: TTProgressBar):
        pred = outputs[0].cpu()
        true_labels = batch_samples.target.cpu()
        self.single_classes_acc.update(pred, true_labels)
        self.confuse_matrix.update(pred, true_labels)

        # log
        desc = f"{"val":^5}|{"class_name":^15}|{"Accuracy":^15}|"
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples, pred)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        # log
        acc_results = self.single_classes_acc.result()
        for i in range(self.num_classes):
            progress_str = f"{"val":^5}|{self.class_names[i]:^15}|{acc_results[i]:^15.3f}|"
            print(progress_str)

        # plot
        self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        return (self.top1.result() + self.topn.result()) / 2
