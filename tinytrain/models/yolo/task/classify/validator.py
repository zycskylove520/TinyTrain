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
        title = f"{'val':^5}|{'classes':^15}|{'Top1_Acc':^15}|{topn_acc:^15}|"
        desc = f"{'val':^5}|{self.num_classes:^15}|{top1_accuracy:^15.3f}|{topn_accuracy:^15.3f}|"
        pbar.set_title(title)
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        # 把本地结果做成 tensor
        top1_tensor = torch.tensor(self.top1.result(), device=self.device)
        topn_tensor = torch.tensor(self.topn.result(), device=self.device)

        # 跨 rank 求平均
        self._all_reduce_mean(top1_tensor)
        self._all_reduce_mean(topn_tensor)

        # cpu 回取
        top1_accuracy = top1_tensor.item()
        topn_accuracy = topn_tensor.item()

        # 记录训练结果
        if self.trainer.train_result is not None:
            self.trainer.train_result.add("top1_accuracy", top1_accuracy)
            self.trainer.train_result.add(f"top{self.n}_accuracy", topn_accuracy)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.single_classes_acc.reset()
        self.confuse_matrix.reset()

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo, pbar: TTProgressBar):
        pred = outputs[0].cpu()
        true_labels = batch_samples.target.cpu()
        self.single_classes_acc.update(pred, true_labels)
        self.confuse_matrix.update(pred, true_labels)

        # log
        desc = f"{'val':^5}|{'class_name':^15}|{'Accuracy':^15}|"
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples, pred)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        # 单类准确率 -> tensor
        acc_tensor = torch.tensor(
            self.single_classes_acc.result(), device=self.device)  # shape [num_classes]

        # 跨 rank 求平均
        self._all_reduce_mean(acc_tensor)
        acc_results = acc_tensor.cpu().tolist()

        # 混淆矩阵 -> tensor 后累加
        cm_tensor = torch.tensor(
            self.confuse_matrix.confusion_matrix, dtype=torch.int64, device=self.device)  # shape [C,C]
        self._all_reduce_tensor(cm_tensor)  # SUM
        cm_result = cm_tensor.cpu()

        # 仅 rank0 打印 & 绘图
        if RANK in {-1, 0}:
            for i in range(self.num_classes):
                progress_str = f"{'val':^5}|{self.class_names[i]:^15}|{acc_results[i]:^15.3f}|"
                print(progress_str)

            # 更新混淆矩阵实例（用全局 cm）
            self.confuse_matrix.confusion_matrix = cm_result
            self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        top1_tensor = torch.tensor(self.top1.result(), device=self.device)
        topn_tensor = torch.tensor(self.topn.result(), device=self.device)
        self._all_reduce_mean(top1_tensor)
        self._all_reduce_mean(topn_tensor)
        return (top1_tensor.item() + topn_tensor.item()) / 2
