import torch

from tinytrain.data import DetectBatchDataInfo
from tinytrain.engine import BaseTrainer
from tinytrain.engine.validator import BaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.box_metrics import BoxMetrics
from tinytrain.metrics.confusion_matrix import DetectConfusionMatrix
from tinytrain.metrics.img_result import YOLODetectImgResult
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.utils.nms import detect_nms


class YOLODetectionValidator(BaseValidator):
    def __init__(self, trainer: BaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.save_dir = trainer.save_dir
        self.num_classes = self.config_manager.dataset["nc"]
        self.class_names = list(self.config_manager.dataset["names"].values())

        # box metrics
        self.box_metrics = BoxMetrics(class_names=self.class_names)

        # confuse matrix
        self.confuse_matrix = DetectConfusionMatrix(num_classes=self.num_classes,
                                                    class_names=self.class_names)

        self.img_result = YOLODetectImgResult(self.save_dir, mode="val", rgb=self.config_manager.augment["rgb"], draw_conf_threshold=0.25)

    def preprocess(self, batch_samples: DetectBatchDataInfo) -> DetectBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        batch_samples.bboxes = batch_samples.bboxes.to(self.device, non_blocking=True)
        batch_samples.bboxes_idx = batch_samples.bboxes_idx.to(self.device, non_blocking=True)
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        # 进行nms
        outputs: list[torch.Tensor] = detect_nms(pred=preds[0],
                                                 conf_threshold=self.config_manager.inference["val_conf_threshold"],
                                                 nms_threshold=self.config_manager.inference["val_nms_threshold"],
                                                 max_detect_num=100)
        return outputs

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.box_metrics.reset()

    def update_metrics_on_training(self, outputs: list[torch.Tensor], batch_samples: DetectBatchDataInfo, pbar: TTProgressBar):
        # 真实框解码
        batch_samples.bboxes = self.decode_boxes(batch_samples)
        # 分割样本
        sample_list = self.batch_samples_split(batch_samples)

        self.box_metrics.update(outputs, sample_list)

        desc = f"{'val':^5}|{'classes':^15}|{'Precision':^15}|{'MAR':^15}|{'MAP50':^15}|{'MAP50_95':^15}|{'MAP_Small':^15}|{'MAP_Medium':^15}|{'MAP_Large':^15}|"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.box_metrics.compute()
        precision = self.box_metrics.precision()
        mar_100 = self.box_metrics.mar_100()
        map50 = self.box_metrics.map50()
        map50_95 = self.box_metrics.map50_95()
        map_small = self.box_metrics.map_small()
        map_medium = self.box_metrics.map_medium()
        map_large = self.box_metrics.map_large()


        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{self.num_classes:^15}|{precision:^15.3f}|{mar_100:^15.3f}|{map50:^15.3f}|{map50_95:^15.3f}|{map_small:^15.3f}|{map_medium:^15.3f}|{map_large:^15.3f}|"
            print(progress_str)

            # metrics result
            if self.trainer.train_result is not None:
                self.trainer.train_result.add("precision", precision)
                self.trainer.train_result.add("mar", mar_100)
                self.trainer.train_result.add("map50", map50)
                self.trainer.train_result.add("map50_95", map50_95)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.confuse_matrix.reset()
        self.box_metrics.reset()

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: DetectBatchDataInfo, pbar: TTProgressBar):
        # 真实框解码
        batch_samples.bboxes = self.decode_boxes(batch_samples)
        # 分割样本
        sample_list = self.batch_samples_split(batch_samples)

        self.box_metrics.update(outputs, sample_list)
        self.confuse_matrix.update(outputs, sample_list)

        # log
        desc = f"calculating per-class precision and recall..."
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples=batch_samples, preds=outputs)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.box_metrics.compute()

        # log
        precision_per_class = self.box_metrics.per_class_precision().float()  # 每个类别的precision
        recall_per_class = self.box_metrics.per_class_recall().float()  # 每个类别的recall
        classes = self.box_metrics.classes().int()  # 类别列表

        if RANK in {-1, 0}:
            print(f"{'val':^5}|{'class_name':^15}|{'Precision':^15}|{'Recall':^15}|")
            lines = []
            pr_table = torch.full((self.num_classes, 2), -1.0, dtype=torch.float32)
            pr_table[classes, 0] = precision_per_class
            pr_table[classes, 1] = recall_per_class

            for i, pr in enumerate(pr_table):
                progress_str = f"{'val':^5}|{self.class_names[i]:^15}|{max(pr[0].item(), 0):^15.3f}|{max(pr[1].item(), 0):^15.3f}|"
                lines.append(progress_str)
            print("\n".join(lines))

        # plot
        if RANK in {-1, 0}:
            self.box_metrics.plot(self.save_dir)
            self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        weights = [0.1, 0.2, 0.2, 0.5]
        return (
                self.box_metrics.precision() * weights[0] +
                self.box_metrics.mar_100() * weights[1] +
                self.box_metrics.map50() * weights[2] +
                self.box_metrics.map50_95() * weights[3]
        )

    def decode_boxes(self, batch_samples: DetectBatchDataInfo):
        batch_boxes = batch_samples.bboxes
        target_shapes = batch_samples.target_shapes.to(self.device, non_blocking=True)
        # target_shapes[bboxes_idx] 已经对齐，直接广播乘法
        target_shapes = target_shapes[batch_samples.bboxes_idx]  # [N, 2]
        # 广播到 [N,4]
        target_shapes = target_shapes.repeat_interleave(2, dim=1)
        return batch_boxes * target_shapes

    def batch_samples_split(self, batch_samples: DetectBatchDataInfo):
        """
        将一批真实标签分割成一个个组成的列表
        @return:
        """
        _, counts = torch.unique(batch_samples.bboxes_idx, return_counts=True)
        bboxes_split = torch.split(batch_samples.bboxes, counts.tolist())
        target_split = torch.split(batch_samples.target, counts.tolist())

        # 结果直接 view 堆叠，避免显式 cat
        target_list = [
            torch.cat([b, t.view(-1, 1)], dim=1)
            for b, t in zip(bboxes_split, target_split)
        ]
        return target_list
