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

from __future__ import annotations

import torch

from typing import TYPE_CHECKING

from tinytrain.data.data_format import DetectBatchDataInfo
from tinytrain.engine import TTBaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.detect_metrics import DetectMetrics, DetectConfusionMatrix, DetectImgResult
from tinytrain.utils.progress_bar import TTProgressBar
from tinytrain.utils.nms import detect_nms

if TYPE_CHECKING:
    from tinytrain.engine import TTBaseTrainer


class YOLODetectionValidator(TTBaseValidator):
    def __init__(self, trainer: TTBaseTrainer):
        super().__init__(trainer)
        self.num_classes = self.config_manager.dataset["nc"]
        self.class_names = {int(k): v for k, v in self.config_manager.dataset["names"].items()}

        # metrics
        self.detect_metrics = DetectMetrics(class_names=self.class_names)

        # confuse matrix
        self.confuse_matrix = DetectConfusionMatrix(num_classes=self.num_classes, class_names=self.class_names)

        self.img_result = DetectImgResult(self.save_dir,
                                          mode="val",
                                          rgb=self.config_manager.augment["rgb"],
                                          draw_conf_threshold=self.config_manager.inference["val_draw_conf_threshold"],
                                          )

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
                                                 nms_threshold=self.config_manager.inference["val_nms_threshold"])
        return outputs

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.reset()

    def update_metrics_on_training(self, outputs: list[torch.Tensor], batch_samples: DetectBatchDataInfo, pbar: TTProgressBar):
        # 真实框解码
        batch_samples.bboxes = self.decode_boxes(batch_samples)
        # 分割样本
        sample_list = self.batch_samples_split(batch_samples)

        self.detect_metrics.update(outputs, sample_list)

        desc = f"{'val':^5}|{'classes':^15}|{'Precision':^15}|{'Recall':^15}|{'MAR':^15}|{'MAP50':^15}|{'MAP50_95':^15}|{'MAP_S':^15}|{'MAP_M':^15}|{'MAP_L':^15}|"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.compute()
        precision = self.detect_metrics.precision()
        recall = self.detect_metrics.recall()
        mar_100 = self.detect_metrics.mar_100()
        map50 = self.detect_metrics.map50()
        map50_95 = self.detect_metrics.map50_95()
        map_small = self.detect_metrics.map_small()
        map_medium = self.detect_metrics.map_medium()
        map_large = self.detect_metrics.map_large()

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{self.num_classes:^15}|{precision:^15.3f}|{recall:^15.3f}|{mar_100:^15.3f}|{map50:^15.3f}|{map50_95:^15.3f}|{map_small:^15.3f}|{map_medium:^15.3f}|{map_large:^15.3f}|"
            print(progress_str)

            # metrics result
            if self.trainer.train_metrics is not None:
                self.trainer.train_metrics.add("precision", precision)
                self.trainer.train_metrics.add("recall", recall)
                self.trainer.train_metrics.add("mar", mar_100)
                self.trainer.train_metrics.add("map50", map50)
                self.trainer.train_metrics.add("map50_95", map50_95)
                self.trainer.train_metrics.add("map_small", map_small)
                self.trainer.train_metrics.add("map_medium", map_medium)
                self.trainer.train_metrics.add("map_large", map_large)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.confuse_matrix.reset()
        self.detect_metrics.reset()

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: DetectBatchDataInfo, pbar: TTProgressBar):
        # 真实框解码
        batch_samples.bboxes = self.decode_boxes(batch_samples)
        # 分割样本
        sample_list = self.batch_samples_split(batch_samples)

        self.detect_metrics.update(outputs, sample_list)
        self.confuse_matrix.update(outputs, sample_list)

        # log
        desc = f"calculating per-class precision and recall..."
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples=batch_samples, preds=outputs)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.detect_metrics.compute()

        # log
        precision_per_class = self.detect_metrics.per_class_precision().float()  # 每个类别的precision
        recall_per_class = self.detect_metrics.per_class_recall().float()  # 每个类别的recall
        classes = self.detect_metrics.classes().int()  # 类别列表

        if RANK in {-1, 0}:
            print(f"{'val':^5}|{'class_name':^15}|{'Precision':^15}|{'Recall':^15}|")
            lines = []
            pr_table = torch.full((self.num_classes, 2), -1.0, dtype=torch.float32)
            if len(classes) >= 1:
                pr_table[classes, 0] = precision_per_class
                pr_table[classes, 1] = recall_per_class

            for i, pr in enumerate(pr_table):
                progress_str = f"{'val':^5}|{self.class_names[i]:^15}|{max(pr[0].item(), 0):^15.3f}|{max(pr[1].item(), 0):^15.3f}|"  # type: ignore[arg-type]
                lines.append(progress_str)
            print("\n".join(lines))

            # plot
            self.detect_metrics.plot(self.save_dir)
            self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        weights = [0.0, 0.0, 0.1, 0.9]
        return (
                self.detect_metrics.precision() * weights[0] +
                self.detect_metrics.mar_100() * weights[1] +
                self.detect_metrics.map50() * weights[2] +
                self.detect_metrics.map50_95() * weights[3]
        )

    def decode_boxes(self, batch_samples: DetectBatchDataInfo):
        batch_boxes = batch_samples.bboxes
        target_shapes = batch_samples.target_shapes.to(self.device, non_blocking=True)
        # target_shapes[bboxes_idx] 已经对齐，直接广播乘法
        wh = target_shapes[batch_samples.bboxes_idx]  # [N, 2]
        # 构造 [w,h,w,h] 并转成同 dtype/device
        wh4 = torch.stack([wh[:, 0], wh[:, 1], wh[:, 0], wh[:, 1]], dim=1)
        return batch_boxes * wh4

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
