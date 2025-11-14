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

import numpy as np
import torch

from tinytrain.data.data_format import PoseBatchDataInfo
from tinytrain.engine import TTBaseTrainer
from tinytrain.engine.validator import TTBaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.detect_metrics import DetectMetrics, DetectConfusionMatrix
from tinytrain.metrics.performance_metrics import PerformanceMetrics
from tinytrain.metrics.pose_metrics import PoseImgResult
from tinytrain.utils import LOGGER
from tinytrain.utils.progress_bar import TTProgressBar
from tinytrain.utils.box_utils import kpt_iou
from tinytrain.utils.nms import detect_nms_with_keypoint


class YOLOPoseValidator(TTBaseValidator):
    def __init__(self, trainer: TTBaseTrainer):
        super().__init__(trainer)
        self.class_names = {int(k): v for k, v in self.config_manager.dataset["names"].items()}
        self.num_classes: int = self.config_manager.dataset["nc"]
        self.keypoint_shape: list[int] = self.config_manager.dataset["keypoint_shape"]

        humanoid_pose = self.keypoint_shape == [17, 3]
        self.sigma = (
                np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
                / 10.0) if humanoid_pose else np.ones(self.keypoint_shape[0]) / self.keypoint_shape[0]

        # box metrics
        self.detect_metrics = DetectMetrics(class_names=self.class_names)
        self.pose_metrics = PerformanceMetrics(prefix="Pose", class_names=self.class_names)

        # confuse matrix
        self.confuse_matrix = DetectConfusionMatrix(num_classes=self.num_classes, class_names=self.class_names)

        self.img_result = PoseImgResult(self.keypoint_shape,
                                        self.save_dir,
                                        mode="val",
                                        rgb=self.config_manager.augment["rgb"],
                                        draw_conf_threshold=self.config_manager.inference["val_draw_conf_threshold"]
                                        )

    def preprocess(self, batch_samples: PoseBatchDataInfo) -> PoseBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        batch_samples.bboxes = batch_samples.bboxes.to(self.device, non_blocking=True)
        batch_samples.bboxes_idx = batch_samples.bboxes_idx.to(self.device, non_blocking=True)
        batch_samples.batch_keypoints = batch_samples.batch_keypoints.to(self.device, non_blocking=True)
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> list[torch.Tensor]:
        # 进行nms
        outputs: list[torch.Tensor] = detect_nms_with_keypoint(pred=preds[0],
                                                               keypoint_shape=self.keypoint_shape,
                                                               conf_threshold=self.config_manager.inference["val_conf_threshold"],
                                                               nms_threshold=self.config_manager.inference["val_nms_threshold"])
        return outputs

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.reset()
        self.pose_metrics.reset()

    def update_metrics_on_training(self, outputs: list[torch.Tensor], batch_samples: PoseBatchDataInfo, pbar: TTProgressBar):
        preds = outputs

        # 真实目标解码
        self.decode_boxes(batch_samples)

        batch = len(preds)
        for i in range(batch):
            # 选出对应批次的bboxes_idx
            idx = batch_samples.bboxes_idx == i

            # 获取该批次的真实值
            gt_bboxes = batch_samples.bboxes[idx]
            gt_targets = batch_samples.target[idx]
            gt_keypoints = batch_samples.batch_keypoints[idx]
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            gt_box_areas = gt_bboxes[:, 2:].prod(1) * 0.53

            # 获取该批次的预测值
            pred = preds[i]
            pred_scores = pred[:, 4]
            pred_targets = pred[:, 5]
            pred_keypoints = pred[:, 6:].reshape(-1, self.keypoint_shape[0], self.keypoint_shape[1])

            # 计算detect metrics
            p = [pred[:, :6]]
            t = [torch.cat([gt_bboxes, gt_targets.view(-1, 1)], dim=1)]
            self.detect_metrics.update(p, t)

            # 计算pose metrics
            tp = self.calculate_tp(gt_targets.cpu().numpy(), pred_targets.cpu().numpy(), gt_keypoints, pred_keypoints, gt_box_areas, self.sigma)
            self.pose_metrics.update(tp, pred_scores.cpu().numpy(), pred_targets.cpu().numpy(), gt_targets.cpu().numpy())

        desc = f"{'val':^5}|{'classes':^15}[Detect: {'Precision':^15}|{'Recall':^15}|{'MAP50':^15}|{'MAP50_95':^15}][Pose: {'Precision':^15}|{'Recall':^15}|{'MAP50':^15}|{'MAP50_95':^15}]"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.compute()
        self.pose_metrics.compute()

        d_precision = self.detect_metrics.precision()
        d_recall = self.detect_metrics.recall()
        d_map50 = self.detect_metrics.map50()
        d_map50_95 = self.detect_metrics.map50_95()
        p_precision = self.pose_metrics.mp()
        p_recall = self.pose_metrics.mr()
        p_map50 = self.pose_metrics.map50()
        p_map50_95 = self.pose_metrics.map50_95()

        # log
        progress_str = f"{'val':^5}|{self.num_classes:^15}[Detect: {d_precision:^15.3f}|{d_recall:^15.3f}|{d_map50:^15.3f}|{d_map50_95:^15.3f}][Pose: {p_precision:^15.3f}|{p_recall:^15.3f}|{p_map50:^15.3f}|{p_map50_95:^15.3f}]"
        if RANK in {-1, 0}:
            print(progress_str)

            # metrics result
            if self.trainer.train_metrics is not None:
                self.trainer.train_metrics.add("detect_precision", d_precision)
                self.trainer.train_metrics.add("detect_recall", d_recall)
                self.trainer.train_metrics.add("detect_map50", d_map50)
                self.trainer.train_metrics.add("detect_map50_95", d_map50_95)
                self.trainer.train_metrics.add("pose_precision", p_precision)
                self.trainer.train_metrics.add("pose_recall", p_recall)
                self.trainer.train_metrics.add("pose_map50", p_map50)
                self.trainer.train_metrics.add("pose_map50_95", p_map50_95)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.confuse_matrix.reset()
        self.detect_metrics.reset()
        self.pose_metrics.reset()

        LOGGER.info(f"Calculate iou=0.5, per class precision and recall...")

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: PoseBatchDataInfo, pbar: TTProgressBar):
        preds = outputs

        # 真实目标解码
        self.decode_boxes(batch_samples)

        batch = len(preds)
        for i in range(batch):
            # 选出对应批次的bboxes_idx
            idx = batch_samples.bboxes_idx == i

            # 获取该批次的真实值
            gt_bboxes = batch_samples.bboxes[idx]
            gt_targets = batch_samples.target[idx]
            gt_keypoints = batch_samples.batch_keypoints[idx]
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            gt_box_areas = gt_bboxes[:, 2:].prod(1) * 0.53

            # 获取该批次的预测值
            pred = preds[i]
            pred_scores = pred[:, 4]
            pred_targets = pred[:, 5]
            pred_keypoints = pred[:, 6:].reshape(-1, self.keypoint_shape[0], self.keypoint_shape[1])

            # 计算detect metrics & confuse matrix
            p = [pred[:, :6]]
            t = [torch.cat([gt_bboxes, gt_targets.view(-1, 1)], dim=1)]
            self.detect_metrics.update(p, t)
            self.confuse_matrix.update(p, t)

            # 计算pose metrics
            tp = self.calculate_tp(gt_targets.cpu().numpy(), pred_targets.cpu().numpy(), gt_keypoints, pred_keypoints, gt_box_areas, self.sigma)
            self.pose_metrics.update(tp, pred_scores.cpu().numpy(), pred_targets.cpu().numpy(), gt_targets.cpu().numpy())

        # log
        desc = f"calculating per-class precision and recall..."
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.plot(batch_samples=batch_samples, preds=outputs)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.detect_metrics.compute()
        self.pose_metrics.compute()

        # log
        detect_precision_per_class = self.detect_metrics.per_class_precision().float()  # 每个类别的detect precision
        detect_recall_per_class = self.detect_metrics.per_class_recall().float()  # 每个类别的detect recall
        pose_precision_per_class = self.pose_metrics.p()  # 每个类别的pose precision
        pose_recall_per_class = self.pose_metrics.r()  # 每个类别的pose recall
        classes = self.detect_metrics.classes().int()  # 类别列表

        if RANK in {-1, 0}:
            print(f"{'val':^5}|{'class_name':^15}[Detect: {'Precision':^15}|{'Recall':^15}][Pose: {'Precision':^15}|{'Recall':^15}]")
            pr_table = torch.full((self.num_classes, 4), -1.0, dtype=torch.float32)
            if len(classes) >= 1:
                pr_table[classes, 0] = detect_precision_per_class
                pr_table[classes, 1] = detect_recall_per_class
            if pose_precision_per_class.size == classes.numel():
                pr_table[classes, 2] = torch.from_numpy(pose_precision_per_class).float()
            if pose_recall_per_class.size == classes.numel():
                pr_table[classes, 3] = torch.from_numpy(pose_recall_per_class).float()

            lines = []
            for i, pr in enumerate(pr_table):
                progress_str = f"{'val':^5}|{self.class_names[i]:^15}[Detect: {max(pr[0].item(), 0):^15.3f}|{max(pr[1].item(), 0):^15.3f}][Pose: {max(pr[2].item(), 0):^15.3f}|{max(pr[3].item(), 0):^15.3f}]"  # type: ignore[arg-type]
                lines.append(progress_str)
            print("\n".join(lines))

            # plot
            self.detect_metrics.plot(self.save_dir)
            self.pose_metrics.plot(self.save_dir)
            self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        weights = [0.05, 0.45, 0.05, 0.45]
        return (
                self.detect_metrics.map50() * weights[0] +
                self.detect_metrics.map50_95() * weights[1] +
                self.pose_metrics.map50() * weights[2] +
                self.pose_metrics.map50_95() * weights[3]
        )

    def decode_boxes(self, batch_samples: PoseBatchDataInfo):
        target_shapes = batch_samples.target_shapes.to(self.device, non_blocking=True)

        wh = target_shapes[batch_samples.bboxes_idx]  # [N, 2]
        wh4 = torch.stack([wh[:, 0], wh[:, 1], wh[:, 0], wh[:, 1]], dim=1)  # 广播到 [N,4]

        batch_samples.bboxes = batch_samples.bboxes * wh4

        # 缩放关键点
        batch_samples.batch_keypoints[..., :2] *= wh.unsqueeze(1)

    def calculate_tp(self, gt_cls, pred_cls, gt_keypoint, pred_keypoint, area, sigma):
        iou = kpt_iou(gt_keypoint, pred_keypoint, area=area, sigma=sigma)
        return self.pose_metrics.match_predictions(pred_cls, gt_cls, iou.cpu().numpy())
