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
import torch.nn.functional as F

from tinytrain.data.data_format import SegmentBatchDataInfo
from tinytrain.engine import TTBaseTrainer
from tinytrain.engine.validator import TTBaseValidator
from tinytrain.global_var import RANK
from tinytrain.metrics.detect_metrics import DetectMetrics, DetectConfusionMatrix
from tinytrain.metrics.performance_metrics import PerformanceMetrics
from tinytrain.metrics.segment_metrics import SegmentImgResult
from tinytrain.utils import LOGGER
from tinytrain.utils.progress_bar import TTProgressBar
from tinytrain.utils.any_utils import make_N_tuple
from tinytrain.utils.box_utils import mask_iou, cxcywh_2_lxlyrxry
from tinytrain.utils.nms import detect_nms_with_mask
from tinytrain.utils.segment_utils import decode_pred_masks


class YOLOSegmentValidator(TTBaseValidator):
    def __init__(self, trainer: TTBaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.class_names = {int(k): v for k, v in self.config_manager.dataset["names"].items()}
        self.num_classes: int = self.config_manager.dataset["nc"]

        # box metrics
        self.detect_metrics = DetectMetrics(class_names=self.class_names)
        self.segment_metrics = PerformanceMetrics(prefix="Segment", class_names=self.class_names)

        # confuse matrix
        self.confuse_matrix = DetectConfusionMatrix(num_classes=self.num_classes, class_names=self.class_names)

        self.img_result = SegmentImgResult(make_N_tuple(self.config_manager.dataset["img_size"]),
                                           self.save_dir,
                                           mode="val",
                                           rgb=self.config_manager.augment["rgb"],
                                           draw_conf_threshold=self.config_manager.inference["val_draw_conf_threshold"]
                                           )

    def preprocess(self, batch_samples: SegmentBatchDataInfo) -> SegmentBatchDataInfo:
        # 在这里做归一化速度提升
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        batch_samples.bboxes = batch_samples.bboxes.to(self.device, non_blocking=True)
        batch_samples.bboxes_idx = batch_samples.bboxes_idx.to(self.device, non_blocking=True)
        batch_samples.batch_masks = batch_samples.batch_masks.to(self.device, non_blocking=True)
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        pred, protos = preds[0]
        # 进行nms
        p: list[torch.Tensor] = detect_nms_with_mask(pred=pred,
                                                     nc=self.num_classes,
                                                     conf_threshold=self.config_manager.inference["val_conf_threshold"],
                                                     nms_threshold=self.config_manager.inference["val_nms_threshold"]
                                                     )
        return p, protos

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.reset()
        self.segment_metrics.reset()

    def update_metrics_on_training(self, outputs: tuple[list[torch.Tensor], torch.Tensor], batch_samples: SegmentBatchDataInfo, pbar: TTProgressBar):
        preds = outputs[0]
        protos = outputs[1]

        # 真实目标解码
        self.decode_boxes(batch_samples)

        batch = len(preds)
        for i in range(batch):
            # 选出对应批次的bboxes_idx
            idx = batch_samples.bboxes_idx == i

            # 获取该批次的真实值
            gt_bboxes = batch_samples.bboxes[idx]
            gt_targets = batch_samples.target[idx]

            gt_mask = batch_samples.batch_masks[i] if self.config_manager.dataset["overlap_mask"] else batch_samples.batch_masks[idx].float()

            # 获取该批次的预测值
            pred = preds[i]
            pred_boxes = pred[:, :4]
            pred_scores = pred[:, 4].cpu().numpy()
            pred_targets = pred[:, 5].cpu().numpy()
            pred_masks_vec = pred[:, 6:]

            # 计算detect metrics
            p = [pred[:, :6]]
            t = [torch.cat([gt_bboxes, gt_targets.view(-1, 1)], dim=1)]
            self.detect_metrics.update(p, t)

            # 计算segment metrics
            if len(pred_boxes) and len(gt_bboxes):
                pred_mask = decode_pred_masks(protos[i], cxcywh_2_lxlyrxry(pred_boxes), pred_masks_vec, batch_samples.target_shapes[0])

                tp = self.calculate_tp(pred_targets, gt_targets.cpu().numpy(), pred_mask, gt_mask, self.config_manager.dataset["overlap_mask"])
                self.segment_metrics.update(tp, pred_scores, pred_targets, gt_targets.cpu().numpy())

        desc = f"{'val':^5}|{'classes':^15}[Detect: {'Precision':^15}|{'Recall':^15}|{'MAP50':^15}|{'MAP50_95':^15}][Segment: {'Precision':^15}|{'Recall':^15}|{'MAP50':^15}|{'MAP50_95':^15}]"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        self.detect_metrics.compute()
        self.segment_metrics.compute()

        d_precision = self.detect_metrics.precision()
        d_recall = self.detect_metrics.recall()
        d_map50 = self.detect_metrics.map50()
        d_map50_95 = self.detect_metrics.map50_95()
        p_precision = self.segment_metrics.mp()
        p_recall = self.segment_metrics.mr()
        p_map50 = self.segment_metrics.map50()
        p_map50_95 = self.segment_metrics.map50_95()

        # log
        progress_str = f"{'val':^5}|{self.num_classes:^15}[Detect: {d_precision:^15.3f}|{d_recall:^15.3f}|{d_map50:^15.3f}|{d_map50_95:^15.3f}][Segment: {p_precision:^15.3f}|{p_recall:^15.3f}|{p_map50:^15.3f}|{p_map50_95:^15.3f}]"
        if RANK in {-1, 0}:
            print(progress_str)

            # metrics result
            if self.trainer.train_result is not None:
                self.trainer.train_result.add("detect_precision", d_precision)
                self.trainer.train_result.add("detect_recall", d_recall)
                self.trainer.train_result.add("detect_map50", d_map50)
                self.trainer.train_result.add("detect_map50_95", d_map50_95)
                self.trainer.train_result.add("segment_precision", p_precision)
                self.trainer.train_result.add("segment_recall", p_recall)
                self.trainer.train_result.add("segment_map50", p_map50)
                self.trainer.train_result.add("segment_map50_95", p_map50_95)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.confuse_matrix.reset()
        self.detect_metrics.reset()
        self.segment_metrics.reset()

        LOGGER.info(f"Calculate iou=0.5, per class precision and recall...")

    def update_metrics_on_train_completed(self, outputs: list[torch.Tensor], batch_samples: SegmentBatchDataInfo, pbar: TTProgressBar):
        preds = outputs[0]
        protos = outputs[1]

        # 真实目标解码
        self.decode_boxes(batch_samples)

        batch = len(preds)
        for i in range(batch):
            # 选出对应批次的bboxes_idx
            idx = batch_samples.bboxes_idx == i

            # 获取该批次的真实值
            gt_bboxes = batch_samples.bboxes[idx]
            gt_targets = batch_samples.target[idx]
            gt_mask = batch_samples.batch_masks[i] if self.config_manager.dataset["overlap_mask"] else batch_samples.batch_masks[idx].float()

            # 获取该批次的预测值
            pred = preds[i]
            pred_boxes = pred[:, :4]
            pred_scores = pred[:, 4].cpu().numpy()
            pred_targets = pred[:, 5].cpu().numpy()
            pred_masks_vec = pred[:, 6:]

            # 计算detect metrics & confuse matrix
            p = [pred[:, :6]]
            t = [torch.cat([gt_bboxes, gt_targets.view(-1, 1)], dim=1)]
            self.detect_metrics.update(p, t)
            self.confuse_matrix.update(p, t)

            # 计算segment metrics
            if len(pred_boxes) and len(gt_bboxes):
                pred_mask = decode_pred_masks(protos[i], cxcywh_2_lxlyrxry(pred_boxes), pred_masks_vec, batch_samples.target_shapes[0])

                tp = self.calculate_tp(pred_targets, gt_targets.cpu().numpy(), pred_mask, gt_mask, self.config_manager.dataset["overlap_mask"])
                self.segment_metrics.update(tp, pred_scores, pred_targets, gt_targets.cpu().numpy())

                self.img_result.add_sample(batch_samples.data[i], pred, protos[i])

        # log
        desc = f"calculating per-class precision and recall..."
        pbar.set_description(desc)

        # plot
        if RANK in {-1, 0}:
            self.img_result.do_plot()

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.detect_metrics.compute()
        self.segment_metrics.compute()

        # log
        detect_precision_per_class = self.detect_metrics.per_class_precision().float()  # 每个类别的detect precision
        detect_recall_per_class = self.detect_metrics.per_class_recall().float()  # 每个类别的detect recall
        seg_precision_per_class = self.segment_metrics.p()  # 每个类别的segment precision
        seg_recall_per_class = self.segment_metrics.r()  # 每个类别的segment recall
        classes = self.detect_metrics.classes().int()  # 类别列表

        if RANK in {-1, 0}:
            print(f"{'val':^5}|{'class_name':^15}[Detect: {'Precision':^15}|{'Recall':^15}][Segment: {'Precision':^15}|{'Recall':^15}]")
            pr_table = torch.full((self.num_classes, 4), -1.0, dtype=torch.float32)
            if len(classes) >= 1:
                pr_table[classes, 0] = detect_precision_per_class
                pr_table[classes, 1] = detect_recall_per_class
            if seg_precision_per_class.size == classes.numel():
                pr_table[classes, 2] = torch.from_numpy(seg_precision_per_class).float()
            if seg_recall_per_class.size == classes.numel():
                pr_table[classes, 3] = torch.from_numpy(seg_recall_per_class).float()

            lines = []
            for i, pr in enumerate(pr_table):
                progress_str = f"{'val':^5}|{self.class_names[i]:^15}[Detect: {max(pr[0].item(), 0):^15.3f}|{max(pr[1].item(), 0):^15.3f}][Segment: {max(pr[2].item(), 0):^15.3f}|{max(pr[3].item(), 0):^15.3f}]"  # type: ignore[arg-type]
                lines.append(progress_str)
            print("\n".join(lines))

            # plot
            self.detect_metrics.plot(self.save_dir)
            self.segment_metrics.plot(self.save_dir)
            self.confuse_matrix.plot(self.save_dir)

    def get_fitness(self) -> float:
        weights = [0.05, 0.45, 0.05, 0.45]
        return (
                self.detect_metrics.map50() * weights[0] +
                self.detect_metrics.map50_95() * weights[1] +
                self.segment_metrics.map50() * weights[2] +
                self.segment_metrics.map50_95() * weights[3]
        )

    def decode_boxes(self, batch_samples: SegmentBatchDataInfo):
        # 解码box
        target_shapes = batch_samples.target_shapes.to(self.device, non_blocking=True)

        wh = target_shapes[batch_samples.bboxes_idx]  # [N, 2]
        wh4 = torch.stack([wh[:, 0], wh[:, 1], wh[:, 0], wh[:, 1]], dim=1)

        batch_samples.bboxes = batch_samples.bboxes * wh4

    def calculate_tp(self, pred_cls, gt_cls, pred_masks, gt_masks, overlap=False):
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            pred_cls (torch.Tensor | np.ndarray): Tensor of shape (N,) representing predicted class indices.
            gt_cls (torch.Tensor | np.ndarray): Tensor of shape (M,) representing ground truth class indices.
            pred_masks (torch.Tensor | None): Tensor representing predicted masks, if available. The shape should
                match the ground truth masks.
            gt_masks (torch.Tensor | None): Tensor of shape (M, H, W) representing ground truth masks, if available.
            overlap (bool): Flag indicating if overlapping masks should be considered.

        Returns:
            (np.ndarray): A correct prediction matrix of shape (N, 10), where 10 represents different IoU levels.

        Note:
            - If `overlap` is True, overlapping masks are taken into account when computing IoU.
        """
        if overlap:
            nl = len(gt_cls)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        return self.segment_metrics.match_predictions(pred_cls, gt_cls, iou.cpu().numpy())
