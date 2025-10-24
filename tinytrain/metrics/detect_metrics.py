import cv2
import seaborn as sns
import torchmetrics
import pandas as pd
import torch.distributed as dist
import numpy as np
import torch

from typing import Optional
from pathlib import Path
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from tinytrain.global_var import WORLD_SIZE, RANK
from tinytrain.utils import LOGGER
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry

from .base import TTBaseMetric, BaseImgResult


class DetectMetrics(TTBaseMetric):
    """
    检测任务统一评估指标封装类，基于 torchmetrics.detection.MeanAveragePrecision。

    功能
    ----
    1. 支持 mAP@0.5、mAP@0.75、mAP@[0.5:0.95] 等常用指标；
    2. 支持小/中/大目标 mAP 与 mAR；
    3. 支持 IoU-Recall 曲线与 PR 曲线绘制；
    4. 支持类别级指标输出、可视化；
    5. 支持缓存与重置，方便训练/验证循环。
    """

    def __init__(self, class_metrics: bool = False, class_names: dict[int, str] = None):
        """
        Args:
            class_metrics (bool):
                是否计算类别级指标。
            class_names (dict[int, str] | None):
                类别名称列表，用于可视化时替换索引。
        """
        super(DetectMetrics, self).__init__()
        self.class_metrics = class_metrics
        self.class_names = class_names

        self.metrics = torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh',
                                                                   iou_type='bbox',
                                                                   class_metrics=self.class_metrics,
                                                                   extended_summary=True,
                                                                   sync_on_compute=False  # 必须取消同步，使用单卡验证
                                                                   )
        self.results = None

        self.nc = len(self.class_names) if class_names else 1
        # recall_curve的维度为：(TxK), 其中T是 IoU 阈值的数量，K是类别数量
        self.recall_curve = torch.zeros((10, self.nc), dtype=torch.float32)
        # precision_curve的维度为：(TxRxK), 其中T是 IoU 阈值的数量，R是召回率阈值的数量，K是类别数量
        self.precision_curve = torch.zeros((10, 101, self.nc), dtype=torch.float32)

    def reset(self):
        """重置内部状态，开始新一轮评估。"""
        self.results = None
        self.recall_curve = torch.zeros((10, self.nc), dtype=torch.float32)
        self.precision_curve = torch.zeros((10, 101, self.nc), dtype=torch.float32)
        self.metrics.reset()

    def update(self, pred: list[torch.Tensor], target: list[torch.Tensor]):
        """
        更新指标（每批调用一次）。

        Args:
            pred (list[Tensor]): 模型输出（NMS 后）。
                每个元素形状 [N, 6] → (x1, y1, x2, y2, score, class_idx)。
            target (list[Tensor]): 真实标签。
                每个元素形状 [M, 5] → (x1, y1, x2, y2, class_idx)。
        """
        pred_list = []
        target_list = []

        for p, t in zip(pred, target):
            # 空预测时构造空张量
            if p.shape[0] == 0:
                pred_list.append({
                    "boxes": torch.empty((0, 4), device=p.device, dtype=torch.float32),
                    "scores": torch.empty(0, device=p.device, dtype=torch.float32),
                    "labels": torch.empty(0, device=p.device, dtype=torch.int64),
                })
            else:
                pred_list.append({
                    "boxes": p[:, :4],
                    "scores": p[:, 4],
                    "labels": p[:, 5].long(),
                })

            if t.shape[0] == 0:
                target_list.append({
                    "boxes": torch.empty((0, 4), device=p.device, dtype=torch.float32),
                    "labels": torch.empty(0, device=p.device, dtype=torch.int64),
                })
            else:
                target_list.append({
                    "boxes": t[:, :4],
                    "labels": t[:, 4].long(),
                })

        self.metrics.update(pred_list, target_list)

    def compute(self):
        """计算最终指标，并缓存曲线数据。"""
        # --------------------------------------------------------
        # 1. 只在 rank0 上真正计算
        # --------------------------------------------------------
        if RANK in {-1, 0}:
            self.results = self.metrics.compute()
            results_to_send = self.results
        else:
            results_to_send = None

        # --------------------------------------------------------
        # 2. 把 results（dict）同步给所有 rank
        #    由于 dict 里都是 tensor，可以直接 broadcast_object_list
        # --------------------------------------------------------
        if WORLD_SIZE > 1:
            # list 长度必须为 1
            obj_list = [results_to_send]
            dist.broadcast_object_list(obj_list, src=0)
            self.results = obj_list[0]

        # curve
        # all_recall_matrix的维度为：(TxKxAxM)
        # 其中T是 IoU 阈值的数量，K是类别数量， A是区域数量，M是每幅图像的最大检测数量。
        # 区域数量：默认四个区域，area=all、area=small、area=medium和area=large
        # 最大检测数量：默认三个检测数量：1，10，100
        all_recall_matrix = self.results.get("recall")
        # 只取area=all，并取最大检测数量=100
        if all_recall_matrix is not None and all_recall_matrix.numel() > 0:
            self.recall_curve = all_recall_matrix[:, :, 0, 2]

        # all_precision_matrix的维度为：(TxRxKxAxM)
        # 其中T是 IoU 阈值的数量，R是置信度阈值的数量，K是类别数量, A是区域数量，M是每幅图像的最大检测数量。
        # 区域数量：默认四个区域，area=all、area=small、area=medium和area=large
        # 最大检测数量：默认三个检测数量：1，10，100
        all_precision_matrix = self.results.get("precision")
        # 只取area=all，并取最大检测数量=100
        if all_precision_matrix is not None and all_precision_matrix.numel() > 0:
            self.precision_curve = all_precision_matrix[:, :, :, 0, 2]

    def map50(self):
        """
        返回模型在 IoU 阈值为 0.5 时的平均精度（mAP@0.5）。

        mAP@0.5 是目标检测中常用的评估指标，表示在 IoU 阈值为 0.5 时的平均精度。
        该指标衡量模型在不同类别上的平均检测性能。

        Returns:
            float: mAP@0.5 的值。如果 self.results 为空，则返回 0.0。
        """
        return max(self.results["map_50"].item(), 0.) if self.results else 0.

    def map75(self):
        """
        返回模型在 IoU 阈值为 0.75 时的平均精度（mAP@0.75）。

        mAP@0.75 是一个更严格的评估指标，表示在 IoU 阈值为 0.75 时的平均精度。
        该指标衡量模型在更高重叠度下的检测性能。

        Returns:
            float: mAP@0.75 的值。如果 self.results 为空，则返回 0.0。
        """
        return max(self.results["map_75"].item(), 0.) if self.results else 0.

    def map50_95(self):
        """
        返回模型在 IoU 阈值范围 [0.5, 0.95] 内的平均精度（mAP@[0.5:0.95]）。

        mAP@[0.5:0.95] 是一个综合评估指标，表示在 IoU 阈值从 0.5 到 0.95（步长为 0.05）的平均精度。
        该指标衡量模型在不同重叠度下的整体检测性能。

        Returns:
            float: mAP@[0.5:0.95] 的值。如果 self.results 为空，则返回 0.0。
        """
        return max(self.results["map"].item(), 0.) if self.results else 0.

    def map_small(self):
        """小目标 mAP"""
        return max(self.results["map_small"].item(), 0.) if self.results else 0.

    def map_medium(self):
        """中目标 mAP"""
        return max(self.results["map_medium"].item(), 0.) if self.results else 0.

    def map_large(self):
        """大目标 mAP"""
        return max(self.results["map_large"].item(), 0.) if self.results else 0.

    def mar_1(self):
        """mAR@1"""
        return max(self.results[f"mar_1"].item(), 0.) if self.results else 0.

    def mar_10(self):
        """mAR@10"""
        return max(self.results[f"mar_10"].item(), 0.) if self.results else 0.

    def mar_100(self):
        """
        返回模型在最大检测数为 100 时的平均召回率（mAR@100）。

        mAR@100 是目标检测中的一个评估指标，表示在每个图像中最多检测 100 个目标时的平均召回率。
        该指标衡量模型在固定检测数下的召回能力。

        Returns:
            float: mAR@100 的值。如果 self.results 为空，则返回 0.0。
        """
        return max(self.results[f"mar_100"].item(), 0.) if self.results else 0.

    def per_class_recall(self):
        """类别级 Recall@0.5"""
        # 只计算每个类别在iou=0.5的情况下的recall
        LOGGER.info(f"Calculate iou=0.5, per class recall.")
        return self.recall_curve[0, :]

    def recall(self):
        """
        在 iou=0.5 下的总体 Recall
        """
        return max(self.recall_curve[0, :].mean().item(), 0.) if self.results else 0.

    def per_class_precision(self, conf_threshold=0.25):
        """类别级 Precision@conf"""
        # 只计算每个类别在iou=0.5,conf=conf_threshold的情况下的precision
        LOGGER.info(f"Calculate conf={conf_threshold} & iou=0.5, per class precision.")
        conf = int(conf_threshold * 100)
        return self.precision_curve[0, conf, :]

    def precision(self, conf_threshold=0.25):
        """在 iou=0.5,conf=conf_threshold 下的总体 Precision"""
        conf = int(conf_threshold * 100)
        return max(self.precision_curve[0, conf, :].mean().item(), 0.) if self.results else 0.

    def classes(self):
        """
        返回检测到的类别索引列表
        """
        if self.results:
            return self.results["classes"].reshape(-1)
        else:
            raise ValueError("box metrics no classes!")

    def plot_recall_curve(self, save_dir: Path):
        """
        绘制 Recall vs IoU 曲线：
        单个类别 → 只画 1 条线（不额外画 Mean）
        多个类别 → 每类别一条细线 + 一条 Mean
        """
        if self.results is None or self.recall_curve.numel() == 0:
            LOGGER.warning("No detection results available, skipping Recall-IoU curve.")
            return

        # 如果类别为空，也跳过
        if len(self.classes()) == 0:
            LOGGER.warning("No classes detected, skipping Recall-IoU curve.")
            return

        # 类别名
        if self.class_names is None:
            class_labels = [f'Class {i}' for i in self.classes()]
        else:
            class_labels = [self.class_names[i.item()] for i in self.classes()]
        n_classes = len(class_labels)

        # 组装 DataFrame
        df = pd.DataFrame(
            self.recall_curve.numpy(),
            columns=class_labels
        )
        df['IoU'] = np.array(self.metrics.iou_thresholds)
        df_long = pd.melt(df,
                          id_vars=['IoU'],
                          var_name='Class',
                          value_name='Recall')

        plt.figure(figsize=(10, 6))

        # 如果只有一个类别，只画类别线，不再画 Mean
        if n_classes == 1:
            sns.lineplot(data=df_long,
                         x='IoU',
                         y='Recall',
                         color='tab:blue',
                         linewidth=2.5,
                         label=class_labels[0])
        else:
            # 类别曲线（细线）
            sns.lineplot(data=df_long,
                         x='IoU',
                         y='Recall',
                         hue='Class',
                         palette='tab20',
                         linewidth=1.2,
                         legend=False)
            # 平均曲线（粗黑线）
            mean_recall = self.recall_curve.mean(dim=1).numpy()
            mean_df = pd.DataFrame({'IoU': np.array(self.metrics.iou_thresholds),
                                    'Recall': mean_recall,
                                    'Class': 'Mean'})
            sns.lineplot(data=mean_df,
                         x='IoU',
                         y='Recall',
                         color='black',
                         linewidth=3,
                         label='Mean')

            # 图例：类别 + Mean
            handles, labels = plt.gca().get_legend_handles_labels()
            handles = handles[-1:] + handles[:-1]
            labels = labels[-1:] + labels[:-1]
            plt.legend(handles, labels,
                       title='Class',
                       loc='upper left',
                       bbox_to_anchor=(1.02, 1),
                       fontsize=8)

        plt.title('Recall vs IoU Curve')
        plt.xlabel('IoU Threshold')
        plt.ylabel('Recall')
        plt.tight_layout()
        plt.savefig(save_dir / '(Detect)R_Curve.png', dpi=150)
        plt.close()

    def plot_pr_curve(self, save_dir: Path):
        """
        绘制 PR 曲线（IoU = 0.5, 0.75, 0.95）
        """
        if self.results is None or self.precision_curve.numel() == 0:
            LOGGER.warning("No detection results available, skipping PR curve.")
            return

        if len(self.classes()) == 0:
            LOGGER.warning("No classes detected, skipping PR curve.")
            return

        target_iou_vals = [0.5, 0.75, 0.95]
        idx_list = [0, 5, 9]  # precision_curve 对应下标

        dfs = []
        for iou, idx in zip(target_iou_vals, idx_list):
            # 在类别维度上取平均
            prec = self.precision_curve[idx].mean(dim=1).numpy()  # shape [R]
            df = pd.DataFrame({'Recall': np.array(self.metrics.rec_thresholds),
                               'Precision': prec,
                               'IoU': iou})
            dfs.append(df)

        data_long = pd.concat(dfs, ignore_index=True)

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=data_long,
                     x='Recall',
                     y='Precision',
                     hue='IoU',
                     palette='tab10',
                     linewidth=2.5)

        plt.title('Precision-Recall Curve (IoU=0.5/0.75/0.95)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(title='IoU')
        plt.tight_layout()
        plt.savefig(save_dir / '(Detect)PR_Curve.png', dpi=150)
        plt.close()

    def plot(self, save_dir: Path):
        """一键绘制所有曲线。"""
        self.plot_recall_curve(save_dir)
        self.plot_pr_curve(save_dir)


class DetectConfusionMatrix:
    """
    检测任务专用混淆矩阵工具。

    功能
    ----
    1. 以 IoU 阈值为准，将预测框匹配到 GT 框，统计 TP / FP / FN。
    2. 额外引入“背景”类别，用于表示漏检（FN）与误检（FP）。
    3. 支持重置、增量更新、绘制原始/归一化矩阵。
    """

    def __init__(self, num_classes: int, class_names: dict[int, str], conf_threshold=0.25, iou_threshold=0.45):
        """
        Args:
            num_classes (int): 前景类别数量。
            class_names (dict[int, str]): 类别名称列表，顺序必须与索引对齐。
            conf_threshold (float): 置信度阈值，低于阈值的预测框丢弃。
            iou_threshold (float): IoU 阈值，大于该值视为匹配成功。
        """
        self.num_classes = num_classes
        self.class_names = class_names.copy()
        self.class_names[len(self.class_names)] = "background"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 混淆矩阵, num_classes + 1是因为包含背景, x轴是预测类别, y轴是真实类别
        self.confusion_matrix = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64)

    def reset(self):
        """将矩阵清零。"""
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)

    def update(self, pred: list[torch.Tensor], target: list[torch.Tensor]):
        """
        更新混淆矩阵（按 batch 循环调用）。

        Args:
            pred (list[Tensor]): NMS 后结果，每元素形状 [N, 6]：
                (cx, cy, w, h, score, class_idx)。
            target (list[Tensor]): 真实标签，每元素形状 [M, 5]：
                (cx, cy, w, h, class_idx)。
        """
        for p, t in zip(pred, target):
            # 判断t是否为空tensor，即背景图片没有真实标签
            if t.shape[0] == 0:
                # 模型预测经过nms后还有框
                if p.shape[0] != 0:
                    # 置信度过滤
                    p = p[p[:, 4] > self.conf_threshold]
                    detection_classes = p[:, 5].int()
                    for dc in detection_classes:
                        self.confusion_matrix[self.num_classes, dc] += 1  # false positives FP
                continue

            # 图片有真实标签
            gt_classes = t[:, 4].int()

            # 判断p是否为空tensor，即通过nms过滤后没有框
            if p.shape[0] == 0:
                for gc in gt_classes:
                    self.confusion_matrix[gc, self.num_classes] += 1  # background FN
                continue

            # p不为空，t也不为空
            # 置信度过滤
            p = p[p[:, 4] > self.conf_threshold]
            if p.shape[0] == 0:
                # 把当前图片所有 GT 记为 FN
                for gc in gt_classes:
                    self.confusion_matrix[gc, self.num_classes] += 1
                continue

            detection_classes = p[:, 5].int()

            # 将box从cxcywh转为lxlyrxry格式
            t[:, :4] = cxcywh_2_lxlyrxry(t[:, :4])
            p[:, :4] = cxcywh_2_lxlyrxry(p[:, :4])
            # 计算 iou 矩阵
            iou = box_iou(t[:, :4], p[:, :4])

            # 将 iou 转换为成本矩阵
            cost_matrix = -iou.cpu().numpy()  # 转换为最小化问题

            # 使用 linear_sum_assignment 求解最优匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 更新混淆矩阵
            for i, j in zip(row_ind, col_ind):
                if iou[i, j] > self.iou_threshold:
                    gc = gt_classes[i].item()
                    dc = detection_classes[j].item()
                    self.confusion_matrix[gc, dc] += 1  # TP
                else:
                    gc = gt_classes[i].item()
                    self.confusion_matrix[gc, self.num_classes] += 1  # background FN

            # 剩余的未匹配的预测框通通记为预测成了背景
            unmatched_pred = [j for j in range(p.shape[0]) if j not in col_ind]
            for j in unmatched_pred:
                dc = detection_classes[j].item()
                self.confusion_matrix[self.num_classes, dc] += 1  # FP

    def plot(self, save_dir: Path):
        """
        绘制并保存原始、归一化两份混淆矩阵图。

        Args:
            save_dir (Path): 保存目录。
        """
        LOGGER.info(f"plotting confusion matrix...")
        confusion_matrix = self.confusion_matrix.numpy()

        class_names = list(self.class_names.values())
        # 绘制混淆矩阵
        # 根据标签数量自适应图表宽度
        width = height = self.num_classes + 1
        plt.figure(figsize=(width * 2, height * 2))  # 设置图表大小
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # 设置x轴标签竖着排，y轴标签横着排
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(save_dir / 'confusion_matrix.png')  # 保存为 PNG 文件
        plt.close()  # 关闭图像窗口，释放资源

        # 绘制归一化后的混淆矩阵
        # 按行归一化（真实类别为行）
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-5)

        plt.figure(figsize=(width * 2, height * 2))  # 设置图表大小
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')

        # 设置x轴标签竖着排，y轴标签横着排
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(save_dir / 'confusion_matrix_normalized.png')  # 保存为 PNG 文件
        plt.close()  # 关闭图像窗口，释放资源


class DetectLabelInfo:
    """
    目标检测数据集标签统计与可视化工具类。

    功能
    ----
    1. 边界框空间分布热力图 (`boxes_statistics`)
    2. 中心点及宽高分布散点图 (`cxcy_and_wh`)
    3. 一键保存所有图表 (`plot`)
    """

    def __init__(self, num_classes: int, class_names: list[str], labels: np.ndarray, bboxes: np.ndarray, max_samples: Optional[int] = None, prefix:str="train"):
        """
        Args:
            num_classes (int): 类别总数。
            class_names (list[str]): 与索引对应的类别名称列表。
            labels (np.ndarray): 一维数组，长度 = 边界框数量，存储每个框的类别索引。
            bboxes (np.ndarray): 二维数组，形状 [box_num, 4]，格式为 cxcywh（已归一化到 [0,1]）。
            prefix (str): 保存图表名称前缀。
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.prefix = prefix

        # 抽样逻辑
        if max_samples is not None and len(labels) > max_samples:
            rng = np.random.default_rng(42)  # 固定随机种子，结果可复现
            idx = rng.choice(len(labels), size=max_samples, replace=False)
            self.labels = labels[idx]
            self.bboxes = bboxes[idx]
        else:
            self.labels = labels
            self.bboxes = bboxes

        self.max_samples = max_samples

    def boxes_statistics(self, axes):
        """
        在 640×640 画布上绘制所有边界框的叠加热力图。

        参数
        ----
        axes: matplotlib Axes 对象，用于子图绘制。
        """

        # 默认绘制在640x640的大小的图上
        width, height = 640, 640
        image = np.full((height, width, 3), fill_value=255, dtype=np.uint8)

        # 图像中心点
        center_x, center_y = width // 2, height // 2
        for bbox in self.bboxes:
            cx, cy, w, h = bbox
            # 将框的中心点移动到图像中心
            x_min = int(center_x - (w / 2) * width)
            y_min = int(center_y - (h / 2) * height)
            x_max = int(center_x + (w / 2) * width)
            y_max = int(center_y + (h / 2) * height)

            # 生成随机颜色
            color = (np.random.randint(0, 201), np.random.randint(0, 201), np.random.randint(0, 201))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=1)

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Boxes Statistics')
        axes[0, 0].axis('off')

    def cxcy_and_wh(self, axes):
        """
        绘制两类散点图：
        1. 中心点 (cx, cy) 分布
        2. 宽高 (w, h) 分布

        参数
        ----
        axes: 长度为 2 的 Axes 列表，分别用于两幅子图。
        """
        data1 = {
            "cx": [],
            "cy": [],
            "label": []
        }
        data2 = {
            "width": [],
            "height": [],
            "label": []
        }
        for bbox, label in zip(self.bboxes, self.labels):
            cx, cy, w, h = bbox
            data1["cx"].append(cx)
            data1["cy"].append(cy)
            data1["label"].append(label)
            data2["width"].append(w)
            data2["height"].append(h)
            data2["label"].append(label)

        # 转换为 Pandas DataFrame
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # 绘制散点图
        sns.scatterplot(x='cx', y='cy', hue='label', data=df1, palette='viridis', legend=False, ax=axes[1, 0])
        sns.scatterplot(x='width', y='height', hue='label', data=df2, palette='viridis', legend=False, ax=axes[1, 1])

        # 设置 x 轴和 y 轴刻度范围为 0 到 1，间隔为 0.1
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_xticks(np.arange(0, 1.1, 0.1))
        axes[1, 0].set_yticks(np.arange(0, 1.1, 0.1))

        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks(np.arange(0, 1.1, 0.1))
        axes[1, 1].set_yticks(np.arange(0, 1.1, 0.1))

        # 为每个散点图设置名称
        axes[1, 0].set_title('Center X and Y')
        axes[1, 1].set_title('Width and Height')

        # 设置 x 轴和 y 轴标签
        axes[1, 0].set_xlabel('Center X')
        axes[1, 0].set_ylabel('Center Y')

        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Height')

    def plot(self, save_dir: Path):
        """
        一键绘制并保存所有统计图表：
        label_infos.png（包含 boxes_statistics + cxcy_and_wh）
        """
        if len(self.bboxes) == 0:
            LOGGER.warning(f"{self.prefix} No bounding boxes available; skipping label visualization plots.")
            return

        # 创建一个2行2列的子图网格
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        self.boxes_statistics(axes)
        self.cxcy_and_wh(axes)

        # 调整子图之间的间距
        fig.tight_layout()

        # 保存图像
        fig.savefig(save_dir / f'{self.prefix}_label_infos.png', bbox_inches='tight', dpi=300)

        # 关闭 figure，防止内存泄漏
        plt.close(fig)


class DetectImgResult(BaseImgResult):
    """
    目标检测可视化实现：
    在图片上画框、写类别与置信度。
    """

    def __init__(self, save_dir: Path, plot_count: int = 4, mode: str = "val", max_sub_len: int = 3, rgb=True, draw_conf_threshold: float = 0.25):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)
        self.draw_conf_threshold = draw_conf_threshold

    def _prepare_imgs(self, batch_samples):
        """默认：返回 (imgs_np, None)，预测由外部赋值。"""
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        return imgs_np, None

    def _draw_one_img(self, img: np.ndarray, pred: torch.Tensor) -> np.ndarray:
        """
        绘制检测框：框 + 类别 + 置信度。
        """
        from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry

        if pred.numel() == 0:
            return img

        mask = pred[:, 4] > self.draw_conf_threshold
        pred = pred[mask].cpu().numpy()
        bboxes, confs, labels = pred[:, :4], pred[:, 4], pred[:, 5]
        bboxes = cxcywh_2_lxlyrxry(bboxes).astype(int)

        # Pillow 用 RGB；如果 img 是 BGR，先转 RGB
        pil_img = Image.fromarray(img[..., ::-1] if not self.rgb else img)
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2), conf, cls in zip(bboxes, confs, labels):
            if conf >= self.draw_conf_threshold:
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                cls_txt = str(int(cls))
                cls_w, cls_h = self._text_size(draw, cls_txt)
                draw.rectangle((x2 - cls_w - 2, y1, x2, y1 + cls_h + 2),
                               fill=(255, 255, 255))
                draw.text((x2 - cls_w, y1), cls_txt, fill=(0, 0, 0), font=self.font)

                conf_txt = f"{conf:.2f}"
                conf_w, conf_h = self._text_size(draw, conf_txt)
                draw.rectangle((x1, y2 - conf_h - 2, x1 + conf_w + 20, y2),
                               fill=(255, 255, 255))
                draw.text((x1, y2 - conf_h), conf_txt, fill=(0, 0, 0), font=self.font)

        drawn = np.array(pil_img)  # RGB
        return drawn
