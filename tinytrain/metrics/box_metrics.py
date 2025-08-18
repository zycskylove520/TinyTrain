import torch
import torchmetrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.distributed as dist

from pathlib import Path

from tinytrain.global_var import WORLD_SIZE, RANK
from tinytrain.metrics.base.base_metrics import BaseMetric
from tinytrain.utils import LOGGER


class BoxMetrics(BaseMetric):
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

    def __init__(self,
                 class_metrics: bool = False,
                 class_names: list = None,
                 ):
        """
        Args:
            class_metrics (bool):
                是否计算类别级指标。
            class_names (list[str] | None):
                类别名称列表，用于可视化时替换索引。
        """
        super(BoxMetrics, self).__init__()
        self.class_metrics = class_metrics
        self.class_names = class_names

        self.metrics = torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh',
                                                                   iou_type='bbox',
                                                                   class_metrics=self.class_metrics,
                                                                   extended_summary=True,
                                                                   sync_on_compute=False  # 必须取消同步，使用单卡验证
                                                                   )
        self.results = None

        # recall_curve的维度为：(TxK), 其中T是 IoU 阈值的数量，K是类别数量
        self.recall_curve = torch.zeros((1, 1), dtype=torch.float32)
        # precision_curve的维度为：(TxRxK), 其中T是 IoU 阈值的数量，R是召回率阈值的数量，K是类别数量
        self.precision_curve = torch.zeros((1, 1, 1), dtype=torch.float32)

    def reset(self):
        """重置内部状态，开始新一轮评估。"""
        self.results = None
        self.recall_curve = torch.zeros((1, 1), dtype=torch.float32)
        self.precision_curve = torch.zeros((1, 1, 1), dtype=torch.float32)
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
                print("t is empty")
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
        all_recall_matrix = self.results["recall"]
        # 只取area=all，并取最大检测数量=100
        self.recall_curve = all_recall_matrix[:, :, 0, 2]

        # all_precision_matrix的维度为：(TxRxKxAxM)
        # 其中T是 IoU 阈值的数量，R是置信度阈值的数量，K是类别数量, A是区域数量，M是每幅图像的最大检测数量。
        # 区域数量：默认四个区域，area=all、area=small、area=medium和area=large
        # 最大检测数量：默认三个检测数量：1，10，100
        all_precision_matrix = self.results["precision"]
        # 只取area=all，并取最大检测数量=100
        self.precision_curve = all_precision_matrix[:, :, :, 0, 2]

    def map50(self):
        """
        返回模型在 IoU 阈值为 0.5 时的平均精度（mAP@0.5）。

        mAP@0.5 是目标检测中常用的评估指标，表示在 IoU 阈值为 0.5 时的平均精度。
        该指标衡量模型在不同类别上的平均检测性能。

        Returns:
            float: mAP@0.5 的值。如果 self.results 为空，则返回 0.0。
        """
        return self.results["map_50"].item() if self.results else 0.

    def map75(self):
        """
        返回模型在 IoU 阈值为 0.75 时的平均精度（mAP@0.75）。

        mAP@0.75 是一个更严格的评估指标，表示在 IoU 阈值为 0.75 时的平均精度。
        该指标衡量模型在更高重叠度下的检测性能。

        Returns:
            float: mAP@0.75 的值。如果 self.results 为空，则返回 0.0。
        """
        return self.results["map_75"].item() if self.results else 0.

    def map50_95(self):
        """
        返回模型在 IoU 阈值范围 [0.5, 0.95] 内的平均精度（mAP@[0.5:0.95]）。

        mAP@[0.5:0.95] 是一个综合评估指标，表示在 IoU 阈值从 0.5 到 0.95（步长为 0.05）的平均精度。
        该指标衡量模型在不同重叠度下的整体检测性能。

        Returns:
            float: mAP@[0.5:0.95] 的值。如果 self.results 为空，则返回 0.0。
        """
        return self.results["map"].item() if self.results else 0.

    def map_small(self):
        """小目标 mAP"""
        return self.results["map_small"].item() if self.results else 0.

    def map_medium(self):
        """中目标 mAP"""
        return self.results["map_medium"].item() if self.results else 0.

    def map_large(self):
        """大目标 mAP"""
        return self.results["map_large"].item() if self.results else 0.

    def mar_1(self):
        """mAR@1"""
        return self.results[f"mar_1"].item() if self.results else 0.

    def mar_10(self):
        """mAR@10"""
        return self.results[f"mar_10"].item() if self.results else 0.

    def mar_100(self):
        """
        返回模型在最大检测数为 100 时的平均召回率（mAR@100）。

        mAR@100 是目标检测中的一个评估指标，表示在每个图像中最多检测 100 个目标时的平均召回率。
        该指标衡量模型在固定检测数下的召回能力。

        Returns:
            float: mAR@100 的值。如果 self.results 为空，则返回 0.0。
        """
        return self.results[f"mar_100"].item() if self.results else 0.

    def per_class_recall(self):
        """类别级 Recall@0.5"""
        # 只计算每个类别在iou=0.5的情况下的recall
        LOGGER.info(f"Calculate iou=0.5, per class recall.")
        return self.recall_curve[0, :]

    def recall(self):
        """
        在 iou=0.5 下的总体 Recall
        """
        return self.recall_curve[0, :].mean().item() if self.results else 0.

    def per_class_precision(self, conf_threshold=0.25):
        """类别级 Precision@conf"""
        # 只计算每个类别在iou=0.5,conf=conf_threshold的情况下的precision
        LOGGER.info(f"Calculate conf={conf_threshold} & iou=0.5, per class precision.")
        conf = int(conf_threshold * 100)
        return self.precision_curve[0, conf, :]

    def precision(self, conf_threshold=0.25):
        """在 iou=0.5,conf=conf_threshold 下的总体 Precision"""
        conf = int(conf_threshold * 100)
        return self.precision_curve[0, conf, :].mean().item() if self.results else 0.

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
        import matplotlib.lines as mlines

        # 类别名
        if self.class_names is None:
            class_labels = [f'Class {i}' for i in self.classes()]
        else:
            class_labels = [self.class_names[i] for i in self.classes()]
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
        plt.savefig(save_dir / 'R_Curve.png', dpi=150)
        plt.close()

    def plot_pr_curve(self, save_dir: Path):
        """
        绘制 PR 曲线（IoU = 0.5, 0.75, 0.95）
        """
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
        plt.savefig(save_dir / 'PR_Curve.png', dpi=150)
        plt.close()

    def plot(self, save_dir: Path):
        """一键绘制所有曲线。"""
        self.plot_recall_curve(save_dir)
        self.plot_pr_curve(save_dir)
