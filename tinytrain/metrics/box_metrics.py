import torch
import torchmetrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


class BoxMetrics:
    def __init__(self,
                 iou_thresholds: list = None,
                 rec_thresholds: list = None,
                 max_detection_thresholds: list = None,
                 class_metrics: bool = False,
                 extended_summary: bool = True,
                 class_names: list = None,
                 ):
        self.iou_thresholds = iou_thresholds
        self.rec_thresholds = rec_thresholds
        self.max_detection_thresholds = max_detection_thresholds
        self.class_metrics = class_metrics
        self.extended_summary = extended_summary
        self.class_names = class_names

        if self.iou_thresholds is None:
            self.iou_thresholds = torch.linspace(0.5, 0.95, round((0.95 - 0.5) / 0.05) + 1).tolist()
        if self.rec_thresholds is None:
            self.rec_thresholds = torch.linspace(0.0, 1.00, round(1.00 / 0.01) + 1).tolist()
        if self.max_detection_thresholds is None:
            self.max_detection_thresholds = [1, 100, 300]
        self.metrics = torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh',
                                                                   iou_type='bbox',
                                                                   iou_thresholds=self.iou_thresholds,
                                                                   rec_thresholds=self.rec_thresholds,
                                                                   class_metrics=self.class_metrics,
                                                                   extended_summary=self.extended_summary,
                                                                   max_detection_thresholds=self.max_detection_thresholds
                                                                   )
        self.results = None

        # recall_curve的维度为：(TxK), 其中T是 IoU 阈值的数量，K是类别数量
        self.recall_curve = torch.zeros((1, 1), dtype=torch.float32)
        # precision_curve的维度为：(TxRxK), 其中T是 IoU 阈值的数量，R是召回率阈值的数量，K是类别数量
        self.precision_curve = torch.zeros((1, 1, 1), dtype=torch.float32)

    def reset(self):
        # self.metrics = torchmetrics.detection.MeanAveragePrecision(box_format='cxcywh',
        #                                                            iou_type='bbox',
        #                                                            iou_thresholds=self.iou_thresholds,
        #                                                            rec_thresholds=self.rec_thresholds,
        #                                                            class_metrics=self.class_metrics,
        #                                                            extended_summary=self.extended_summary,
        #                                                            max_detection_thresholds=self.max_detection_thresholds
        #                                                            )
        self.recall_curve = torch.zeros((1, 1), dtype=torch.float32)
        self.precision_curve = torch.zeros((1, 1), dtype=torch.float32)
        self.metrics.reset()

    def update(self, pred: list[torch.Tensor], target: list[torch.Tensor]):
        """
        @param pred: 检测模型输出的预测矩阵经过nms后的结果，list的长度为batch，list内部的tensor要求shape为:[after_nms_num_boxes, 4+score+class_idx],要求:
        boxes的format必须为lxlyrxry格式,score是类别分数，class_idx是box预测的类别索引
        @param target: 真实标签，list的长度为batch，list内部的tensor要求shape为:[num_boxes, 4+class_idx],要求:
        boxes的format必须为lxlyrxry格式,class_idx是真实的类别索引
        """
        pred_list = []
        target_list = []
        for p, t in zip(pred, target):
            # 准备预测结果和目标数据
            pred_list.append(dict(
                boxes=p[:, :4],  # 预测的边界框
                scores=p[:, 4],  # 预测的置信度
                labels=p[:, 5:].squeeze(-1).int(),  # 预测的类别
            ))
            target_list.append(
                dict(
                    boxes=t[:, :4],  # 真实的边界框
                    labels=t[:, 4:].squeeze(-1).int(),  # 真实的类别
                )
            )
        self.metrics.update(pred_list, target_list)

    def compute(self):
        # 更新完计算统计指标
        self.results = self.metrics.compute()

        # curve
        # all_recall_matrix的维度为：(TxKxAxM)
        # 其中T是 IoU 阈值的数量，K是类别数量， A是区域数量，M是每幅图像的最大检测数量。
        # 区域数量：默认四个区域，area=all、area=small、area=medium和area=large
        # 最大检测数量：默认三个检测数量：1，100，300
        if self.extended_summary:
            all_recall_matrix = self.results["recall"]
            # 只取area=all，并将所有检测数量情况计算平均的情况
            self.recall_curve = torch.mean(all_recall_matrix[:, :, 0, :], -1)

        # all_precision_matrix的维度为：(TxRxKxAxM)
        # 其中T是 IoU 阈值的数量，R是置信度阈值的数量，K是类别数量, A是区域数量，M是每幅图像的最大检测数量。
        # 区域数量：默认四个区域，area=all、area=small、area=medium和area=large
        # 最大检测数量：默认三个检测数量：1，100，300
        if self.extended_summary:
            all_precision_matrix = self.results["precision"]
            # 只取area=all，并将所有检测数量情况计算平均的情况
            self.precision_curve = torch.mean(all_precision_matrix[:, :, :, 0, :], -1)

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
        return self.results["map_small"].item() if self.results else 0.

    def map_medium(self):
        return self.results["map_medium"].item() if self.results else 0.

    def map_large(self):
        return self.results["map_large"].item() if self.results else 0.

    def mar_a(self):
        return self.results[f"mar_{self.max_detection_thresholds[0]}"].item() if self.results else 0.

    def mar_b(self):
        return self.results[f"mar_{self.max_detection_thresholds[1]}"].item() if self.results else 0.

    def mar_c(self):
        """
        返回模型在最大检测数为 300 时的平均召回率（mAR@300）。

        mAR@300 是目标检测中的一个评估指标，表示在每个图像中最多检测 300 个目标时的平均召回率。
        该指标衡量模型在固定检测数下的召回能力。

        Returns:
            float: mAR@300 的值。如果 self.results 为空，则返回 0.0。
        """
        return self.results[f"mar_{self.max_detection_thresholds[2]}"].item() if self.results else 0.

    def per_class_recall(self):
        # 只计算每个类别在iou=0.5的情况下的recall
        return self.recall_curve[0, :]

    def recall(self):
        # 计算总的recall
        return max(self.per_class_recall().mean().item(), 0)

    def per_class_precision(self, conf_threshold=0.25):
        # 只计算每个类别在iou=0.5,conf=conf_threshold的情况下的precision
        conf = int(conf_threshold * 100)
        return self.precision_curve[0, conf, :]

    def precision(self):
        # 计算总的precision
        return max(self.per_class_precision().mean().item(), 0)

    def classes(self):
        """
        观察到的类别列表
        @return:
        """
        if self.results:
            return self.results["classes"].reshape(-1)
        else:
            raise ValueError("box metrics no classes!")

    def plot_recall_curve(self, save_dir: Path):
        # 检查 class_names 是否为 None
        if self.class_names is None:
            # 如果 class_names 为 None，使用默认的类别索引作为列名
            column_names = [f'Class {i}' for i in range(self.classes())]
        else:
            # 如果 class_names 不为 None，使用传入的类别名称作为列名
            column_names = [self.class_names[i] for i in self.classes()]

        # 将数据转换为 Pandas DataFrame
        data = pd.DataFrame(self.recall_curve.numpy(), columns=column_names)
        data['IoU'] = np.array(self.iou_thresholds)

        # 将数据从宽格式转换为长格式
        data_long = pd.melt(data, id_vars=['IoU'], var_name='Class', value_name='Recall')

        # 动态调整图形大小
        K = self.recall_curve.shape[1]  # 类别数量
        figsize = (12, 6 + K // 10)  # 根据类别数量调整高度

        # 使用 Seaborn 绘制曲线
        plt.figure(figsize=figsize)
        sns.lineplot(data=data_long, x='IoU', y='Recall', hue='Class', palette='tab20')

        # 添加图例并调整位置和字体大小
        plt.legend(title='Class', loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

        # 添加标题和坐标轴标签
        plt.title('Recall Curve for Each Class')
        plt.xlabel('IoU Threshold')
        plt.ylabel('Recall')

        # 显示图形
        plt.tight_layout(rect=(0, 0, 0.85, 1))  # 调整布局，留出空间给图例
        # plt.show()

        plt.savefig(save_dir / 'R_Curve.png')  # 保存为 PNG 文件
        plt.close()  # 关闭图像窗口，释放资源

    def plot_pr_curve(self, save_dir: Path):
        """
        绘制在iou=0.5的情况下的PR曲线
        @return:
        """
        precision_recall_curve = self.precision_curve[0]
        # 检查 class_names 是否为 None
        if self.class_names is None:
            # 如果 class_names 为 None，使用默认的类别索引作为列名
            column_names = [f'Class {i}' for i in range(self.classes())]
        else:
            # 如果 class_names 不为 None，使用传入的类别名称作为列名
            column_names = [self.class_names[i] for i in self.classes()]

        # 将数据转换为 Pandas DataFrame
        data = pd.DataFrame(precision_recall_curve.numpy(), columns=column_names)
        data['Recall'] = np.array(self.rec_thresholds)

        # 将数据从宽格式转换为长格式
        data_long = pd.melt(data, id_vars=['Recall'], var_name='Class', value_name='Precision')

        # 动态调整图形大小
        K = precision_recall_curve.shape[1]  # 类别数量
        figsize = (10, 6 + K // 10)  # 根据类别数量调整高度

        # 使用 Seaborn 绘制曲线
        plt.figure(figsize=figsize)
        sns.lineplot(data=data_long, x='Recall', y='Precision', hue='Class', palette='tab20')

        # 添加图例并调整位置
        plt.legend(title='Class', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        # 添加标题和坐标轴标签
        plt.title('Precision-Recall Curve for Each Class:iou=0.5')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # 显示图形
        plt.tight_layout()
        # plt.show()

        plt.savefig(save_dir / 'PR_Curve.png')  # 保存为 PNG 文件
        plt.close()  # 关闭图像窗口，释放资源

    def plot(self, save_dir: Path):
        self.plot_recall_curve(save_dir)
        self.plot_pr_curve(save_dir)
