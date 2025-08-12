from typing import Optional

import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from matplotlib import pyplot as plt


class LabelInfo:
    """
    数据集标签统计与可视化工具类。

    功能
    ----
    1. 标签类别分布直方图 (`label_histogram`)
    2. 边界框空间分布热力图 (`boxes_statistics`)
    3. 中心点及宽高分布散点图 (`cxcy_and_wh`)
    4. 一键保存所有图表 (`plot`)
    """

    def __init__(self, num_classes: int, class_names: list[str], labels: np.ndarray, bboxes: np.ndarray,max_samples: Optional[int] = None):
        """
        Args:
            num_classes (int): 类别总数。
            class_names (list[str]): 与索引对应的类别名称列表。
            labels (np.ndarray): 一维数组，长度 = 边界框数量，存储每个框的类别索引。
            bboxes (np.ndarray): 二维数组，形状 [box_num, 4]，格式为 cxcywh（已归一化到 [0,1]）。
        """
        self.num_classes = num_classes
        self.class_names = class_names

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
        self.labels = labels
        self.bboxes = bboxes

    def label_histogram(self, save_dir: Path):
        """
        绘制类别实例数量直方图并保存。

        图表说明
        --------
        - 横轴：类别名称
        - 纵轴：实例数量（条形顶部显示数值）
        - 自动根据类别数量调整宽度
        """
        instances = np.arange(0, self.num_classes, dtype=int)
        for label in self.labels:
            instances[int(label)] += 1
        data = {
            'Classes': self.class_names,
            'Instances': instances
        }
        df = pd.DataFrame(data)

        # 根据标签数量自适应图表宽度
        width = self.num_classes
        height = 6  # 固定高度
        plt.figure(figsize=(width, height))  # 设置图表大小
        ax = sns.barplot(x='Classes', y='Instances', data=df, hue='Classes', palette=sns.color_palette("hsv", self.num_classes), legend=False)

        # 在每个条形上添加文本标签
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        # 设置x轴刻度
        plt.xticks(ticks=self.class_names, labels=self.class_names, rotation=90)  # 设置x轴刻度和标签
        # 设置y轴刻度
        plt.yticks([])  # 关闭y轴刻度
        # plt.yticks(ticks=instances, labels=instances)  # 设置y轴刻度和标签

        # 设置图表标题和标签
        plt.title('label distribution histogram')

        # 保存图像
        plt.savefig(save_dir / 'label_histogram.png', bbox_inches='tight', dpi=300)

        # 关闭 figure，防止内存泄漏
        plt.close()

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
        1. label_histogram.png
        2. label_infos.png（包含 boxes_statistics + cxcy_and_wh）
        """
        # 标签直方图单独绘制
        self.label_histogram(save_dir)

        # 创建一个2行2列的子图网格
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        self.boxes_statistics(axes)
        self.cxcy_and_wh(axes)

        # 调整子图之间的间距
        fig.tight_layout()

        # 保存图像
        fig.savefig(save_dir / 'label_infos.png', bbox_inches='tight', dpi=300)

        # 关闭 figure，防止内存泄漏
        plt.close(fig)