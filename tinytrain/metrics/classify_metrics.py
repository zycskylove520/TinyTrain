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

import math
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

from tinytrain.utils import LOGGER

from .base import BaseImgResult, TTBaseMetric


class ClassesLabelHistogram:
    """
    标签类别分布直方图 (`label_histogram`)
    """

    def __init__(self, num_classes: int, class_names: list[str], labels: np.ndarray, prefix: str = "train"):
        """
        Args:
            num_classes (int): 类别总数。
            class_names (list[str]): 与索引对应的类别名称列表。
            labels (np.ndarray): 一维数组，长度 = 边界框数量，存储每个框的类别索引。
            prefix (str): 保存图表名称前缀。
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.labels = labels
        self.prefix = prefix

    def plot(self, save_dir: Path):
        """
        绘制类别实例数量直方图并保存。

        图表说明
        --------
        - 横轴：类别名称
        - 纵轴：实例数量（条形顶部显示数值）
        - 自动根据类别数量调整宽度
        """
        if len(self.labels) == 0:
            LOGGER.warning(f"{self.prefix} No bounding boxes available; skipping classes label histogram plots.")
            return

        instances = np.zeros(self.num_classes, dtype=int)
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
        plt.savefig(save_dir / f'{self.prefix}_label_histogram.png', bbox_inches='tight', dpi=300)

        # 关闭 figure，防止内存泄漏
        plt.close()


class ClassifyConfusionMatrix:
    """
    分类任务专用混淆矩阵。

    功能
    ----
    1. 统计每个类别预测与真实标签的匹配次数。
    2. 支持重置 (`reset`) 与增量更新 (`update`)。
    3. 自动绘制原始与归一化混淆矩阵图，并保存到磁盘。
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        """
        Args:
            num_classes (int): 类别数量。
            class_names (list[str]): 类别名称列表，与索引一一对应。
        """
        self.num_classes = num_classes
        self.class_names = class_names

        # 混淆矩阵, x轴是预测类别，y轴是真实类别
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self):
        """将矩阵清零，准备新一轮统计。"""
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        更新混淆矩阵。

        Args:
            pred (Tensor): 模型预测 logits，形状 [batch, num_classes]。
            label (Tensor): 真实标签，形状 [batch]。
        """
        pred_idx = torch.argmax(pred, dim=1)
        for true_label, pred_label in zip(label, pred_idx):
            self.confusion_matrix[true_label, pred_label] += 1

    def plot(self, save_dir: Path):
        """
        绘制并保存原始、归一化两份混淆矩阵图。

        Args:
            save_dir (Path): 保存目录。
        """
        LOGGER.info(f"plotting confusion matrix...")
        confusion_matrix = self.confusion_matrix.numpy()
        # 绘制混淆矩阵
        width = height = self.num_classes
        plt.figure(figsize=(width * 2, height * 2))  # 设置图表大小
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
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
        # 归一化处理：每一行除以该行的和
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=0, keepdims=True) + 1e-5)

        plt.figure(figsize=(width * 2, height * 2))  # 设置图表大小
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Normalized)')

        # 设置x轴标签竖着排，y轴标签横着排
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig(save_dir / 'confusion_matrix_normalized.png')  # 保存为 PNG 文件
        plt.close()  # 关闭图像窗口，释放资源


class ClassifyImgResult(BaseImgResult):
    """
    分类任务可视化实现：
    在图片左上角叠加 true / pred 标签文本。
    """

    def __init__(self, save_dir: Path, plot_count: int = 4, mode: str = "val", max_sub_len: int = 3, rgb=True, **kwargs):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)
        # 额外存一份 class_names
        self.class_names = kwargs.get('class_names_dict', {})
        self.title = None

    def _prepare_imgs(self, batch_samples):
        """重写：返回 (imgs_np, true_labels)"""
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()  # bchw->bhwc
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        true_labels = batch_samples.target.cpu().numpy()
        return imgs_np, true_labels  # 返回 (imgs, true_labels)

    def _draw_one_img(self, img: np.ndarray, info) -> np.ndarray:
        """
        绘制单张分类图：
        左上角写 true / pred 标签。
        """
        pred_tensor, true_label = info
        pred_label = int(torch.argmax(pred_tensor).item())

        pil_img = Image.fromarray(img.astype(np.uint8))
        # 若原通道顺序与需求相反，则翻转
        drawn = np.array(pil_img)
        if not self.rgb and drawn.shape[-1] == 3:  # BGR->RGB
            drawn = drawn[..., ::-1]
        elif self.rgb and drawn.shape[-1] == 3:  # RGB
            drawn = drawn if img.shape[-1] == 3 and img.dtype == np.uint8 else drawn
        # 这里 img 是 uint8 RGB，如果网络给的是 BGR 会经过 _prepare_imgs 的 ::-1 处理
        # 上面判断条件可再按实际 pipeline 微调
        self.title = (f"true:{self.class_names.get(true_label, true_label)}\n"
                      f"pred:{self.class_names.get(pred_label, pred_label)}")
        return drawn

    def plot(self, batch_samples, pred: torch.Tensor):
        """重写：使用子图标题显示标签。"""
        if self.is_plot:
            return

        if self.plot_tick >= self.plot_count:
            self.is_plot = True
            return

        self.plot_tick += 1

        imgs_np, true_labels = self._prepare_imgs(batch_samples)
        pred_np = pred.cpu()
        B, *_ = imgs_np.shape
        fill_len = min(self.max_sub_len, math.ceil(B ** 0.5))
        n_show = min(B, fill_len * fill_len)

        if self._fig is None:
            dpi = 100
            self._fig, self._axs = plt.subplots(
                fill_len,
                fill_len,
                dpi=dpi,
                constrained_layout=True,  # 取代 tight_layout
                gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            for ax in self._axs.flat:
                ax.axis('off')

        for idx in range(n_show):
            row, col = divmod(idx, fill_len)
            ax = self._axs[row, col]
            drawn = self._draw_one_img(imgs_np[idx], (pred_np[idx], true_labels[idx]))
            ax.imshow(drawn)
            ax.set_title(self.title)  # 设置标题
            ax.set_xticks([])
            ax.set_yticks([])

        self._fig.savefig(
            self.save_dir / f'{self.mode}_img_result_{self.plot_tick}.png',
            dpi=100, bbox_inches='tight', pad_inches=0)

        plt.close(self._fig)


class ClassifyTopKAccuracy(TTBaseMetric):
    """
    分类任务 Top-k 准确率（Top-k Accuracy）指标。

    功能
    ----
    1. 支持任意 k（Top-1、Top-5 等）。
    2. 支持 `reset()` 与 `update()` 的增量统计。
    3. 最终返回百分比形式的准确率。

    使用示例
    --------
    >>> metric = ClassifyTopKAccuracy(k=5)
    >>> metric.update(logits, labels)
    >>> acc = metric.result()
    """

    def __init__(self, k=1):
        """
        Args:
            k (int): 前 k 个预测中只要包含真实标签即算正确。
        """
        super().__init__()
        self.k = k
        self.num_correct = 0
        self.num_total = 0

    def reset(self):
        """清零计数器，开始新一轮统计。"""
        self.num_correct = 0
        self.num_total = 0

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        更新计数器。

        Args:
            pred (Tensor): 模型输出 logits，形状 [batch, num_classes]。
            label (Tensor): 真实标签，形状 [batch]。
        """
        # 获取前 k 个预测结果
        _, predicted = torch.topk(pred, self.k, dim=1)
        self.num_total += label.shape[0]
        self.num_correct += (predicted == label.unsqueeze(1)).sum().item()

    def compute(self):
        """
        返回百分比形式的 Top-k 准确率。

        Returns:
            float: 范围 0-100。
        """
        return 100 * self.num_correct / self.num_total


class ClassifySingleClassesAccuracy(TTBaseMetric):
    """
    逐类别准确率计算器。

    功能
    ----
    1. 支持 Top-1 或 Top-k 统计（当前实现默认为 Top-1）。
    2. 输出每个类别的独立准确率列表。
    3. 支持类别名称映射，方便后续日志或可视化。

    使用示例
    --------
    >>> metric = ClassifySingleClassesAccuracy(num_classes=10, classes_name=["cat", "dog", ...])
    >>> metric.update(logits, labels)
    >>> acc_list = metric.result()
    """

    def __init__(self, num_classes, classes_name=None, k=1):
        """
        Args:
            num_classes (int): 类别总数。
            classes_name (list[str] | None): 类别名称列表，长度需等于 num_classes。
            k (int): 当前固定为 1（Top-1 准确率）。
        """
        super().__init__()
        self.num_classes = num_classes
        self.classes_name = classes_name
        self.k = k

        self.class_correct = [0] * num_classes  # 每个类别的正确预测数
        self.class_total = [0] * num_classes  # 每个类别的总数

    def reset(self):
        """清零所有类别的正确/总数计数器。"""
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        更新逐类别计数器。

        Args:
            pred (Tensor): 模型输出 logits，形状 [batch, num_classes]。
            label (Tensor): 真实标签，形状 [batch]。
        """
        _, predicted = torch.max(pred, dim=1)
        bool_classes = (predicted == label).squeeze()
        for i in range(len(label)):
            label_ = label[i]
            self.class_correct[label_] += bool_classes[i].item()
            self.class_total[label_] += 1

    def compute(self):
        """
        返回每个类别的准确率列表。

        Returns:
            list[float]: 长度等于类别数，元素为百分比 0-100。
        """
        acc_results = []
        for i in range(self.num_classes):
            accuracy = 100 * self.class_correct[i] / self.class_total[i] if self.class_total[i] > 0 else 0
            acc_results.append(accuracy)
        return acc_results
