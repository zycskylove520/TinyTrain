import torch
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

from tinytrain.utils import LOGGER
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry


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


class DetectConfusionMatrix:
    """
    检测任务专用混淆矩阵工具。

    功能
    ----
    1. 以 IoU 阈值为准，将预测框匹配到 GT 框，统计 TP / FP / FN。
    2. 额外引入“背景”类别，用于表示漏检（FN）与误检（FP）。
    3. 支持重置、增量更新、绘制原始/归一化矩阵。
    """
    def __init__(self, num_classes: int, class_names: list[str], conf_threshold=0.25, iou_threshold=0.45):
        """
        Args:
            num_classes (int): 前景类别数量。
            class_names (list[str]): 类别名称列表，顺序必须与索引对齐。
            conf_threshold (float): 置信度阈值，低于阈值的预测框丢弃。
            iou_threshold (float): IoU 阈值，大于该值视为匹配成功。
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_names.append("background")
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

            # 判断p是否为空tensor，即通过nms过滤后没有框
            if p.shape[0] == 0:
                # 图片有真实标签
                gt_classes = t[:, 4].int()
                for gc in gt_classes:
                    self.confusion_matrix[gc, self.num_classes] += 1  # background FN
                continue

            # p不为空，t也不为空
            # 置信度过滤
            p = p[p[:, 4] > self.conf_threshold]
            if p.shape[0] == 0:
                continue

            gt_classes = t[:, 4].int()
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

        # 绘制混淆矩阵
        # 根据标签数量自适应图表宽度
        width = height = self.num_classes + 1
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
