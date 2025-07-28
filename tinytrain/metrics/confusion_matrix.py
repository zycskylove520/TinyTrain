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
    分类算法使用的混淆矩阵
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        """
        :param num_classes: 分类类别数量
        :param class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names

        # 混淆矩阵, x轴是预测类别，y轴是真实类别
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self):
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        """
        @param pred: 分类模型预测输出，shape:[batch,num_classes]
        @param label: 真实标签，shape:[batch]
        """
        pred_idx = torch.argmax(pred, dim=1)
        for true_label, pred_label in zip(label, pred_idx):
            self.confusion_matrix[true_label, pred_label] += 1

    def plot(self, save_dir: Path):
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
    检测算法使用的混淆矩阵
    """

    def __init__(self, num_classes: int, class_names: list[str], conf_threshold=0.25, iou_threshold=0.45):
        """
        @param num_classes: 检测类别数量
        :param class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_names.append("background")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 混淆矩阵, num_classes + 1是因为包含背景, x轴是预测类别, y轴是真实类别
        self.confusion_matrix = torch.zeros((num_classes + 1, num_classes + 1), dtype=torch.int64)

    def reset(self):
        self.confusion_matrix = torch.zeros_like(self.confusion_matrix)

    def update(self, pred: list[torch.Tensor], target: list[torch.Tensor]):
        """
        @param pred: 检测模型输出的预测矩阵经过nms后的结果，list的长度为batch，list内部的tensor要求shape为:[after_nms_num_boxes, 4+score+class_idx],要求:
        boxes的format必须为cxcywh格式,score是类别分数，class_idx是box预测的类别索引
        @param target:真实标签，list的长度为batch，list内部的tensor要求shape为:[num_boxes, 4+class_idx],要求:
        boxes的format必须为cxcywh格式,class_idx是真实的类别索引
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
