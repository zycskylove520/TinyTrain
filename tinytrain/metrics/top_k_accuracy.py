import torch


class ClassifyTopKAccuracy:
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

    def result(self):
        """
        返回百分比形式的 Top-k 准确率。

        Returns:
            float: 范围 0-100。
        """
        return 100 * self.num_correct / self.num_total


class ClassifySingleClassesAccuracy:
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

    def result(self):
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
