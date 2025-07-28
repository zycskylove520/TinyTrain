import torch


class ClassifyTopKAccuracy:
    """
    top-k accuracy metric.
    """

    def __init__(self, k=1):
        """
        @param k: 计算前k准确率
        """
        self.k = k
        self.num_correct = 0
        self.num_total = 0

    def reset(self):
        self.num_correct = 0
        self.num_total = 0

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        # 获取前 k 个预测结果
        _, predicted = torch.topk(pred, self.k, dim=1)
        self.num_total += label.shape[0]
        self.num_correct += (predicted == label.unsqueeze(1)).sum().item()

    def result(self):
        return 100 * self.num_correct / self.num_total


class ClassifySingleClassesAccuracy:
    """
    计算每个类别的准确率。
    """

    def __init__(self, num_classes, classes_name=None, k=1):
        self.num_classes = num_classes
        self.classes_name = classes_name
        self.k = k

        self.class_correct = [0] * num_classes  # 每个类别的正确预测数
        self.class_total = [0] * num_classes  # 每个类别的总数

    def reset(self):
        self.class_correct = [0] * self.num_classes
        self.class_total = [0] * self.num_classes

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        _, predicted = torch.max(pred, dim=1)
        bool_classes = (predicted == label).squeeze()
        for i in range(len(label)):
            label_ = label[i]
            self.class_correct[label_] += bool_classes[i].item()
            self.class_total[label_] += 1

    def result(self):
        acc_results = []
        for i in range(self.num_classes):
            accuracy = 100 * self.class_correct[i] / self.class_total[i] if self.class_total[i] > 0 else 0
            acc_results.append(accuracy)
        return acc_results
