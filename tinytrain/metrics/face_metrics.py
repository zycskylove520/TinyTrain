import numpy as np

from sklearn import metrics
from pathlib import Path

from .base import BaseMetric


class FaceRecognitionMetrics(BaseMetric):
    """
    人脸识别1:1验证场景评测指标计算器。

    功能
    ----
    1. 支持增量 accumulate → 一次性 compute → 绘图保存
    2. 主要指标：
        - AUC
        - TPR@FAR = 1e-3
        - 最优阈值（Youden's index）
        - Balanced Accuracy（= (TPR + TNR)/2 ）
    """

    def __init__(self):
        """
        初始化空缓冲区与指标占位变量。
        """
        super().__init__()

        # 原始样本缓冲区
        self.scores = []  # 每次 update 的 cosine 得分
        self.labels = []  # 0 负样本  1 正样本

        # 最终指标（compute 后才有意义）
        self._auc = 0.
        self._bal_acc = 0.
        self._tpr_1e3 = 0.  # TPR when FAR = 1e-3
        self._tpr_1e4 = 0.  # TPR when FAR = 1e-4
        self._best_threshold = 0.  # cosine 最大阈值

        # ROC 曲线原始数据，供 plot 复用
        self._fpr = None
        self._tpr = None
        self._threshold = None

    def reset(self):
        """
        清空缓冲区与所有计算结果，支持多轮复用。
        """
        self.scores = []
        self.labels = []
        self._auc = 0.
        self._bal_acc = 0.
        self._tpr_1e3 = 0.
        self._tpr_1e4 = 0.
        self._best_threshold = 0.

        self._fpr = None
        self._tpr = None
        self._threshold = None

    def update(self, scores: np.ndarray, labels: np.ndarray):
        """
        批量追加一次推理结果。

        Args:
            scores (np.ndarray): 一维相似度得分。
            labels (np.ndarray): 一维 0/1 标签，0 表示负样本，1 表示正样本。
        """
        scores = np.asarray(scores).ravel()
        labels = np.asarray(labels).ravel()
        if scores.size != labels.size:
            raise ValueError(f"batch size mismatch: {scores.size} vs {labels.size}")
        self.scores.append(scores)
        self.labels.append(labels)

    def compute(self):
        """
        汇总所有样本后计算指标：
        1. ROC → AUC
        2. Youden 索引 → 最优阈值
        3. 在该阈值下计算 Balanced Accuracy
        4. 插值得到 TPR@FAR=1e-3
        """
        if not self.scores:  # 防止空计算
            raise RuntimeError("No samples accumulated; call update() first.")

        all_scores = np.concatenate(self.scores)
        all_labels = np.concatenate(self.labels)

        # ---- 2. ROC 曲线 ---- #
        self._fpr, self._tpr, self._threshold = metrics.roc_curve(
            all_labels, all_scores, drop_intermediate=False)
        self._auc = metrics.auc(self._fpr, self._tpr)

        # ---- 3. Youden 最优阈值 ---- #
        best_idx = np.argmax(self._tpr - self._fpr)
        self._best_threshold = self._threshold[best_idx]

        # ---- 4. Balanced Accuracy ---- #
        pred = (all_scores >= self._best_threshold).astype(int)
        # TPR & TNR 平均
        self._bal_acc = 0.5 * (
                metrics.recall_score(all_labels, pred, pos_label=1) +
                metrics.recall_score(all_labels, pred, pos_label=0)
        )

        # ---- 5. TPR@FAR=1e-3 ---- #
        if self._fpr[-1] < 1e-3:  # 全程 FAR 都没到 1e-3
            self._tpr_1e3 = self._tpr[-1]  # 保守外推
        else:
            self._tpr_1e3 = np.interp(1e-3, self._fpr, self._tpr)

        # ---- 6. TPR@FAR=1e-4 ---- #
        if self._fpr[-1] < 1e-4:  # 全程 FAR 都没到 1e-4
            self._tpr_1e4 = self._tpr[-1]
        else:
            self._tpr_1e4 = np.interp(1e-4, self._fpr, self._tpr)

    def auc(self):
        """
        返回计算得到的 AUC 值。

        Returns:
            float: AUC；若尚未 compute 则返回 0。
        """
        return self._auc

    def balanced_accuracy(self):
        """
        返回最优阈值下的 Balanced Accuracy。

        Returns:
            float: Balanced Accuracy；若尚未 compute 则返回 0。
        """
        return self._bal_acc

    def tpr_1e3(self):
        """
        返回 FAR=1e-3 时对应的 TPR。

        Returns:
            float: TPR@FAR=1e-3；若尚未 compute 则返回 0。
        """
        return self._tpr_1e3

    def tpr_1e4(self):
        """
        返回 FAR=1e-4 时对应的 TPR。

        Returns:
            float: TPR@FAR=1e-4；若尚未 compute 则返回 0。
        """
        return self._tpr_1e4

    def best_threshold(self):
        """
        返回 Youden 索引最优阈值（cosine 得分）。

        Returns:
            float: 最优阈值；若尚未 compute 则返回 0。
        """
        return self._best_threshold

    def plot(self, save_dir: Path):
        """
        绘制 ROC 曲线、AUC 面积及 TPR@FAR=1e-3 标注图并保存。

        Args:
            save_dir (Path): 保存目录路径，将生成 ROC_curve.png。
        """
        import matplotlib.pyplot as plt

        if self._fpr is None:
            raise RuntimeError("Metrics not computed yet; call compute() first.")
        #  画图
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self._fpr, self._tpr, linewidth=2, label=f"ROC curve (AUC = {self._auc:.3f})")
        ax.fill_between(self._fpr, self._tpr, alpha=0.15)  # 填充 AUC 区域
        ax.axhline(y=self._tpr_1e3, color="r", linestyle="--",
                   label=f"TPR@FAR=1e-3 = {self._tpr_1e3:.3f}")
        ax.set_xscale("log")  # 横坐标用 log 更直观
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("True Accept Rate (TPR)")
        ax.set_title(f" ROC Curve (AUC = {self._auc:.3f})")
        ax.legend(loc="lower right")
        ax.grid(True, which="major", linestyle="--", alpha=0.4)

        # 保存 / 显示
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "ROC_curve.png", bbox_inches="tight", dpi=300)

        plt.close()
