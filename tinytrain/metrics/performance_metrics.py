import os
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

from tinytrain.metrics.base import BaseMetric
from tinytrain.utils import LOGGER


class PerformanceMetrics(BaseMetric):
    def __init__(self, prefix: str = "", class_names: dict[int, str] = None):
        super().__init__()
        self.prefix = prefix
        self.class_names = class_names
        self.iouv = np.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.eps = 1e-16

        self._tp = []
        self._scores = []
        self._pred_classes = []
        self._target_classes = []

        self.results = {}

    def reset(self, *args, **kwargs):
        self._tp = []
        self._scores = []
        self._pred_classes = []
        self._target_classes = []
        self.results = {}

    def update(self, tp: np.ndarray, score: np.ndarray, pred_class: np.ndarray, target_class: np.ndarray):
        """
        Args:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            score (list[np.ndarray]): Array of confidence scores of the detections.
            pred_class (np.ndarray): Array of predicted classes of the detections.
            target_class (np.ndarray): Array of true classes of the detections.
        """
        self._tp.append(tp)
        self._scores.append(score)
        self._pred_classes.append(pred_class)
        self._target_classes.append(target_class)

    def compute(self):
        tp = np.concatenate(self._tp, axis=0)
        scores = np.concatenate(self._scores, axis=0)
        pred_classes = np.concatenate(self._pred_classes, axis=0)
        target_classes = np.concatenate(self._target_classes, axis=0)

        # Sort by objectness
        i = np.argsort(-scores)
        tp, scores, pred_classes = tp[i], scores[i], pred_classes[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_classes, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        x, prec_values = np.linspace(0, 1, 1000), []

        # Average precision, precision and recall curves
        ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            # 计算第ci个类别有多少个真实标签和预测标签
            i = pred_classes == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # 使用正常的方式去计算iou在0.5的情况下，不同置信度下的tp个数和fp的个数是非常耗时的，如果把置信度区间划分为1000份，那么需要循环1000次
            # 从性能上是不可取的。
            # 因此，使用cumsum的方式对所有的预测框进行累加，模拟置信度从0到1的情况，预测正确的目标框的个数

            # Accumulate FPs and TPs
            # 累计不同iou阈值下的FP的数量，所有没匹配上的都视为FP
            fpc = (1 - tp[i]).cumsum(0)
            # 累计不同iou阈值下的TP的数量
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + self.eps)  # recall curve
            # recall_curve只绘制iou为0.5时的recall
            r_curve[ci] = np.interp(-x, -scores[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            # precision_curve只绘制iou为0.5时的recall
            p_curve[ci] = np.interp(-x, -scores[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + self.eps)

        i = self.smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + self.eps) - tp).round()  # false positives

        self.results["tp"] = tp
        self.results["fp"] = fp
        self.results["p"] = p
        self.results["r"] = r
        self.results["f1"] = f1
        self.results["ap"] = ap
        self.results["unique_classes"] = unique_classes.astype(int)
        self.results["p_curve"] = p_curve
        self.results["r_curve"] = r_curve
        self.results["f1_curve"] = f1_curve
        self.results["x"] = x  # x轴刻度
        self.results["prec_values"] = prec_values  # y轴刻度

    def p(self):
        return self.results["p"] if len(self.results["p"]) else []

    def r(self):
        return self.results["r"] if len(self.results["r"]) else []

    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.results["ap"][:, 0] if len(self.results["ap"]) else []

    def ap(self):
        """
           Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

           Returns:
               (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
           """
        return self.results["ap"].mean(1) if len(self.results["ap"]) else []

    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.results["p"].mean() if len(self.results["p"]) else 0.0

    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.results["r"].mean() if len(self.results["r"]) else 0.0

    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.results["ap"][:, 0].mean() if len(self.results["ap"]) else 0.0

    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.results["ap"][:, 5].mean() if len(self.results["ap"]) else 0.0

    def map50_95(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.results["ap"].mean() if len(self.results["ap"]) else 0.0

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False) -> np.ndarray:
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (np.ndarray): Predicted class indices of shape(N,).
            true_classes (np.ndarray): Target class indices of shape(M,).
            iou (np.ndarray): An MxN tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (np.ndarray): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes

        for i, threshold in enumerate(self.iouv):
            if use_scipy:
                import scipy
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True

        return correct.astype(bool)

    def plot(self, save_dir: Path, plot_pr=True, plot_f1=True, plot_p=True, plot_r=True):
        if self.class_names:
            names = [v for k, v in self.class_names.items() if k in self.results["unique_classes"]]  # list: only classes that have data
            names = dict(enumerate(names))
        else:
            names = dict(enumerate(self.results["unique_classes"]))

        if len(names) == 0:
            LOGGER.info("No classes to plot.")
            return

        if plot_pr:
            self.plot_PR_curve(names, save_dir)
        if plot_f1:
            self.plot_F1_curve(names, save_dir)
        if plot_p:
            self.plot_P_curve(names, save_dir)
        if plot_r:
            self.plot_R_curve(names, save_dir)

    def plot_PR_curve(self, names, save_dir: Path):
        """
        Plot Precision-Recall curve for each class, using AP values.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, c in enumerate(self.results["unique_classes"]):
            ap_value = self.results["ap"][i, 0]  # AP@0.5 for the current class
            ax.plot(self.results["r_curve"][i], self.results["p_curve"][i], label=f'{names[i]} (AP@0.5: {ap_value:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"({self.prefix})precision_recall_curve.png", dpi=150)
        plt.close()

    def plot_F1_curve(self, names, save_dir: Path):
        """
        Plot F1 curve for each class.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, c in enumerate(self.results["unique_classes"]):
            ax.plot(self.results["x"], self.results["f1_curve"][i], label=f'{names[i]} (F1: {self.results["f1"][i]:.2f})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Curve')
        ax.legend()
        ax.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"({self.prefix})f1_score_curve.png", dpi=150)
        plt.close()

    def plot_P_curve(self, names, save_dir: Path):
        """
        Plot Precision curve for each class.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, c in enumerate(self.results["unique_classes"]):
            ax.plot(self.results["x"], self.results["p_curve"][i], label=f'{names[i]} (P: {self.results["p"][i]:.2f})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Precision')
        ax.set_title('Precision Curve')
        ax.legend()
        ax.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"({self.prefix})precision_curve.png", dpi=150)
        plt.close()

    def plot_R_curve(self, names, save_dir: Path):
        """
        Plot Recall curve for each class.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, c in enumerate(self.results["unique_classes"]):
            ax.plot(self.results["x"], self.results["r_curve"][i], label=f'{names[i]} (R: {self.results["r"][i]:.2f})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Recall')
        ax.set_title('Recall Curve')
        ax.legend()
        ax.grid(True)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir / f"({self.prefix})recall_curve.png", dpi=150)
        plt.close()

    @staticmethod
    def compute_ap(recall, precision):
        """
        Compute the average precision (AP) given the recall and precision curves.

        Args:
            recall: The recall curve.
            precision: The precision curve.

        Returns:
            (float): Average precision.
            (np.ndarray): Precision envelope curve.
            (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = "interp"  # methods: 'continuous', 'interp'
        if method == "interp":
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapezoid(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    @staticmethod
    def smooth(y, f=0.05):
        """Box filter of fraction f."""
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
