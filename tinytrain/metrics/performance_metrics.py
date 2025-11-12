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

import os
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

from tinytrain.metrics.base import TTBaseMetric
from tinytrain.utils import LOGGER


class PerformanceMetrics(TTBaseMetric):
    """
    通用性能评估器。

    一次性收集所有图片的预测结果（tp/score/pred_class/target_class），
    在 compute() 中按类构建 PR 曲线，计算 AP50、AP75、AP50-95、mAP、F1 等指标，
    并提供 plot 系列方法将 P/R/F1/PR 曲线保存到本地。
    """

    def __init__(self, prefix: str = "", class_names: dict[int, str] = None):
        """
        初始化评估器。

        Args:
            prefix (str, optional):
                文件名前缀，用于保存图表时区分不同实验。默认为空字符串。
            class_names (dict[int, str] | None, optional):
                {class_id: class_name} 映射，仅用于绘图时显示可读标签。默认为 None。
        """
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

    def reset(self):
        """
        清空所有缓存的预测结果与计算结果，准备开始新一轮评估。
        """
        self._tp = []
        self._scores = []
        self._pred_classes = []
        self._target_classes = []
        self.results = {}

    def update(self, tp: np.ndarray, score: np.ndarray, pred_class: np.ndarray, target_class: np.ndarray):
        """
        单张图片的预测结果写入缓存。

        Args:
            tp (np.ndarray):
                形状 (D, 10) 的 bool 数组，D 个检测框在 10 个 IoU 阈值下是否匹配。
            score (np.ndarray):
                形状 (D,) 的 float 数组，置信度分数。
            pred_class (np.ndarray):
                形状 (D,) 的 int 数组，预测类别 id。
            target_class (np.ndarray):
                形状 (L,) 的 int 数组，真值类别 id。
        """
        self._tp.append(tp)
        self._scores.append(score)
        self._pred_classes.append(pred_class)
        self._target_classes.append(target_class)

    def compute(self):
        """
        汇总所有 update 进来的数据，按类计算 PR 曲线、AP、F1 曲线及最佳 F1 阈值下的 P/R/F1，
        结果存入 self.results 字典，供后续指标函数与绘图函数使用。
        """
        if not len(self._tp):
            return

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
        """
        返回各类在最佳 F1 阈值下的精确率数组。

        Returns:
            np.ndarray:
                形状 (nc,) 的 float 数组，nc 为存在数据的类别数；若无数据返回空数组。
        """
        return self.results["p"] if self.results and len(self.results["p"]) else np.array([])

    def r(self):
        """
        返回各类在最佳 F1 阈值下的召回率数组。

        Returns:
            np.ndarray:
                形状 (nc,) 的 float 数组；若无数据返回空数组。
        """
        return self.results["r"] if self.results and len(self.results["r"]) else np.array([])

    def ap50(self):
        """
        返回各类 AP@IoU=0.5 的数组。

        Returns:
            np.ndarray:
                形状 (nc,) 的 float 数组；若无数据返回空数组。
        """
        return self.results["ap"][:, 0] if self.results and len(self.results["ap"]) else np.array([])

    def ap(self):
        """
        返回各类 AP@IoU=0.5:0.95 的均值数组（即 AP50-95）。

        Returns:
            np.ndarray:
                形状 (nc,) 的 float 数组；若无数据返回空数组。
        """
        return self.results["ap"].mean(1) if self.results and len(self.results["ap"]) else np.array([])

    def mp(self):
        """
        返回所有类别的平均精确率 (mean Precision)。

        Returns:
            float:
                平均精确率；若无数据返回 0.0。
        """
        return self.results["p"].mean() if self.results and len(self.results["p"]) else 0.0

    def mr(self):
        """
        返回所有类别的平均召回率 (mean Recall)。

        Returns:
            float:
                平均召回率；若无数据返回 0.0。
        """
        return self.results["r"].mean() if self.results and len(self.results["r"]) else 0.0

    def map50(self):
        """
        返回 AP50 的均值 (mAP@0.5)。

        Returns:
            float:
                mAP@0.5；若无数据返回 0.0。
        """
        return self.results["ap"][:, 0].mean() if self.results and len(self.results["ap"]) else 0.0

    def map75(self):
        """
        返回 AP75 的均值 (mAP@0.75)。

        Returns:
            float:
                mAP@0.75；若无数据返回 0.0。
        """
        return self.results["ap"][:, 5].mean() if self.results and len(self.results["ap"]) else 0.0

    def map50_95(self):
        """
        返回 AP50-95 的均值 (mAP@0.5:0.95)。

        Returns:
            float:
                mAP@0.5:0.95；若无数据返回 0.0。
        """
        return self.results["ap"].mean() if self.results and len(self.results["ap"]) else 0.0

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False) -> np.ndarray:
        """
        将预测框与真值框按类别+IoU 进行匹配，输出 (D,10) 的匹配矩阵。

        Args:
            pred_classes (np.ndarray):
                形状 (D,) 的 int 数组，预测类别。
            true_classes (np.ndarray):
                形状 (L,) 的 int 数组，真值类别。
            iou (np.ndarray):
                形状 (L, D) 的 float 数组，两两 IoU。
            use_scipy (bool, optional):
                是否使用 scipy.linear_sum_assignment 做最优匹配；默认 False 用贪心法。

        Returns:
            np.ndarray:
                形状 (D, 10) 的 bool 数组，每个检测框在 10 个 IoU 阈值下是否算 TP。
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
        """
        一次性绘制并保存 PR、F1、P、R 四条曲线到指定目录。

        Args:
            save_dir (Path):
                保存图片的文件夹路径。
            plot_pr (bool, optional):
                是否绘制 PR 曲线，默认 True。
            plot_f1 (bool, optional):
                是否绘制 F1 曲线，默认 True。
            plot_p (bool, optional):
                是否绘制 Precision 曲线，默认 True。
            plot_r (bool, optional):
                是否绘制 Recall 曲线，默认 True。
        """
        if self.class_names:
            if self.results:
                names = [v for k, v in self.class_names.items() if k in self.results["unique_classes"]]  # list: only classes that have data
                names = dict(enumerate(names))
            else:
                names = {}
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
        绘制所有类别的 P-R 曲线并保存。

        Args:
            names (dict[int, str]):
                {类别索引: 可读名称}，用于图例。
            save_dir (Path):
                保存图片的文件夹路径。
        """
        if not self.results:
            LOGGER.info("No data available for PR curve.")
            return

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
        绘制所有类别的 F1-Confidence 曲线并保存。

        Args:
            names (dict[int, str]):
                类别名称映射。
            save_dir (Path):
                保存文件夹。
        """
        if not self.results:
            LOGGER.info("No data available for PR curve.")
            return

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
        绘制所有类别的 Precision-Confidence 曲线并保存。

        Args:
            names (dict[int, str]):
                类别名称映射。
            save_dir (Path):
                保存文件夹。
        """
        if not self.results:
            LOGGER.info("No data available for PR curve.")
            return

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
        绘制所有类别的 Recall-Confidence 曲线并保存。

        Args:
            names (dict[int, str]):
                类别名称映射。
            save_dir (Path):
                保存文件夹。
        """
        if not self.results:
            LOGGER.info("No data available for PR curve.")
            return

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
        根据单类 recall/precision 曲线计算 AP（11 点插值或连续积分法）。

        Args:
            recall (np.ndarray):
                召回率曲线，形状 (N,)。
            precision (np.ndarray):
                精确率曲线，形状 (N,)。

        Returns:
            tuple[float, np.ndarray, np.ndarray]:
                ap:      标量，平均精度。
                mpre:    精度包络曲线，形状 (N+2,)。
                mrec:    带哨兵值的召回曲线，形状 (N+2,)。
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
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    @staticmethod
    def smooth(y, f=0.05):
        """
        对 1D 数组 y 做简单滑动平均（box filter），用于平滑 F1 曲线找峰值。

        Args:
            y (np.ndarray):
                原始 1D 曲线。
            f (float, optional):
                平滑窗口占长度比例，默认 0.05。

        Returns:
            np.ndarray:
                平滑后的曲线，长度与 y 相同。
        """
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
