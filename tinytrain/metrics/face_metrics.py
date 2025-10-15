import numpy as np
import sklearn

from matplotlib import pyplot as plt
from pathlib import Path
from scipy import interpolate
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .base import TTBaseMetric


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


class FaceRecognitionMetrics(TTBaseMetric):
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

    def __init__(self, nrof_folds=10, pca=0):
        """
        初始化空缓冲区与指标占位变量。
        """
        super().__init__()
        self.nrof_folds = nrof_folds
        self.pca = pca

        self.embeddings1 = []
        self.embeddings2 = []
        self.matchs = []

        self.results = {}

    def reset(self):
        """
        清空缓冲区与所有计算结果，支持多轮复用。
        """
        self.embeddings1 = []
        self.embeddings2 = []
        self.matchs = []
        self.results = {}

    def update(self, embedding1: np.ndarray, embedding2: np.ndarray, match: np.ndarray):
        """
        批量追加一次推理结果。

        Args:
            scores (np.ndarray): 一维相似度得分。
            labels (np.ndarray): 一维 0/1 标签，0 表示负样本，1 表示正样本。
        """
        self.embeddings1.append(embedding1)
        self.embeddings2.append(embedding2)
        self.matchs.append(match)

    def compute(self):
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = np.concatenate(self.embeddings1)
        embeddings2 = np.concatenate(self.embeddings2)
        actual_issame = np.concatenate(self.matchs)

        tpr, fpr, accuracy = self._calculate_roc(thresholds,
                                                 embeddings1,
                                                 embeddings2,
                                                 np.asarray(actual_issame),
                                                 nrof_folds=self.nrof_folds,
                                                 pca=self.pca)
        thresholds = np.arange(0, 4, 0.001)
        val, val_std, far = self._calculate_val(thresholds,
                                                embeddings1,
                                                embeddings2,
                                                np.asarray(actual_issame),
                                                1e-3,
                                                nrof_folds=self.nrof_folds)

        self.results["tpr"] = tpr
        self.results["fpr"] = fpr
        self.results["accuracy"] = accuracy
        self.results["val"] = val
        self.results["val_std"] = val_std
        self.results["far"] = far

    def avg_accuracy(self) -> float:
        """跨折平均 Accuracy（Youden 最优阈值处已折平均）"""
        if "accuracy" not in self.results:
            raise RuntimeError("accuracy 未计算，请先调用 compute()")
        return float(np.mean(self.results["accuracy"]))  # 已是 (n_folds,) 均值

    def tpr_at_far1e3(self) -> float:
        """TPR@FAR=1e-3（跨折平均）"""
        return float(self.val())  # compute() 里已折平均

    def tpr_at_far1e4(self) -> float:
        """可选计算TPR@FAR=1e-4（跨折平均）"""
        # 利用已有 _calculate_val 接口，far_target=1e-4
        if not {"embeddings1", "embeddings2", "matchs"}.issubset(vars(self)):
            raise RuntimeError("原始数据缺失，无法计算 TPR@FAR=1e-4")
        thresholds = np.arange(0, 4, 0.001)
        val_mean, _, _ = self._calculate_val(
            thresholds,
            np.vstack(self.embeddings1),
            np.vstack(self.embeddings2),
            np.hstack(self.matchs),
            far_target=1e-4,
            nrof_folds=self.nrof_folds
        )
        return float(val_mean)

    def best_threshold(self) -> float:
        """Youden 指数最优阈值（跨折平均 ROC 上）"""
        if {"tpr", "fpr"} - self.results.keys():
            raise RuntimeError("TPR/FPR 未计算，请先调用 compute()")
        tpr = self.tpr()
        fpr = self.fpr()
        th = np.arange(0, 4, 0.01)  # 与 compute() 保持一致
        youden = tpr - fpr
        return float(th[np.argmax(youden)])

    def tpr(self):
        return self.results.get("tpr", np.zeros(400))  # 400 与 compute 里阈值数一致

    def fpr(self):
        return self.results.get("fpr", np.zeros(400))

    def accuracy(self):
        # accuracy 在 compute 里是 (n_folds, n_thresholds) -> 返回 (n_folds, 400)
        return self.results.get("accuracy", np.zeros((10, 400)))  # 10 折

    def val(self):
        return self.results.get("val", 0.0)  # 标量，可以返回 0

    def val_std(self):
        return self.results.get("val_std", 0.0)

    def far(self):
        return self.results.get("far", 0.0)

    def plot(self, save_dir: Path):
        """
        save_dir : 若给定，则将每张图存为 <save_dir>/<name>.png
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        self._plot_roc(save_dir)
        self._plot_det(save_dir)
        self._plot_tpr_at_far1e3(save_dir)
        self._plot_acc_vs_th(save_dir)
        self._plot_tpr_fpr_vs_th(save_dir)
        self._plot_far_val_vs_th(save_dir)

    # ------------------------------------------------------------------
    # 1. ROC
    # ------------------------------------------------------------------
    def _plot_roc(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        auc = metrics.auc(self.fpr(), self.tpr())
        ax.plot(self.fpr(), self.tpr(), lw=2, label=f'AUC={auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
        ax.legend()
        ax.grid()
        fig.savefig(save_dir / 'ROC_Curve.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 2. DET  (FRR=1-TPR)
    # ------------------------------------------------------------------
    def _plot_det(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(self.fpr(), 1. - self.tpr(), lw=2)
        ax.set(xlabel='FAR', ylabel='FRR', title='DET Curve')
        ax.grid(True, which='both')
        fig.savefig(save_dir / 'DET_FRRvsFAR_Curve.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 3. TPR@FAR=1e-3 横线
    # ------------------------------------------------------------------
    def _plot_tpr_at_far1e3(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        tar, std = self.val(), self.val_std()
        ax.axhline(tar, c='r', lw=2, label=f'{tar:.4f}±{std:.4f}')
        ax.axhline(tar + std, c='r', ls='--', lw=1)
        ax.axhline(tar - std, c='r', ls='--', lw=1)
        ax.set(ylim=(0, 1), ylabel='TPR', title='TPR@FAR=1e-3')
        ax.legend()
        ax.grid()
        fig.savefig(save_dir / 'TPRatFAR1e-3_Curve.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 4. Accuracy vs Threshold
    # ------------------------------------------------------------------
    def _plot_acc_vs_th(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        th = np.arange(0, 4, 0.01)  # 400 点
        acc_mean = np.mean(self.accuracy())  # 标量
        acc_std = np.std(self.accuracy())  # 标量

        # 画一条水平线即可
        ax.axhline(acc_mean, c='C0', lw=2, label=f'Accuracy={acc_mean:.4f}±{acc_std:.4f}')
        ax.fill_between(th, acc_mean - acc_std, acc_mean + acc_std,
                        color='C0', alpha=0.2)
        ax.set(xlabel='Distance threshold', ylabel='Accuracy',
               title='Accuracy vs Threshold')
        ax.legend()
        ax.grid()
        fig.savefig(save_dir / 'Accuracy_vs_Threshold_Curve.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 5. TPR & FPR vs Threshold
    # ------------------------------------------------------------------
    def _plot_tpr_fpr_vs_th(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        th = np.linspace(0, 4, len(self.tpr()))
        ax.plot(th, self.tpr(), lw=2, label='TPR')
        ax.plot(th, self.fpr(), lw=2, label='FPR')
        ax.set(xlabel='Distance threshold', ylabel='Rate',
               title='TPR & FPR vs Threshold')
        ax.legend()
        ax.grid()
        fig.savefig(save_dir / 'TPR_FPR_vs_Threshold_Curve.png', dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 6. FAR & VAL vs Threshold
    # ------------------------------------------------------------------
    def _plot_far_val_vs_th(self, save_dir: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        th = np.arange(0, 4, 0.001)
        dist = np.sum(np.square(np.subtract(np.vstack(self.embeddings1),
                                            np.vstack(self.embeddings2))), 1)
        label = np.hstack(self.matchs)

        far_curve, val_curve = [], []
        for t in th:
            v, f = self._calculate_val_far(t, dist, label)
            val_curve.append(v)
            far_curve.append(f)

        ax.plot(th, val_curve, lw=2, label='VAL (TPR)')
        ax.plot(th, far_curve, lw=2, label='FAR')
        idx = np.argmin(np.abs(np.array(far_curve) - 1e-3))
        ax.axvline(th[idx], c='r', ls='--', label=f'FAR=1e-3 th={th[idx]:.3f}')
        ax.set(xlabel='Distance threshold', ylabel='Rate',
               title='FAR & VAL vs Threshold')
        ax.legend()
        ax.grid()
        fig.savefig(save_dir / 'FAR_VAL_vs_Threshold_Curve.png', dpi=150)
        plt.close()

    def _calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
        assert (embeddings1.shape[0] == embeddings2.shape[0])
        assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        if pca == 0:
            diff = np.subtract(embeddings1, embeddings2)
            dist = np.sum(np.square(diff), 1)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if pca > 0:
                embed1_train = embeddings1[train_set]
                embed2_train = embeddings2[train_set]
                _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
                pca_model = PCA(n_components=pca)
                pca_model.fit(_embed_train)
                embed1 = pca_model.transform(embeddings1)
                embed2 = pca_model.transform(embeddings2)
                embed1 = sklearn.preprocessing.normalize(embed1)
                embed2 = sklearn.preprocessing.normalize(embed2)
                diff = np.subtract(embed1, embed2)
                dist = np.sum(np.square(diff), 1)

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = self._calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = self._calculate_accuracy(
                    threshold, dist[test_set],
                    actual_issame[test_set])
            _, _, accuracy[fold_idx] = self._calculate_accuracy(
                thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy

    def _calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(
            np.logical_and(np.logical_not(predict_issame),
                           np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc

    def _calculate_val(self, thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
        assert (embeddings1.shape[0] == embeddings2.shape[0])
        assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        indices = np.arange(nrof_pairs)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = self._calculate_val_far(
                    threshold, dist[train_set], actual_issame[train_set])

            # --------- 防重复插值保护 ---------
            uniq_far, idx = np.unique(far_train, return_index=True)
            uniq_th = thresholds[idx]
            if uniq_far.size < 2:
                threshold = float(uniq_th[-1] if np.max(far_train) < far_target else uniq_th[0])
            else:
                f = interpolate.interp1d(uniq_far, uniq_th, kind='slinear',
                                         bounds_error=False, fill_value=(uniq_th[0], uniq_th[-1]))
                threshold = float(f(far_target))
            # ----------------------------------

            val[fold_idx], far[fold_idx] = self._calculate_val_far(
                threshold, dist[test_set], actual_issame[test_set])

        val_mean = np.mean(val)
        far_mean = np.mean(far)
        val_std = np.std(val)
        return val_mean, val_std, far_mean

    def _calculate_val_far(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(
            np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))
        val = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
        return val, far
