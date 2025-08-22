# tinytrain/metrics/pose_metrics.py
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from mmeval import PCKAccuracy, EndPointError, KeypointAUC, KeypointNME

from tinytrain.metrics import BaseMetric
from tinytrain.utils import LOGGER
from tinytrain.global_var import RANK, WORLD_SIZE


class PoseMetrics(BaseMetric):
    """
    姿态估计任务统一评估指标封装类。

    功能
    ----
    1. PCK 指标（PCK@0.05/0.1/0.2 ...）
    2. AUC（PCK-AUC）
    3. EPE（平均端点误差）
    4. NME（归一化平均误差）
    5. 支持类别级指标
    6. 支持 PCK-α 曲线绘制
    7. 支持分布式同步
    """

    def __init__(
        self,
        num_keypoints: int,
        class_metrics: bool = False,
        class_names: Optional[List[str]] = None,
        norm_factor: float = 1.0,
        auc_alpha: Sequence[float] = tuple(np.arange(0.0, 0.5, 0.01)),
    ):
        """
        Args:
            num_keypoints: 关键点总数
            class_metrics: 是否计算类别级指标
            class_names: 类别名称列表，用于可视化
            norm_factor: NME 用的归一化因子（例如 bbox 对角线长度）
            auc_alpha: 计算 AUC 时的 α 阈值采样序列
        """
        super().__init__()

        self.num_keypoints = num_keypoints
        self.class_metrics = class_metrics
        self.class_names = class_names
        self.norm_factor = norm_factor
        self.auc_alpha = auc_alpha

        # PCK 指标（阈值 0.05/0.1/0.2）
        self.pck05 = PCKAccuracy(thr=0.05, keypoint_indices=None)
        self.pck10 = PCKAccuracy(thr=0.10, keypoint_indices=None)
        self.pck20 = PCKAccuracy(thr=0.20, keypoint_indices=None)

        # AUC / EPE / NME
        self.auc = KeypointAUC(alphas=self.auc_alpha)
        self.epe = EndPointError()
        self.nme = KeypointNME(norm_mode="keypoint_distance", norm_factor=norm_factor)

        # 缓存计算结果
        self.results = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self):
        """重置所有内部状态，开始新一轮评估。"""
        self.results = None
        self.pck05.reset()
        self.pck10.reset()
        self.pck20.reset()
        self.auc.reset()
        self.epe.reset()
        self.nme.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """
        每迭代一次调用一次。

        Args:
            pred: 模型输出关键点坐标 [B, K, 2]，浮点
            target: 真实关键点坐标 [B, K, 2]，浮点
            mask: 有效关键点掩码 [B, K]，bool/uint8
        """
        # 1. 转 numpy（mmeval 默认 numpy 输入）
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)

        # 2. 更新指标
        self.pck05.add(pred_np, target_np, mask_np)
        self.pck10.add(pred_np, target_np, mask_np)
        self.pck20.add(pred_np, target_np, mask_np)
        self.auc.add(pred_np, target_np, mask_np)
        self.epe.add(pred_np, target_np, mask_np)
        self.nme.add(pred_np, target_np, mask_np)

    def compute(self):
        """计算并缓存最终指标。"""
        # 只在 rank0 真正计算
        if RANK in {-1, 0}:
            self.results = {
                "PCK@0.05": self.pck05.evaluate(),
                "PCK@0.10": self.pck10.evaluate(),
                "PCK@0.20": self.pck20.evaluate(),
                "AUC": self.auc.evaluate(),
                "EPE": self.epe.evaluate(),
                "NME": self.nme.evaluate(),
            }
            results_to_send = self.results
        else:
            results_to_send = None

        # 广播给所有 rank
        if WORLD_SIZE > 1:
            obj_list = [results_to_send]
            dist.broadcast_object_list(obj_list, src=0)
            self.results = obj_list[0]

    # ------------------------------------------------------------------
    # 指标访问
    # ------------------------------------------------------------------
    def pck05_score(self) -> float:
        """返回 PCK@0.05"""
        return self.results["PCK@0.05"]["PCK"] if self.results else 0.0

    def pck10_score(self) -> float:
        """返回 PCK@0.10"""
        return self.results["PCK@0.10"]["PCK"] if self.results else 0.0

    def pck20_score(self) -> float:
        """返回 PCK@0.20"""
        return self.results["PCK@0.20"]["PCK"] if self.results else 0.0

    def auc_score(self) -> float:
        """返回 PCK-AUC"""
        return self.results["AUC"]["AUC"] if self.results else 0.0

    def epe_score(self) -> float:
        """返回平均端点误差 EPE"""
        return self.results["EPE"]["EPE"] if self.results else 0.0

    def nme_score(self) -> float:
        """返回归一化平均误差 NME"""
        return self.results["NME"]["NME"] if self.results else 0.0

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def plot_pck_curve(self, save_dir: Path):
        """绘制 PCK-α 曲线"""
        if not self.results:
            LOGGER.warning("PoseMetrics.compute() 尚未被调用，跳过绘图。")
            return

        auc_dict = self.results["AUC"]
        alphas = auc_dict["alphas"]
        pck_per_alpha = auc_dict["PCKs"]  # shape [num_alphas]

        plt.figure(figsize=(8, 6))
        sns.lineplot(x=alphas, y=pck_per_alpha, linewidth=2.5)
        plt.title("PCK-α Curve")
        plt.xlabel("Threshold α")
        plt.ylabel("PCK")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "PCK_curve.png", dpi=150)
        plt.close()

    def plot_epe_hist(self, save_dir: Path):
        """绘制 EPE 分布直方图"""
        if not self.results:
            return

        epe_dict = self.results["EPE"]
        epe_per_kpt = epe_dict["EPE_per_keypoint"]  # [K]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=np.arange(len(epe_per_kpt)), y=epe_per_kpt, palette="Blues_d")
        plt.title("EPE Distribution Per Keypoint")
        plt.xlabel("Keypoint ID")
        plt.ylabel("EPE")
        plt.tight_layout()
        plt.savefig(save_dir / "EPE_hist.png", dpi=150)
        plt.close()

    def plot(self, save_dir: Path):
        """一键绘制所有曲线"""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_pck_curve(save_dir)
        self.plot_epe_hist(save_dir)