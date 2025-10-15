from __future__ import annotations

import numpy as np
import torch

from typing import TYPE_CHECKING

from tinytrain.models.ocr.ocr_data_format import LPRBatchDataInfo
from tinytrain.engine import TTBaseValidator
from tinytrain.global_var import RANK
from tinytrain.utils.progress_bar import TTProgressBar

if TYPE_CHECKING:
    from tinytrain.engine import TTBaseTrainer


class LPRValidator(TTBaseValidator):
    def __init__(self, trainer: TTBaseTrainer, world_size: int):
        super().__init__(trainer, world_size)
        self.num_classes = self.config_manager.dataset["nc"]
        self.chars = list(self.config_manager.dataset["names"].values())

        self.tp = 0
        self.tn_len = 0
        self.tn_chr = 0
        self.acc = 0.

    def preprocess(self, batch_samples: LPRBatchDataInfo) -> LPRBatchDataInfo:
        mean = self.config_manager.augment["mean"]
        std = self.config_manager.augment["std"] + 1e-8
        batch_samples.data = ((batch_samples.data.to(self.device, non_blocking=True).float() / 255.0) - mean) / std
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        return batch_samples

    def postprocess(self, preds: list[torch.Tensor]) -> torch.Tensor:
        pred = preds[0]
        return pred

    def start_metrics_on_training(self, pbar: TTProgressBar):
        self.tp = 0
        self.tn_len = 0
        self.tn_chr = 0
        self.acc = 0.

    def update_metrics_on_training(self, outputs: torch.Tensor, batch_samples: LPRBatchDataInfo, pbar: TTProgressBar):
        tp, tn_len, tn_chr = self.eval_batch(outputs, batch_samples.target, batch_samples.lengths, blank_id=len(self.chars) - 1)
        self.tp += tp
        self.tn_len += tn_len
        self.tn_chr += tn_chr

        # log update
        desc = f"{'val':^5}|{'classes':^15}|{'Acc':^15}|{"tn_len":^15}|{"tn_chr":^15}|"
        pbar.set_description(desc)

    def end_metrics_on_training(self, pbar: TTProgressBar):
        # metrics result
        self.acc = self.tp * 1.0 / (self.tp + self.tn_len + self.tn_chr)

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{self.num_classes:^15}|{self.acc:^15.3f}|{self.tn_len:^15}|{self.tn_chr:^15}|"
            print(progress_str)

            if self.trainer.train_result is not None:
                self.trainer.train_result.add("accuracy", self.acc)
                self.trainer.train_result.add("tn_len", self.tn_len)
                self.trainer.train_result.add("tn_chr", self.tn_chr)

    def start_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.start_metrics_on_training(pbar)

    def update_metrics_on_train_completed(self, outputs: torch.Tensor, batch_samples: LPRBatchDataInfo, pbar: TTProgressBar):
        self.update_metrics_on_training(outputs, batch_samples, pbar)

    def end_metrics_on_train_completed(self, pbar: TTProgressBar):
        self.acc = self.tp * 1.0 / (self.tp + self.tn_len + self.tn_chr)

        if RANK in {-1, 0}:
            # log
            progress_str = f"{'val':^5}|{self.num_classes:^15}|{self.acc:^15.3f}|{self.tn_len:^15}|{self.tn_chr:^15}|"
            print(progress_str)

    def get_fitness(self) -> float:
        return self.acc

    def greedy_decode_ctc(self, outputs: torch.Tensor, blank_id: int) -> list[list[int]]:
        """
        对 CTC 输出做贪心解码：去 blank + 去重复
        Args:
            outputs: (batch, num_classes, time_steps)  未归一化得分
            blank_id: blank 下标
        Returns:
            List[List[int]]  解码后的标签序列
        """
        batch_size, num_classes, time_steps = outputs.shape
        decoded: list[list[int]] = []

        for b in range(batch_size):
            # 1. 每帧取最大
            best_path = [int(torch.argmax(outputs[b, :, t])) for t in range(time_steps)]

            # 2. 去 blank + 去重复
            dedup = []
            prev = blank_id
            for t, label in enumerate(best_path):
                if label == blank_id:  # 去 blank
                    prev = blank_id
                    continue
                if label == prev:  # 去重复
                    continue
                dedup.append(label)
                prev = label
            decoded.append(dedup)
        return decoded

    def eval_batch(self, outputs: torch.Tensor, flat_targets: torch.Tensor, target_lengths: list[int], blank_id: int) -> tuple[int, int, int]:
        """
        返回 (tp, tn_len, tn_chr)
        tn_len: 长度就不对的样本数
        tn_chr: 长度对但字符错的样本数
        """
        # 1. 把扁平标签拆成 list[list[int]]
        targets: list[list[int]] = []
        start = 0
        for leng in target_lengths:
            targets.append(flat_targets[start:start + leng].tolist())
            start += leng

        # 2. 贪心解码
        preds = self.greedy_decode_ctc(outputs, blank_id)

        # 3. 统计
        tp = tn_len = tn_chr = 0
        for gold, pred in zip(targets, preds):
            if len(gold) != len(pred):
                tn_len += 1
            elif gold == pred:
                tp += 1
            else:
                tn_chr += 1
        return tp, tn_len, tn_chr
