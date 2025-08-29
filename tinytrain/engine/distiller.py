from __future__ import annotations

import time
import torch
import torch.distributed as dist

from typing import TYPE_CHECKING
from pathlib import Path
from torch import autocast, nn

from tinytrain.global_var import LOCAL_RANK
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.global_var import RANK
from tinytrain.utils.callback import Callback

from .trainer import BaseTrainer

if TYPE_CHECKING:
    from .model import BaseModel


class Distiller(BaseTrainer):
    def __init__(self, config_manager: ConfigManager, student_model: BaseModel, teacher_model: BaseModel, callback: Callback, main_script_path: Path = None):
        super().__init__(config_manager=config_manager, model=student_model, callback=callback, main_script_path=main_script_path)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

    # ------------------------------------------------------------------
    # 以下建议子类可重写的方法
    # ------------------------------------------------------------------
    def model_inference_and_loss_calculate(self, teacher_model, student_model, inputs) -> tuple[float, dict]:
        # 计算教师模型的输出
        teacher_class_scores, teacher_bbox_coords = teacher_model(inputs)
        # 计算学生模型的输出
        student_class_scores, student_bbox_coords = student_model(inputs)

        # 计算知识蒸馏的损失
        loss, loss_items = self.sample_distillation_loss(student_class_scores, student_bbox_coords, teacher_class_scores, teacher_bbox_coords, temperature=5.0, alpha=0.5)
        return loss, loss_items

    def sample_distillation_loss(self, student_class_scores, student_bbox_coords, teacher_class_scores, teacher_bbox_coords, temperature, alpha):
        # 计算类别置信度的KL散度损失
        student_softmax = nn.functional.softmax(student_class_scores / temperature, dim=1)
        teacher_softmax = nn.functional.softmax(teacher_class_scores / temperature, dim=1)
        kl_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_class_scores / temperature, dim=1),
                                 teacher_softmax)

        # 计算边界框的L1损失
        bbox_loss = nn.L1Loss()(student_bbox_coords, teacher_bbox_coords)

        # 总损失是KL散度损失和边界框损失的加权和
        total_loss = alpha * kl_loss + (1 - alpha) * bbox_loss

        loss_items = {"kl_loss": kl_loss.deatch(), "bbox_loss": bbox_loss.deatch()}
        return total_loss, loss_items

    # ------------------------------------------------------------------
    # 以下不建议子类重写的方法
    # ------------------------------------------------------------------
    def freeze_layers(self, world_size: int):
        """
        冻结教师模型模型，防止其参数在训练中被更新。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _do_train(self, world_size: int):
        """
        执行模型训练主循环，包括前向、反向、验证、保存等。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        # 训练前检查
        self._before_train(world_size)

        # 只做验证
        if self.config_manager.core["only_val"]:
            self.only_do_validate()
            return

        current_epoch = self.start_epoch  # 一个epoch有几个batch
        num_batch = len(self.train_dataloader)  # 整个训练所有epoch一共有多少个batch
        last_opt_step = -1  # 记录上一次真正更新参数的 batch 序号

        # 先清一次显存,梯度归零
        self._clear_memory()
        self.optimizer.zero_grad()

        # 正式训练
        LOGGER.info(f"start training...")
        train_time_start = time.time()
        self.callbacks.run_callback(self, "on_train_start")
        while True:
            self.callbacks.run_callback(self, "on_train_epoch_start")
            self.model.train()

            # dataloader sampler
            if world_size > 1:
                self.train_dataloader.sampler.set_epoch(current_epoch)

            # 设置打印进度条
            pbar = TTProgressBar(self.train_dataloader, total=num_batch)

            # 开启epoch循环
            smooth_loss_items = None
            for i, batch_samples in enumerate(pbar):
                self.callbacks.run_callback(self, "on_train_batch_start")

                # 当前是第几个batch
                current_batch = current_epoch * num_batch + i

                # 判断本次是否真正执行 optimizer.step()
                is_last_accum_step = (current_batch - last_opt_step) == self.accumulate

                dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() and self.config_manager.core["bf16"] else torch.float16
                if world_size > 1:
                    # 同步AMP dtype
                    dtype_list = [dtype] if RANK == 0 else [None]
                    dist.broadcast_object_list(dtype_list, src=0)
                    dtype = dtype_list[0]

                # move data to device
                batch_samples = self.preprocess_data(batch_samples)  # type: ignore[arg-type]

                # Forward
                with autocast(device_type=self.device.type, dtype=dtype, enabled=self.amp):
                    self.loss, self.loss_items = self.model_inference_and_loss_calculate(self.teacher_model, self.model, batch_samples.data)

                # 添加 L1 正则化
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                self.loss += self.config_manager.core["l1_norm"] * l1_norm

                # loss 缩放为了在 accumulate 次平均后等价
                self.loss = self.loss / self.accumulate

                # Backward
                self.scaler.scale(self.loss).backward()

                # 梯度累积够了，就更新参数
                if is_last_accum_step:
                    self.optimizer_step()
                    last_opt_step = current_batch

                # 对打印的loss做平滑
                if smooth_loss_items is None:
                    smooth_loss_items = self.loss_items
                else:
                    for loss_name, loss_value in self.loss_items.items():
                        smooth_loss_items[loss_name] = (smooth_loss_items[loss_name] * i + self.loss_items[loss_name]) / (i + 1)

                if RANK in {-1, 0}:
                    # log
                    s_loss_values = f"|".join(f"{v:^15.3f}" for v in list(smooth_loss_items.values()))
                    if i == 0:
                        s_loss_names = f"|".join(f"{name:^15}" for name in list(smooth_loss_items.keys()))
                        pbar.set_title(f"\n{'train':^5}|"
                                       f"{'batch':^15}|"
                                       f"{'epoch':^15}|"
                                       f"{'GPU_Mem':^15}|"
                                       f"{'lr':^15}|"
                                       f"{s_loss_names}|"
                                       )

                    s_batch = f"{self.batch_size}"
                    s_epoch = f"{current_epoch + 1}/{self.epochs}"
                    s_memory = f"{self._get_memory():.3g}G"
                    s_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_description(
                        f"{'train':^5}|"
                        f"{s_batch:^15}|"
                        f"{s_epoch:^15}|"
                        f"{s_memory:^15}|"
                        f"{s_lr:^15.3g}|"
                        f"{s_loss_values}|"
                    )

                self.callbacks.run_callback(self, "on_train_batch_end")

            if RANK in {-1, 0}:
                # train result
                self.train_result.add("lr", self.optimizer.param_groups[0]["lr"])
                for loss_name, loss_value in smooth_loss_items.items():
                    self.train_result.add(loss_name, loss_value.item())

            # validation
            self.fitness = self.do_validate()  # 不可设置RANK in {-1, 0}，存在多卡验证情况

            # DDP同步fitness
            if world_size > 1:
                fitness_tensor = torch.tensor(self.fitness, dtype=torch.float32, device=self.device)
                dist.broadcast(fitness_tensor, src=0)
                self.fitness = fitness_tensor.item()

            if self.best_fitness < self.fitness:
                self.best_epoch = current_epoch + 1
                self.best_fitness = self.fitness

            if RANK in {-1, 0}:
                self.train_result.add("fitness", self.fitness)

            if LOCAL_RANK in {-1, 0}:
                # save model
                self.save_model(world_size, current_epoch)
                self.callbacks.run_callback(self, "on_model_save")

            # save train result to csv file
            if RANK in {-1, 0} and self.train_result:
                self.train_result.save_csv()

            self.callbacks.run_callback(self, "on_train_epoch_end")

            # stop training
            if RANK in {-1, 0}:
                if self.config_manager.core["time"] > 0:
                    current_train_time = (time.time() - train_time_start) / 60
                    if current_train_time >= self.config_manager.core["time"]:
                        LOGGER.warning(f"train cost time already is over {self.config_manager.core['time']} minutes, training stopped.")
                        self.stop = True
                if self.early_stopping(current_epoch, self.fitness):
                    LOGGER.warning(f"Early stopping training, training stopped.")
                    self.stop = True
                if current_epoch + 1 == self.epochs:
                    self.stop = True

            stop_tensor = torch.tensor(self.stop, device=self.device) if isinstance(self.stop, bool) else self.stop
            if RANK > -1 and world_size > 1:
                dist.broadcast(stop_tensor, src=0)
            self.stop = stop_tensor

            if self.stop:
                # DDP synchronize
                if world_size > 1:
                    dist.barrier()
                break

            # scheduler
            self.scheduler_step(current_epoch)

            # continue next epoch
            self._clear_memory()
            current_epoch += 1

            # DDP synchronize
            if world_size > 1:
                dist.barrier()

        self._clear_memory()

        if RANK in {-1, 0}:
            # train cost time
            train_time_end = time.time()
            train_time = train_time_end - train_time_start
            print("\n")
            LOGGER.info(f"Training finished. Total time: {train_time:.3f} seconds")

        # final eval
        if self.ema:
            LOGGER.info(f"Since EMA has been enabled, the saved model is EMA model, and its actual prediction results may differ from the true results.")
        LOGGER.info(f"current best epoch: {self.best_epoch}, Load best.pt model to final validate...")
        checkpoint = torch.load(self.best_pt, map_location=self.device, weights_only=False)
        if world_size > 1:
            self.model.module.load_model_state_dict(checkpoint["model"])
        else:
            self.model.load_model_state_dict(checkpoint["model"])
        self.do_validate()

        if RANK in {-1, 0}:
            # export simplified model
            self.simplified_model(world_size)

        self.callbacks.run_callback(self, "on_train_end")
        LOGGER.info(f"train results saved at {self.save_dir}")
