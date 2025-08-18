from __future__ import annotations

import os
import gc
import subprocess
import sys
import time
import warnings
import torch
import torch.distributed as dist

from typing import TYPE_CHECKING, List, Union
from datetime import timedelta
from pathlib import Path
from torch import autocast, optim, nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.global_var import RANK, NUM_THREADS, LOCAL_RANK, WORLD_SIZE
from tinytrain.metrics.base.train_result import TrainResult
from tinytrain.utils import LOGGER
from tinytrain.utils.TT_progress_bar import TTProgressBar
from tinytrain.utils.any_utils import set_random_seed, create_iter_directory, maybe_limit_num_workers
from tinytrain.utils.callback import Callback
from tinytrain.utils.checks import check_amp
from tinytrain.utils.dist import generate_ddp_command
from tinytrain.utils.train_utils import ModelEMA, EarlyStopping
from tinytrain.cfg.TT_register import TTEngineRegistry

if TYPE_CHECKING:
    from .model import BaseModel
    from . import BaseValidator


class BaseTrainer:
    """
    BaseTrainer 是一个通用、可扩展的深度学习训练框架基类，支持单机单卡、单机多卡（DDP）、多机多卡等多种训练模式。

    设计目标：
    - 解耦：将配置、模型、数据、优化器、验证器、回调等模块解耦，方便复用与扩展。
    - 通用性：适用于分类、检测、分割、生成等多种任务。
    - 高性能：支持 AMP（自动混合精度）、梯度累积、EMA、SyncBN、预热学习率等。
    - 分布式：原生支持 PyTorch DDP，自动处理通信、同步、保存等。
    - 可恢复：支持断点续训，自动保存/加载模型、优化器、epoch、fitness 等状态。
    - 可视化：集成 TensorBoard 日志、CSV 结果记录、训练过程图表绘制。

    核心功能：
    - 自动检测设备（CPU / CUDA / MPS）与配置合法性。
    - 自动构建 DataLoader，支持 DDP 采样、动态 batch_size、num_workers 优化。
    - 支持多种优化器（Adam、AdamW、SGD、Lion、AdaBelief 等）与学习率调度器（Linear、Cosine、Exponential、ReduceLROnPlateau 等）。
    - 支持模型层冻结、梯度裁剪、EarlyStopping、训练时长限制。
    - 支持 EMA（指数移动平均）模型保存与验证。
    - 支持模型导出为精简格式（如 fp16）用于部署。
    - 支持回调机制，可在训练各阶段插入自定义逻辑。

    使用方式：
    - 继承此类并重写以下方法：
        - `build_dataset()`：返回数据集实例。
        - `preprocess_data()`：对 batch 数据进行预处理。
        - `get_validator()`：返回验证器实例（可选）。
        - `do_validate()`：执行验证逻辑（可选）。
    - 调用 `trainer.train()` 即可启动训练。

    示例：
    ```python
    trainer = MyTrainer(config_manager, model, callback)
    trainer.train()
    ```

    注意事项：
    - 所有路径类参数建议使用 `pathlib.Path`。
    - 所有配置通过 `ConfigManager` 统一管理，支持 TOML 文件。
    - 所有日志通过 `LOGGER` 输出，支持 rank 过滤。
    - 训练结果统一保存在 `save_dir`，包括权重、日志、配置、图表等。
    """

    def __init__(self, config_manager: ConfigManager, model: BaseModel, callback: Callback, main_script_path: Path = None):
        """
        初始化 BaseTrainer 类，配置训练所需的核心组件。

        Args:
            config_manager (ConfigManager): 配置管理器，提供训练参数。
            model (BaseModel): 模型实例，用于训练和验证。
            callback (Callback): 回调函数，用于在训练过程中执行自定义操作。
            main_script_path (Path, optional): 主脚本路径，用于 DDP 启动。默认 None。
        """
        torch.set_float32_matmul_precision('high')
        self.config_manager: ConfigManager = config_manager
        self.validator: BaseValidator | None = None

        # resume
        self.resume: bool = self.config_manager.core["resume"]

        # device
        self.device = self.check_device()

        # amp
        self.amp: bool = self.config_manager.core["amp"]
        self.scaler: torch.amp.GradScaler | None = None

        # seed
        set_random_seed(self.config_manager.core["seed"], self.config_manager.core["deterministic"])

        # save dir
        self.save_dir: Path | None = None
        self.weight_dir: Path | None = None
        self.args_dir: Path | None = None
        self.last_pt: Path | None = None
        self.best_pt: Path | None = None
        self.simplified_pt: Path | None = None

        # dataset dir
        self.train_dir: Path | list[Path] | None = None
        self.val_dir: Path | list[Path] | None = None

        # dataloader
        self.train_dataloader = None
        self.val_dataloader = None

        # model
        self.model: BaseModel = model
        self.ema: ModelEMA | None = None

        # train
        self.start_epoch: int = 0
        self.epochs: int = self.config_manager.core["epochs"]
        self.warmup_epochs: int = self.config_manager.core["warmup_epochs"]
        self.batch_size: int = self.config_manager.core["batch_size"]
        self.stop: bool = False

        # optimizer and scheduler
        self.accumulate = self.config_manager.core["accumulate"]  # self.accumulate用于梯度累计策略，即每计算多少次loss再进行一次梯度更新
        self.optimizer = None
        self.warmup_scheduler = None
        self.scheduler = None
        self.early_stopping: EarlyStopping | None = None

        # loss and metrics
        self.loss: float = 0
        self.loss_items: list | None = None
        self.fitness: float = 0
        self.best_fitness: float = 0

        # train result
        self.best_epoch: int = 0
        self.train_result: TrainResult | None = None  # 记录训练打印信息，然后保存为csv文件

        # callback
        self.callbacks = callback

        # DDP
        self.main_script_path = main_script_path
        self.intra_node_group = None  # DDP分组

    # ------------------------------------------------------------------
    # 以下建议子类可重写的方法
    # ------------------------------------------------------------------
    def preprocess_data(self, batch_samples: BaseBatchDataInfo) -> BaseBatchDataInfo:
        """
        对批量数据进行预处理，如将数据从 CPU 移动到 GPU。

        Args:
            batch_samples (BaseBatchDataInfo): 原始批量数据。

        Returns:
            BaseBatchDataInfo: 预处理后的批量数据。

        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        raise NotImplementedError("preprocess_data function not implemented in trainer")

    def build_dataset(self, mode="train") -> Dataset:
        """
        构建数据集实例。

        Args:
            mode (str): 数据集模式，可选值为 "train"、"val"。

        Returns:
            Dataset: 数据集实例。

        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def plot_something_before_train(self):
        """
        在训练开始前绘制可视化图表（可选实现）。
        """
        pass

    def freeze_layers(self, world_size: int):
        """
        冻结模型的某些层，防止其参数在训练中被更新。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        freeze_layer_names = []
        for k, v in self.model.module.named_parameters() if world_size > 1 else self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False

    # ------------------------------------------------------------------
    # 以下不建议子类重写的方法
    # ------------------------------------------------------------------
    def train(self):
        """
        启动模型训练，支持单机单卡、单机多卡、多机多卡等多种训练方式。
        """
        nproc_per_node = self._get_nproc_per_node()

        # 判断是否已由 torchrun 启动（DDP 子进程）
        if "LOCAL_RANK" in os.environ:
            LOGGER.info("Detected DDP environment (torchrun), skipping subprocess launch.")
            self._perform_training(WORLD_SIZE)
        elif nproc_per_node > 1:
            self._launch_ddp_training(nproc_per_node)
        else:
            self._perform_training(1)

    def _get_nproc_per_node(self):
        """
        获取当前节点的 GPU 数量（CPU 或 MPS 返回 0）。

        Returns:
            int: 当前节点可用的 GPU 数量。
        """
        if self.device.type in {"cpu", "mps"}:
            return 0
        elif self.device.type == "cuda":
            device_config = self.config_manager.core["device"]
            if isinstance(device_config, int):
                return 1
            elif isinstance(device_config, list):
                return len(device_config)
        else:
            raise NotImplementedError(f"device type {self.device.type} is not supported.")
        return 0

    @staticmethod
    def _get_intra_node_group():
        """
        创建并返回当前节点内的 DDP 通信组。

        Returns:
            dist.ProcessGroup | None: 当前节点内的通信组，未初始化返回 None。
        """
        if not dist.is_initialized():
            return None

        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        rank = dist.get_rank()
        node_rank = rank // local_world_size

        # 当前节点内的所有全局 rank
        ranks_in_node: List[int] = list(
            range(node_rank * local_world_size, (node_rank + 1) * local_world_size)
        )

        # 创建并返回节点内通信组
        group = dist.new_group(ranks=ranks_in_node)
        return group

    def _launch_ddp_training(self, nproc_per_node):
        """
        使用 torchrun 启动分布式训练。

        Args:
            nproc_per_node (int): 每个节点的 GPU 数量。
        """
        cmd = generate_ddp_command(self, nproc_per_node)
        try:
            LOGGER.info("Starting DDP training...")
            subprocess.run(cmd, check=True)
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            raise
        finally:
            LOGGER.info("Releasing memory...")
            self._clear_memory()

    def _perform_training(self, world_size):
        """
        执行实际训练逻辑，包括 DDP 初始化、训练、异常处理和资源清理。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        try:
            if world_size > 1:
                LOGGER.info("Initializing DDP...")
                self.set_ddp(world_size)
            if self.intra_node_group is None and world_size > 1 and dist.is_initialized():
                self.intra_node_group = self._get_intra_node_group()
            self._do_train(world_size)
        except KeyboardInterrupt:
            LOGGER.warning(f"Training interrupted by user (Rank {RANK}).")
        except SystemExit as e:
            LOGGER.warning(f"SystemExit received (Rank {RANK}), code: {e.code}")
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            raise e
        finally:
            self._graceful_shutdown(world_size)
            LOGGER.info("Releasing memory...")
            self._clear_memory()

    def _graceful_shutdown(self, world_size):
        """
        训练结束或中断时进行优雅退出，清理资源并保存结果。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        try:
            # 1.摧毁DDP
            if world_size > 1 and dist.is_initialized():
                LOGGER.info("Destroying DDP...")
                self.destroy_ddp()

            # 2.触发 DataLoaderIter.__del__()，从而安全地关闭 worker 和线程
            self.train_dataloader = None
            self.val_dataloader = None

            # 3.绘制train result
            if RANK in {-1, 0} and self.train_result:
                self.train_result.plot(start=self.start_epoch + 1)
                self.train_result.close()  # 可以选择不close，那么tensorboard训练完不会关闭

        except Exception as e:
            LOGGER.error(f"Error during graceful shutdown: {e}")

    def _before_train(self, world_size: int):
        """
        训练前的准备工作，包括路径设置、模型初始化、优化器配置等。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        self.callbacks.run_callback(self, "on_prepare_train_start")

        # set save dir and dataset dir and train result
        self.set_save_dir(world_size)
        self.set_dataset_dir()
        if RANK in {-1, 0}:
            self.train_result = TrainResult(save_dir=self.save_dir, launch_tb=self.config_manager.core["launch_tb"])

        # check batch size
        self.check_batch_size(world_size)

        # set dataloaders
        self.set_dataloaders(world_size)

        # set model
        self.set_model(world_size)

        # freeze layers
        self.freeze_layers(world_size)

        # check AMP
        self.check_amp(world_size)

        # set optimizer
        self.set_optimizer(self.model)

        # resume training
        self.resume_training()

        # set scheduler
        self.set_warmup_scheduler()
        self.set_normal_scheduler()

        # EarlyStopping
        self.early_stopping = EarlyStopping(self.config_manager.core["patience"])

        # save config file
        if RANK in {-1, 0}:
            self.save_config_file()

        # plot
        if RANK in {-1, 0}:
            self.plot_something_before_train()

        self.callbacks.run_callback(self, "on_prepare_train_end")

        # DDP synchronize
        if world_size > 1:
            dist.barrier()

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
                    self.loss, self.loss_items = self.model(batch_samples)

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

    def set_ddp(self, world_size: int):
        """
        初始化分布式训练环境（DDP）。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        from torch.distributed import init_process_group

        torch.cuda.set_device(LOCAL_RANK)
        self.device = torch.device("cuda", LOCAL_RANK)

        init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
            device_id=self.device,
        )

    @staticmethod
    def destroy_ddp():
        """
        销毁分布式训练环境（DDP）。
        """
        from torch.distributed import destroy_process_group
        destroy_process_group()

    def set_save_dir(self, world_size: int):
        """
        设置训练结果保存路径，支持多节点同步。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        # 节点内主进程（LOCAL_RANK==0）负责创建目录
        if LOCAL_RANK in {-1, 0}:
            config_core = self.config_manager.core
            save_dir = Path(config_core["save_dir"]).resolve()
            project_name = config_core["project_name"] or "default_project"
            config_core["project_name"] = project_name
            save_dir = save_dir / project_name / config_core["task"]
            self.save_dir = create_iter_directory(save_dir)

        # 节点内广播 save_dir
        if world_size > 1 and dist.is_initialized():
            save_dir_list = [str(self.save_dir)] if LOCAL_RANK == 0 else [None]
            dist.broadcast_object_list(save_dir_list, src=0, group=self.intra_node_group)
            self.save_dir = Path(save_dir_list[0])

        # 所有进程统一构建子目录
        self.args_dir = self.save_dir / "args"
        self.weight_dir = self.save_dir / "weights"
        self.args_dir.mkdir(parents=True, exist_ok=True)
        self.weight_dir.mkdir(parents=True, exist_ok=True)

        self.last_pt = self.weight_dir / "last.pt"
        self.best_pt = self.weight_dir / "best.pt"
        self.simplified_pt = self.weight_dir / "simplified_best.pt"

    def check_device(self) -> torch.device:
        """
        根据配置检查并返回可用设备（CPU、CUDA 或 MPS）。

        Returns:
            torch.device: 可用设备。
        """

        device = self.config_manager.core["device"]

        # 检查设备可用性
        if device is None or device == "cpu":
            current_device = "cpu"
        elif device == "mps" and torch.mps.is_available():
            current_device = "mps"
        elif device == "cuda" or isinstance(device, int) or isinstance(device, list):
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                current_device = "cuda"
            else:
                current_device = "cpu"
        else:
            current_device = "cpu"

        # 根据设备类型设置环境变量和配置
        if current_device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            torch.set_num_threads(NUM_THREADS)
            self.config_manager.core["workers"] = 0
            self.config_manager.core["device"] = "cpu"
            if "LOCAL_RANK" not in os.environ:
                LOGGER.info("Device Type: CPU")
            return torch.device("cpu")
        elif current_device == "mps":
            self.config_manager.core["device"] = "mps"
            if "LOCAL_RANK" not in os.environ:
                LOGGER.info("Device Type: MPS")
            return torch.device("mps")
        elif current_device == "cuda":
            cuda_num = torch.cuda.device_count()
            if device == "cuda":
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                self.config_manager.core["device"] = 0
                if "LOCAL_RANK" not in os.environ:
                    LOGGER.info("Device Type: CUDA, using cuda:0.")
            elif isinstance(device, int):
                if device < cuda_num:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                    self.config_manager.core["device"] = device
                    if "LOCAL_RANK" not in os.environ:
                        LOGGER.info(f"Device Type: CUDA, using device index {device}.")
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    self.config_manager.core["device"] = 0
                    if "LOCAL_RANK" not in os.environ:
                        LOGGER.info(f"Device Type: CUDA, CUDA Device {device} not found, using cuda:0.")
            elif isinstance(device, list):
                true_device = [d for d in device if d < cuda_num]
                false_device = [d for d in device if d >= cuda_num]
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, true_device))
                self.config_manager.core["device"] = true_device
                if len(false_device) > 0:
                    if "LOCAL_RANK" not in os.environ:
                        LOGGER.info(f"Device Type: CUDA, Device {','.join(map(str, false_device))} not found, only using {','.join(map(str, true_device))}")
                else:
                    if "LOCAL_RANK" not in os.environ:
                        LOGGER.info(f"Device Type: CUDA, using {','.join(map(str, true_device))}")
            else:
                raise TypeError(f"device type: {type(device)} is not supported!")

            return torch.device("cuda:0")

    def set_dataset_dir(self) -> None:
        """
        设置训练、验证和测试数据集路径。
        统一返回 list，即使只有一个数据集。
        """

        def _get_dirs(root_dirs: Union[str, List[str]], split_key: str) -> List[Path]:
            """辅助函数：根据 split_key 获取所有存在的路径列表"""
            root_dirs = [root_dirs] if isinstance(root_dirs, (str, Path)) else root_dirs
            dirs = [
                (Path(root) / self.config_manager.dataset[split_key]).resolve()
                for root in root_dirs
            ]
            for d in dirs:
                if not d.exists():
                    raise FileNotFoundError(f"Dataset root: {d} not found.")
            return dirs

        dataset_root_dirs = self.config_manager.dataset["path"]

        self.train_dir = _get_dirs(dataset_root_dirs, "train")
        self.val_dir = _get_dirs(dataset_root_dirs, "val")

    def check_amp(self, world_size: int):
        """
        检查是否支持自动混合精度（AMP），并初始化 GradScaler。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        # DDP 支持 AMP，直接启动，不进行检查
        if world_size > 1:
            if RANK in {-1, 0} and self.amp:
                LOGGER.info("Enabling AMP (Automatic Mixed Precision) for DDP (Distributed Data Parallel) training.")
        else:
            # 检查 AMP 支持
            if RANK in {-1, 0}:
                if self.amp:
                    LOGGER.info("AMP: running Automatic Mixed Precision (AMP) checks...")
                    # 支持手动关闭 AMP 检查，如需关闭 AMP 检查，请注释下面这一行代码
                    self.amp = check_amp(self, self.get_model_instance(world_size), self.train_dataloader.dataset)
                    if self.amp:
                        LOGGER.info("AMP: checks passed ✅")
                    else:
                        LOGGER.warning("AMP: checks failed ❌")

        # 根据设备类型禁用 AMP
        if self.device.type in {"cpu", "mps"}:
            self.amp = False
            if self.device.type == "cpu":
                LOGGER.warning("CPU does not support AMP. Disabling AMP.")
            elif self.device.type == "mps":
                LOGGER.warning("Apple MPS does not support AMP. Disabling AMP.")

        # 同步 AMP 状态到所有进程（DDP 情况）
        if world_size > 1:
            amp_tensor = torch.tensor(self.amp, device=self.device)
            if RANK > -1:
                dist.broadcast(amp_tensor, src=0)
            self.amp = bool(amp_tensor)

        # 初始化 GradScaler
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    def check_batch_size(self, world_size: int):
        """
        检查 batch_size 是否合理，是否支持 DDP。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        # 检查批量大小是否为 1
        if self.batch_size == 1:
            raise ValueError(f"Batch size {self.batch_size} cannot be 1.")

        # 检查批量大小是否为 16 的倍数
        if self.batch_size % 16 != 0:
            LOGGER.warning(
                "Batch size is not a multiple of 16. It is recommended to set batch size as a multiple of 16 for better training performance.")

        # 打印批量大小信息
        if world_size > 1:
            LOGGER.info(f"{world_size} GPU(s) found. Each GPU has a batch size of {self.batch_size}.")
        else:
            LOGGER.info(f"Training batch size: {self.batch_size}")

    def set_dataloaders(self, world_size: int):
        """
        构建训练与验证的 DataLoader。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        # train dataloader
        self.train_dataloader = self.build_dataloader(world_size, mode="train")

        # validation dataloader
        self.val_dataloader = self.build_dataloader(world_size, mode="val")
        self.validator = self.get_validator(world_size)

    def set_optimizer(self, model):
        """
        构建优化器，支持参数分组、学习率缩放、多种优化器选择。

        Args:
            model (nn.Module): 模型实例。
        """

        optimizer_name = self.config_manager.core["optimizer"]
        lr0 = self.config_manager.core["lr0"]
        momentum = self.config_manager.core["momentum"]
        weight_decay = self.config_manager.core["weight_decay"]

        # ---------------- 线性缩放 LR ----------------
        if WORLD_SIZE > 1:
            lr_scaled = self.config_manager.core["lr0"] * max(WORLD_SIZE, 1) * self.accumulate
        else:
            lr_scaled = lr0

        # ---------------- 参数分组 ----------------
        # 1. 收集所有可训练参数，并按 name 排序，确保跨进程顺序一致
        named_params = sorted(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            key=lambda x: x[0]
        )

        # 2. 定义 norm 类
        norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        groups = {
            "no_decay_bias": [],
            "no_decay_norm": [],
            "with_decay": []
        }

        # 3. 遍历排序后的参数，根据所属模块判断分组
        for name, param in named_params:
            # 找到该参数所属的模块
            parent_module = None
            for m_name, m in model.named_modules():
                # 判断 param 是否属于该模块的直属参数
                if any(id(p) == id(param) for p in m.parameters(recurse=False)):
                    parent_module = m
                    break

            if "bias" in name:
                groups["no_decay_bias"].append(param)
            elif parent_module is not None and isinstance(parent_module, norm_types):
                groups["no_decay_norm"].append(param)
            else:
                groups["with_decay"].append(param)

        # ---------------- 构造优化器 ----------------
        common_kwargs = {"lr": lr_scaled}

        # pytorch优化器
        registry = {
            "Adam": (optim.Adam, {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "AdamW": (optim.AdamW, {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "Adamax": (optim.Adamax, {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "NAdam": (optim.NAdam, {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "RAdam": (optim.RAdam, {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "RMSprop": (optim.RMSprop, {"momentum": momentum}),
            "SGD": (optim.SGD, {"momentum": momentum, "nesterov": True}),
            "Adadelta": (optim.Adadelta, {"weight_decay": 0.0}),
            "Adagrad": (optim.Adagrad, {"weight_decay": 0.0}),
        }

        # 第三方优化器
        # key   = 配置文件里优化器的名称
        # value = (模块导入路径, 类名, 默认kwargs)
        _third_party = {
            "Lion": ("lion_pytorch", "Lion", {"weight_decay": 0.0}),
            "AdaBelief": ("adabelief_pytorch", "AdaBelief", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "AdaBound": ("torch_optimizer", "AdaBound", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "AdaMod": ("torch_optimizer", "AdaMod", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "Lamb": ("torch_optimizer", "Lamb", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "Ranger": ("torch_optimizer", "Ranger", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
            "Ranger21": ("torch_optimizer", "Ranger21", {"betas": (momentum, 0.999), "weight_decay": 0.0}),
        }

        if optimizer_name in _third_party:
            mod_path, cls_name, kwargs = _third_party[optimizer_name]
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                opt_cls = getattr(mod, cls_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Optimizer '{optimizer_name}' needs `pip install {mod_path}`"
                ) from e
            registry[optimizer_name] = (opt_cls, kwargs)

        if optimizer_name not in registry:
            LOGGER.warning(f"Optimizer '{optimizer_name}' is not supported, fallback to AdamW.")
            optimizer_name = "AdamW"

        opt_cls, opt_kwargs = registry[optimizer_name]
        opt_kwargs.update(common_kwargs)

        # ---------- 4. 构建优化器 ----------
        # 必须至少一个 param_group
        optimizer = opt_cls(groups["no_decay_bias"], **opt_kwargs)

        if groups["with_decay"]:
            optimizer.add_param_group({"params": groups["with_decay"],
                                       "weight_decay": weight_decay})
        if groups["no_decay_norm"]:
            optimizer.add_param_group({"params": groups["no_decay_norm"],
                                       "weight_decay": 0.0})

        self.optimizer = optimizer

        # 提前初始化 Adagrad 的状态字典（修复 KeyError: 'sum'）
        if isinstance(self.optimizer, torch.optim.Adagrad):
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                        state["sum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def set_warmup_scheduler(self):
        """
        设置预热阶段的学习率调度器。
        """

        if self.warmup_epochs <= 0:
            return
        if self.start_epoch >= self.warmup_epochs:
            return

        warmup_scheduler_name = self.config_manager.core["warmup_scheduler"]
        lr0 = self.config_manager.core["lr0"]
        warmup_lr = self.config_manager.core["warmup_lr"]
        warmup_epochs = self.warmup_epochs
        start_epoch = self.start_epoch

        # 预热开始 epoch 对应的 last_epoch
        last_epoch = max(start_epoch - 1, -1)

        # set warmup_scheduler
        if warmup_scheduler_name == "LinearWarmupLR":
            from tinytrain.utils.scheduler import LinearWarmupLR
            self.warmup_scheduler = LinearWarmupLR(
                self.optimizer,
                warmup_lr=warmup_lr,
                lr0=lr0,
                warmup_epochs=self.warmup_epochs,
                last_epoch=last_epoch
            )
        elif warmup_scheduler_name == "CosineWarmUpLR":
            from tinytrain.utils.scheduler import CosineWarmUpLR
            self.warmup_scheduler = CosineWarmUpLR(
                self.optimizer,
                warmup_lr=warmup_lr,
                lr0=lr0,
                epochs=warmup_epochs,
                last_epoch=last_epoch
            )
        elif warmup_scheduler_name == "ExponentialWarmUpLR":
            from tinytrain.utils.scheduler import ExponentialWarmUpLR
            self.warmup_scheduler = ExponentialWarmUpLR(
                self.optimizer,
                warmup_lr=warmup_lr,
                lr0=lr0,
                epochs=warmup_epochs,
                last_epoch=last_epoch
            )
        elif warmup_scheduler_name == "ConstantWarmupLR":
            from tinytrain.utils.scheduler import ConstantWarmupLR
            # 常量预热学习率衰减器，不做任何学习率衰减
            self.warmup_scheduler = ConstantWarmupLR(
                self.optimizer,
                warmup_lr=warmup_lr,
                lr0=lr0,
                warmup_epochs=warmup_epochs
            )
        else:
            LOGGER.warning(f"Unknown warmup_scheduler '{warmup_scheduler_name}', fallback to LinearWarmupLR.")
            from tinytrain.utils.scheduler import LinearWarmupLR
            self.warmup_scheduler = LinearWarmupLR(
                self.optimizer,
                warmup_lr=warmup_lr,
                lr0=lr0,
                warmup_epochs=self.warmup_epochs,
                last_epoch=last_epoch
            )

    def set_normal_scheduler(self):
        """
        设置正式训练阶段的学习率调度器。
        """

        if self.epochs <= self.warmup_epochs:
            return

        scheduler_name = self.config_manager.core["scheduler"]
        lr0 = self.config_manager.core["lr0"]
        lr1 = self.config_manager.core["lr1"]
        start_epoch = self.start_epoch
        warmup_epochs = self.warmup_epochs

        # 计算正式阶段开始的 last_epoch
        if start_epoch < warmup_epochs:
            last_epoch = -1
            normal_epochs = self.epochs - warmup_epochs
        else:
            last_epoch = start_epoch - warmup_epochs - 1
            normal_epochs = self.epochs - start_epoch

        # ---------- 衰减器选择 ----------
        if scheduler_name == "LinearLR":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr1,
                total_iters=normal_epochs,
                last_epoch=last_epoch
            )
        elif scheduler_name == "CosineLR":
            t_max = normal_epochs if normal_epochs < 100 else normal_epochs // 10
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=lr0 * lr1,
                last_epoch=last_epoch
            )
        elif scheduler_name == "ExponentialLR":
            gamma = lr1 ** (1.0 / normal_epochs)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma,
                last_epoch=last_epoch
            )
        elif scheduler_name == "StepLR":
            step_size = max(1, normal_epochs // 3)
            gamma = lr1 ** (1.0 / (normal_epochs // step_size))
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == "MultiStepLR":
            milestones = [int(normal_epochs * 0.25), int(normal_epochs * 0.5), int(normal_epochs * 0.75)]
            gamma = lr1 ** (1.0 / len(milestones))
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
        elif scheduler_name == "auto":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=5,
                factor=0.5,
                threshold=0.01,
                threshold_mode="rel",
                cooldown=0,
                min_lr=lr0 * lr1,
            )
        else:
            LOGGER.warning(f"Unknown normal scheduler '{scheduler_name}', fallback to LinearLR.")
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr1,
                total_iters=normal_epochs,
                last_epoch=last_epoch
            )

    def set_model(self, world_size: int):
        """
        设置模型，包括移动到设备、启用 SyncBatchNorm、封装 DDP 和 EMA。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        self.model = self.model.to(self.device)

        # 所有参数默认开启梯度
        for param in self.model.parameters():
            param.requires_grad = True

        # 多卡情况下：先转 SyncBN，再封装 DDP
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            # 关键：把普通 BN → SyncBatchNorm
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = DDP(self.model, device_ids=[LOCAL_RANK], gradient_as_bucket_view=True)

        # EMA（仅主进程）
        if self.config_manager.core["ema"] and LOCAL_RANK in {-1, 0}:
            self.ema = ModelEMA(self.model, world_size)
            LOGGER.info("EMA(Exponential Moving Average) is enabled.")

    def build_dataloader(self, world_size: int, mode: str = "train"):
        """
        构建 DataLoader，支持 DDP、动态 batch_size、num_workers 调整。

        Args:
            world_size (int): 分布式训练中的进程数量。
            mode (str): 模式，"train"、"val"。

        Returns:
            DataLoader: 构建好的数据加载器。
        """

        # 构建数据集
        dataset = self.build_dataset(mode=mode)

        # 判断是否shuffle
        if mode == "train":
            shuffle = True
        else:
            shuffle = self.config_manager.core["shuffle_val_dataloader"]

        # 计算每个 rank 实际分到的样本数
        if world_size > 1 and mode == "train":
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, drop_last=True)  # drop_last为True保证每张卡样本数量一致，避免all_gather永久阻塞
            effective_samples = len(sampler)
        else:
            sampler = None
            effective_samples = len(dataset)  # type: ignore[arg-type]

        # 确保 batch_size 不超过实际样本数
        batch_size = min(self.batch_size, effective_samples)
        if batch_size < 2:
            raise ValueError(
                f"Dataset too small for DDP: rank {RANK} has {effective_samples} samples, "
                f"but batch_size is {batch_size}. Reduce world_size or increase dataset."
            )

        # 计算 num_workers
        num_devices = torch.cuda.device_count()
        num_workers = self.config_manager.core["workers"] = min(8, os.cpu_count() // max(num_devices, 1), self.config_manager.core["workers"])

        # linux环境根据共享内存决定最终 num_workers
        num_workers = maybe_limit_num_workers(num_workers, safe_threshold_mb=2048)

        # 生成器（用于可复现性）
        generator = torch.Generator()
        generator.manual_seed(self.config_manager.core["seed"] + RANK)

        # 调整 val/test 的 batch_size和 num_workers
        if mode != "train":
            batch_size = max(2, batch_size // 2)
            num_workers = max(0, num_workers // 2)

        # DDP均摊 num_workers
        if world_size > 1:
            num_workers = max(0, num_workers // world_size)

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # 由 sampler 控制 shuffle
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn", None),
            generator=generator,
            persistent_workers=num_workers > 0,
            drop_last=False,
            pin_memory=True,
            pin_memory_device=str(self.device),
            prefetch_factor=min(4, max(2, os.cpu_count() // max(world_size, 1))) if num_workers > 0 else None,
            multiprocessing_context='spawn' if os.name != 'nt' and num_workers > 0 else None,
        )
        return dataloader

    def resume_training(self):
        """
        从 checkpoint 恢复训练，包括 epoch、optimizer、fitness 等。
        """

        if self.resume and self.config_manager.link["model"].suffix not in {".pt", ".pth"}:
            raise ValueError(f"To resume training, the model file must have a '.pt' or '.pth' suffix.")

        if self.resume and self.config_manager.link["model"].suffix in {".pt", ".pth"}:
            checkpoint = torch.load(self.config_manager.link["model"], map_location="cpu", weights_only=False)
            self.start_epoch = checkpoint["current_epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.fitness = checkpoint["fitness"]

            # 广播 start_epoch 和 fitness 给所有进程
            if WORLD_SIZE > 1:
                sync_tensor = torch.tensor([self.start_epoch, self.fitness], dtype=torch.float32, device=self.device)
                dist.broadcast(sync_tensor, src=0)
                self.start_epoch = int(sync_tensor[0].item())
                self.fitness = sync_tensor[1].item()

            LOGGER.info(f"Resume training from last epoch {self.start_epoch}...")
            if self.start_epoch >= self.epochs:
                LOGGER.info(f"Training has been completed: {self.start_epoch}/{self.epochs}")
                sys.exit()

    def optimizer_step(self):
        """
        安全版 optimizer_step：
        1. 检查梯度是否出现 NaN/Inf，出现则跳过更新并降低 loss-scale；
        2. 正常时才进行梯度裁剪、optimizer.step、ema.update；
        3. 输出日志以便调试。
        """
        # 1. 先把梯度 unscale 回 fp32
        self.scaler.unscale_(self.optimizer)

        # 2. 检查全局梯度是否有 NaN/Inf
        # has_nan_or_inf = False
        # for group in self.optimizer.param_groups:
        #     for p in group["params"]:
        #         if p.grad is not None:
        #             if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
        #                 has_nan_or_inf = True
        #                 break
        #     if has_nan_or_inf:
        #         break
        #
        # # 在多卡环境下，把结果广播到所有 rank
        # if dist.is_initialized():
        #     has_nan = torch.tensor(float(has_nan_or_inf), device="cuda")
        #     dist.all_reduce(has_nan, op=dist.ReduceOp.MAX)
        #     has_nan_or_inf = bool(has_nan.item())
        #
        # if has_nan_or_inf:
        #     # 跳过本次更新，降低 loss-scale
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     if rank == 0:
        #         print(f"[WARN] NaN/Inf detected in gradients, skipping step. "
        #               f"Current scale: {self.scaler.get_scale()}")
        #     self.scaler.update()  # 这会触发 scale 下降
        #     self.optimizer.zero_grad(set_to_none=True)
        #     return

        # 3. 梯度裁剪
        grad_clip = self.config_manager.core.get("grad_clip", 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

        # 4. 安全 step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # 5. EMA 更新
        if LOCAL_RANK in {-1, 0} and self.ema is not None:
            self.ema.update(self.model)

    def scheduler_step(self, current_epoch: int):
        """
        根据当前 epoch 更新学习率。

        Args:
            current_epoch (int): 当前训练轮次。
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            if current_epoch + 1 <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                scheduler_name = self.config_manager.core["scheduler"]
                if scheduler_name == "auto":
                    self.scheduler.step(self.fitness)
                else:
                    self.scheduler.step()

    def save_config_file(self):
        """
        将配置保存为 TOML 文件到指定目录。
        """
        import toml

        # 简化数据结构，确保值符合 TOML 格式
        def simplify_value(v):
            if isinstance(v, dict):
                return {str(k): simplify_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [simplify_value(item) for item in v]
            else:
                return v

        config_dicts = self.config_manager.__dict__.copy()
        for key, value in config_dicts.items():
            try:
                if isinstance(value, dict):
                    value = simplify_value(value)

                    # 将字典保存到 TOML 文件
                    with open(self.args_dir / f"{key}.toml", "w", encoding="utf-8") as f:
                        toml.dump(value, f)  # type: ignore[arg-type]
            except Exception as e:
                raise e

        LOGGER.info(f"Save all config files in directory: {self.args_dir}")

    def get_validator(self, world_size: int):
        """
        获取验证器实例。

        Args:
            world_size (int): 分布式训练中的进程数量。

        Returns:
            BaseValidator: 验证器实例。
        """

        try:
            return TTEngineRegistry.get(self.config_manager, "validator")(self, world_size)
        except Exception as e:
            print(f"no validator found for world size {world_size}")
            raise e

            # return None

    def do_validate(self) -> float:
        """
        执行模型验证并返回 fitness 值。

        Returns:
            float: fitness 值。
        """
        return self.validator.validate() if self.validator else 0.

    def get_model_instance(self, world_size: int) -> nn.Module:
        """
        获取用于保存的模型实例（EMA 或原始模型）。

        Args:
            world_size (int): 分布式训练中的进程数量。

        Returns:
            nn.Module: 模型实例。
        """
        if self.ema:
            model = self.ema.ema_model
        else:
            model = self.model if world_size <= 1 else self.model.module
        return model

    def save_model(self, world_size: int, current_epoch: int):
        """
        保存模型 checkpoint，包括 last.pt、best.pt、epoch_X.pt。

        Args:
            world_size (int): 分布式训练中的进程数量。
            current_epoch (int): 当前训练轮次。
        """
        try:
            # 获取要保存的模型
            model = self.get_model_instance(world_size)
            model.eval()

            # 构建检查点
            checkpoint = {
                "current_epoch": current_epoch + 1,
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "fitness": self.fitness,
                "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
                "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()}
            }

            if LOCAL_RANK in {-1, 0}:
                # 保存最新的模型
                torch.save(checkpoint, self.last_pt.as_posix())

                # 保存最佳模型
                if self.best_fitness == self.fitness:
                    torch.save(checkpoint, self.best_pt.as_posix())

                # 按周期保存模型（仅主进程）
                save_period = self.config_manager.core["save_period"]
                if save_period > 0 and (current_epoch + 1) % save_period == 0:
                    epoch_checkpoint_path = Path(self.weight_dir / f"epoch_{current_epoch + 1}.pt")
                    torch.save(checkpoint, epoch_checkpoint_path.as_posix())

        except Exception as e:
            LOGGER.error(f"Error occurred while saving model: {e}")
            raise e

    def simplified_model(self, world_size: int):
        """
        导出精简模型pt文件（如 fp16）用于部署。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """

        LOGGER.info(f"start export simplified model...")

        # 获取要保存的模型
        model = self.get_model_instance(world_size)

        # 将模型设置为评估模式
        model.eval()

        fp16_pt = self.config_manager.export["FP16_pt"]
        if fp16_pt:
            model = model.half()
            LOGGER.info("Simplified model converted to float16 (fp16) format.")

        checkpoint = {
            'model': model.state_dict(),
            "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
            "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()},
            "fp16": fp16_pt
        }

        torch.save(checkpoint, self.simplified_pt.as_posix())

    def _get_memory(self) -> float:
        """
        获取当前设备显存/内存使用量。

        Returns:
            float: 使用量（单位：GB）。
        """

        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _clear_memory(self):
        """
        清理设备内存/显存。
        """
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def only_do_validate(self):
        """
        仅执行验证流程，不训练。
        """

        self.stop = True  # 传递stop信号
        self.do_validate()
