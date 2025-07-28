from __future__ import annotations

import os
import gc
import subprocess
import sys
import time
import warnings
import torch
import torch.distributed as dist

from typing import TYPE_CHECKING, List
from datetime import timedelta
from pathlib import Path
from torch import autocast, optim, nn
from torch.utils.data.dataloader import DataLoader

from TinyTrain.cfg.config_manager import ConfigManager
from TinyTrain.data import TTBaseDataset
from TinyTrain.data.data_format import BaseBatchDataInfo
from TinyTrain.global_var import RANK, NUM_THREADS, LOCAL_RANK, WORLD_SIZE
from TinyTrain.metrics.train_result import TrainResult
from TinyTrain.utils import LOGGER
from TinyTrain.utils.TT_progress_bar import TTProgressBar
from TinyTrain.utils.any_utils import set_random_seed, create_iter_directory
from TinyTrain.utils.callback import Callback
from TinyTrain.utils.checks import check_amp
from TinyTrain.utils.dist import generate_ddp_command
from TinyTrain.utils.train_utils import ModelEMA, EarlyStopping
from TinyTrain.utils.register import TTRegistry

if TYPE_CHECKING:
    from .model import BaseModel
    from . import BaseValidator


class BaseTrainer:
    def __init__(self, config_manager: ConfigManager, model: BaseModel, callback: Callback, main_script_path: Path = None):
        """
        初始化 BaseTrainer 类。

        @param config_manager: 配置管理器。
        @param model: 模型实例，用于训练和验证。
        @param callback: 回调函数，用于在训练过程中执行自定义操作。
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
        self.test_dir: Path | list[Path] | None = None

        # dataloader
        self.train_dataloader = None
        self.val_dataloader = None
        self.amp_dataloader = None

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

    def preprocess_data(self, batch_samples: BaseBatchDataInfo) -> BaseBatchDataInfo:
        """
        对批量数据进行预处理。一般包括将数据从CPU传递到GPU等操作。

        @param batch_samples: 批量数据信息。
        @return: 预处理后的批量数据信息。
        """
        raise NotImplementedError("preprocess_data function not implemented in trainer")

    def build_dataset(self, mode="train") -> TTBaseDataset:
        """
        构建数据集，需要子类进行重载以构建不同的数据集。

        @param mode: 数据集模式，可选值为 "train"、"val"和"test"。默认为 "train"。
        @return: TTBaseDataset子类实例。
        """
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def plot_something_before_train(self):
        """
        在训练前绘制一些图表（可选实现）。
        """
        pass

    def freeze_layers(self, world_size: int):
        """
        冻结模型的某些层。子类要重载实现自定义冻结需求, 可参照下方代码复制修改使用。

        @param world_size: 分布式训练中的进程数量。
        """

        freeze_layer_names = []
        for k, v in self.model.module.named_parameters() if world_size > 1 else self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False

    def train(self):
        """
        启动模型训练。支持单机单卡、单机多卡、多机多卡等训练方式。
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
        获取每个节点的 GPU 数。CPU或MPS情况下默认为0.
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
        返回一个仅包含当前节点（机器）内所有进程的通信组。
        组内 rank=0 对应 LOCAL_RANK=0 的进程。
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
        启动分布式数据并行（DDP）训练。
        """
        cmd = generate_ddp_command(self, nproc_per_node)
        try:
            LOGGER.info("Starting DDP training...")
            subprocess.run(cmd, check=True)
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            raise e
        finally:
            LOGGER.info("Releasing memory...")
            self._clear_memory()

    def _perform_training(self, world_size):
        """
        执行实际的训练逻辑。
        """
        try:
            if world_size > 1:
                LOGGER.info("Initializing DDP...")
                self.set_ddp(world_size)
            if self.intra_node_group is None and world_size > 1 and dist.is_initialized():
                self.intra_node_group = self._get_intra_node_group()
            self._do_train(world_size)
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            raise e
        except KeyboardInterrupt:
            LOGGER.warning(f"Training interrupted by user (Rank {RANK}).")
            self._graceful_shutdown(world_size)
        except SystemExit as e:
            LOGGER.warning(f"SystemExit received (Rank {RANK}), code: {e.code}")
            self._graceful_shutdown(world_size)
        finally:
            if world_size > 1:
                LOGGER.info("Destroying DDP...")
                self.destroy_ddp()
            LOGGER.info("Releasing memory...")
            self._clear_memory()

    def _graceful_shutdown(self, world_size):
        """
        安全中断退出
        """
        if world_size > 1:
            self.destroy_ddp()
        self._clear_memory()
        sys.exit(0)

    def _before_train(self, world_size: int):
        """
        训练模型前，需要做的各种检查与准备。请勿打乱以下执行顺序！

        @param world_size: 分布式训练中的进程数量。
        """
        self.callbacks.run_callback(self, "on_prepare_train_start")

        # set save dir and dataset dir and train result
        self.set_save_dir(world_size)
        self.set_dataset_dir()
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
        执行模型训练。
        非开发人员不要手动直接调用函数。如果需要训练模型，请使用train函数。

        @param world_size: 分布式训练中的进程数量。
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
            for i, batch_samples in enumerate(pbar):  # 45.382
                self.callbacks.run_callback(self, "on_train_batch_start")

                # 当前是第几个batch
                current_batch = current_epoch * num_batch + i

                # 判断本次是否真正执行 optimizer.step()
                is_last_accum_step = (current_batch - last_opt_step) == self.accumulate

                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and self.config_manager.core["bf16"] else torch.float16
                if world_size > 1:
                    # 同步AMP dtype
                    dtype_list = [dtype] if RANK == 0 else [None]
                    dist.broadcast_object_list(dtype_list, src=0)
                    dtype = dtype_list[0]

                # move data to device  3.518
                batch_samples = self.preprocess_data(batch_samples)  # type: ignore[arg-type]

                # Forward  15.59
                with autocast(device_type=self.device.type, dtype=dtype, enabled=self.amp):
                    self.loss, self.loss_items = self.model(batch_samples)

                # 添加 L1 正则化
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                self.loss += self.config_manager.core["l1_norm"] * l1_norm

                # loss 缩放为了在 accumulate 次平均后等价
                self.loss = self.loss / self.accumulate

                # Backward  23
                self.scaler.scale(self.loss).backward()

                # 梯度累积够了，就更新参数  18.51
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
                        pbar.set_title(f"\n{"train":^5}|"
                                       f"{"batch":^15}|"
                                       f"{"epoch":^15}|"
                                       f"{"GPU_Mem":^15}|"
                                       f"{"lr":^15}|"
                                       f"{s_loss_names}|"
                                       )

                    s_batch = f"{self.train_dataloader.batch_size}"
                    s_epoch = f"{current_epoch + 1}/{self.epochs}"
                    s_memory = f"{self._get_memory():.3g}G"
                    s_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_description(
                        f"{"train":^5}|"
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
            if world_size > 1:
                fitness_tensor = torch.tensor(self.fitness, dtype=torch.float32, device=self.device)
                dist.broadcast(fitness_tensor, src=0)
                self.fitness = fitness_tensor.item()

            if self.best_fitness < self.fitness:
                self.best_epoch = current_epoch + 1
                self.best_fitness = self.fitness

            if LOCAL_RANK in {-1, 0}:
                # save model
                self.save_model(world_size, current_epoch)
                self.callbacks.run_callback(self, "on_model_save")

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
                break

            # scheduler
            self.scheduler_step(current_epoch)

            # continue next epoch
            self._clear_memory()
            current_epoch += 1

            self.callbacks.run_callback(self, "on_train_epoch_end")

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
            # plot and save train result to csv file
            if self.train_result:
                self.train_result.plot(start=self.start_epoch + 1)
                self.train_result.save_csv()

            # export simplified model
            self.simplified_model(world_size)

        # close train result
        if self.train_result:
            self.train_result.close()

        self.callbacks.run_callback(self, "on_train_end")
        LOGGER.info(f"train results saved at {self.save_dir}")

    def set_ddp(self, world_size: int):
        """
        设置分布式数据并行（DDP），和destroy_ddp函数是一对。
        使用set_ddp函数后必须使用destroy_ddp函数。

        @param world_size: 分布式训练中的进程数量。
        """
        from torch.distributed import init_process_group

        torch.cuda.set_device(LOCAL_RANK)
        self.device = torch.device("cuda", LOCAL_RANK)

        init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size
        )

    @staticmethod
    def destroy_ddp():
        """
        销毁分布式数据并行（DDP），和set_ddp函数是一对。
        使用set_ddp函数后必须使用destroy_ddp函数。
        """
        from torch.distributed import destroy_process_group
        destroy_process_group()

    def set_save_dir(self, world_size: int):
        """
        设置模型训练完的保存路径。
        每个节点内部由 LOCAL_RANK==0 负责创建目录并广播给本节点其它进程。
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
        检查设备配置并返回可用设备。

        @return: 主机的device。
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
                        LOGGER.info(f"Device Type: CUDA, Device {",".join(map(str, false_device))} not found, only using {",".join(map(str, true_device))}")
                else:
                    if "LOCAL_RANK" not in os.environ:
                        LOGGER.info(f"Device Type: CUDA, using {",".join(map(str, true_device))}")
            else:
                raise TypeError(f"device type: {type(device)} is not supported!")

            return torch.device("cuda:0")

    def set_dataset_dir(self):
        """
        设置数据集路径。
        """

        dataset_root_dirs = self.config_manager.dataset["path"]

        if isinstance(dataset_root_dirs, list):
            self.train_dir = []
            self.val_dir = []
            self.test_dir = []
            for dataset_root_dir in dataset_root_dirs:
                # train
                train_dir = (Path(dataset_root_dir) / self.config_manager.dataset["train"]).resolve()
                if train_dir.exists():
                    self.train_dir.append(train_dir)
                # validation
                val_dir = (Path(dataset_root_dir) / self.config_manager.dataset["val"]).resolve()
                if val_dir.exists():
                    self.val_dir.append(val_dir)
                # test
                test_dir: Path = (Path(dataset_root_dir) / self.config_manager.dataset["test"]).resolve()
                if test_dir.exists():
                    self.test_dir.append(test_dir)
        else:
            # train
            train_dir = (Path(dataset_root_dirs) / self.config_manager.dataset["train"]).resolve()
            if train_dir.exists():
                self.train_dir = train_dir
            # validation
            val_dir = (Path(dataset_root_dirs) / self.config_manager.dataset["val"]).resolve()
            if val_dir.exists():
                self.val_dir = val_dir
            # test
            test_dir = (Path(dataset_root_dirs) / self.config_manager.dataset["test"]).resolve()
            if test_dir.exists():
                self.test_dir = test_dir

    def check_amp(self, world_size: int):
        """
        检查是否支持自动混合精度（AMP）。

        @param world_size: 分布式训练中的进程数量。
        """

        # DDP 支持 AMP，直接启动，不进行检查
        if world_size > 1:
            self.amp = True
            if RANK in {-1, 0}:
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
        检查批量大小是否合理。

        @param world_size: 分布式训练中的进程数量。
        """
        # 检查批量大小是否能被进程数量整除
        if world_size > 1:
            if self.batch_size % world_size != 0:
                raise ValueError(f"Batch size {self.batch_size} cannot be evenly divided by the number of processes {world_size}.")
            if self.batch_size // world_size == 1:
                raise ValueError(f"Batch size {self.batch_size} evenly divided by the number of processes {world_size} cannot be 1.")

        # 检查批量大小是否为 1
        if self.batch_size == 1:
            raise ValueError(f"Batch size {self.batch_size} cannot be 1.")

        # 检查批量大小是否为 16 的倍数
        if self.batch_size % 16 != 0:
            LOGGER.warning(
                "Batch size is not a multiple of 16. It is recommended to set batch size as a multiple of 16 for better training performance.")

        # 打印批量大小信息
        if world_size > 1:
            LOGGER.info(f"{world_size} GPU(s) found. Each GPU has a batch size of {self.batch_size // world_size}.")
        else:
            LOGGER.info(f"Training batch size: {self.batch_size}")

    def set_dataloaders(self, world_size: int):
        """
        设置Dataloader。

        @param world_size: 分布式训练中的进程数量。
        """
        # train dataloader
        self.train_dataloader = self.build_dataloader(world_size, mode="train")

        # validation dataloader
        self.val_dataloader = self.build_dataloader(world_size, mode="val")
        self.validator = self.get_validator(world_size)

    def set_optimizer(self, model):
        """
        构建优化器，支持：
        1. 参数分组（bias/norm/其他）
        2. 学习率线性放大（基于全局 batch size）
        3. 在 DDP 下保证所有进程参数顺序完全一致
        """

        optimizer_name = self.config_manager.core["optimizer"]
        lr0 = self.config_manager.core["lr0"]
        momentum = self.config_manager.core["momentum"]
        weight_decay = self.config_manager.core["weight_decay"]

        # ---------------- 线性缩放 LR ----------------
        if WORLD_SIZE > 1:
            total_batch = self.batch_size * max(WORLD_SIZE, 1) * self.accumulate
            lr_scaled = lr0 * total_batch / 64.0
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
            elif isinstance(parent_module, norm_types):
                groups["no_decay_norm"].append(param)
            else:
                groups["with_decay"].append(param)

        # ---------------- 构造优化器 ----------------
        common_kwargs = {"lr": lr_scaled}
        if optimizer_name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            common_kwargs.update(betas=(momentum, 0.999), weight_decay=0.0)
            opt_cls = getattr(optim, optimizer_name, optim.Adam)
        elif optimizer_name == "RMSProp":
            common_kwargs.update(momentum=momentum)
            opt_cls = optim.RMSprop
        elif optimizer_name == "SGD":
            common_kwargs.update(momentum=momentum, nesterov=True)
            opt_cls = optim.SGD
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' is not supported.")

        # 先建立优化器（至少需要一个参数组）
        optimizer = opt_cls(groups["no_decay_bias"], **common_kwargs)

        # 追加剩余参数组
        if groups["with_decay"]:
            optimizer.add_param_group({"params": groups["with_decay"],
                                       "weight_decay": weight_decay})
        if groups["no_decay_norm"]:
            optimizer.add_param_group({"params": groups["no_decay_norm"],
                                       "weight_decay": 0.0})

        self.optimizer = optimizer

    def set_warmup_scheduler(self):
        """
        设置预热训练阶段的学习率调度器。
        """

        lr0 = self.config_manager.core["lr0"]
        warmup_lr = self.config_manager.core["warmup_lr"]
        start_epoch = self.start_epoch

        # set warmup_scheduler
        if start_epoch < self.warmup_epochs:
            warmup_start_epochs = start_epoch - 1
        else:
            warmup_start_epochs = self.warmup_epochs - 1

        if self.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_lr / lr0,
                end_factor=1,
                total_iters=self.warmup_epochs,
                last_epoch=warmup_start_epochs if warmup_start_epochs != 0 else -1
            )

    def set_normal_scheduler(self):
        """
        设置正式训练阶段的学习率调度器。
        """

        lr0 = self.config_manager.core["lr0"]
        lr1 = self.config_manager.core["lr1"]
        start_epoch = self.start_epoch

        # set normal scheduler
        if start_epoch < self.warmup_epochs:
            normal_num_epochs = self.epochs - self.warmup_epochs
            normal_start_epochs = -1
        else:
            normal_num_epochs = self.epochs - start_epoch
            normal_start_epochs = start_epoch - self.warmup_epochs - 1

        scheduler_name = self.config_manager.core["scheduler"]
        if scheduler_name == "LinearLR":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1,
                end_factor=lr1,
                total_iters=normal_num_epochs,
                last_epoch=normal_start_epochs
            )
        elif scheduler_name == "CosineLR":
            if self.epochs < 100:
                t_max = self.epochs
            else:
                t_max = self.epochs // 10
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=lr0 * lr1,
                last_epoch=normal_start_epochs
            )
        elif scheduler_name == "auto":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
            raise NotImplementedError(f"Scheduler '{scheduler_name}' is not supported.")

        self.scheduler = scheduler

    def set_model(self, world_size: int):
        """
        设置模型，并在 DDP 模式下自动启用 SyncBatchNorm。
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

        # 计算每个进程的批量大小
        batch_size = self.batch_size // max(world_size, 1)

        # 构建数据集
        dataset = self.build_dataset(mode=mode)

        # 计算每个 rank 实际分到的样本数
        shuffle = mode == "train"
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            effective_samples = len(sampler)
        else:
            sampler = None
            effective_samples = len(dataset)

        # 确保 batch_size 不超过实际样本数
        batch_size = min(batch_size, effective_samples)
        if batch_size <= 0:
            raise ValueError(
                f"Dataset too small for DDP: rank {RANK} has {effective_samples} samples, "
                f"but batch_size is {batch_size}. Reduce world_size or increase dataset."
            )

        # 计算 num_workers
        num_devices = torch.cuda.device_count()
        num_workers = min(8, os.cpu_count() // max(num_devices, 1), self.config_manager.core["workers"])

        # 生成器（用于可复现性）
        generator = torch.Generator()
        generator.manual_seed(self.config_manager.core["seed"] + RANK)

        # 调整 val/test 的 batch_size 和 shuffle
        if mode != "train":
            batch_size = max(1, batch_size // 2)
            shuffle = shuffle and self.config_manager.core["shuffle_val_dataloader"]

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # 由 sampler 控制 shuffle
            num_workers=num_workers if mode == "train" else min(1, num_workers//2),
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn", None),
            generator=generator,
            persistent_workers=num_workers > 0,
            drop_last=False,
            pin_memory=True,
            pin_memory_device=str(self.device),
            prefetch_factor=min(4, max(2, os.cpu_count() // max(world_size, 1))) if num_workers > 0 else None,
            multiprocessing_context='spawn' if os.name != 'nt' else None,
        )
        return dataloader

    def resume_training(self):
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
        执行优化器步骤，包括梯度裁剪和 EMA 更新。
        """
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        grad_clip = self.config_manager.core["grad_clip"]
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.model.zero_grad(set_to_none=True)
        # update ema
        if LOCAL_RANK in {-1, 0} and self.ema is not None:
            self.ema.update(self.model)

    def scheduler_step(self, current_epoch: int):
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
        保存配置文件
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
        获取验证模型所需的验证器。若重载，则需要与do_validate配合使用。

        @param world_size: 分布式训练中的进程数量。
        @return: 验证器实例。
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "validator")(self, world_size)

    def do_validate(self) -> float:
        """
        子类通过重载该函数来验证模型性能，也可以选择不重载不进行验证。若重载，则需要与get_validator配合使用。

        @return: 返回fitness用于评估模型训练的好坏程度。
        """
        return self.validator.validate() if self.validator else 0.

    def get_model_instance(self, world_size: int) -> nn.Module:
        """
        获取要保存的模型实例。
        """
        if self.ema:
            model = self.ema.ema_model
        else:
            model = self.model if world_size <= 1 else self.model.module
        return model

    def save_model(self, world_size: int, current_epoch: int):
        """
        保存模型，并同步 best_fitness / best_epoch 给所有进程。
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

            # 保存最新的模型
            if LOCAL_RANK in {-1, 0}:
                torch.save(checkpoint, self.last_pt.as_posix())

            # 保存最佳模型（仅主进程）
            if LOCAL_RANK in {-1, 0} and self.best_fitness == self.fitness:
                torch.save(checkpoint, self.best_pt.as_posix())

            # 按周期保存模型（仅主进程）
            save_period = self.config_manager.core["save_period"]
            if LOCAL_RANK in {-1, 0} and save_period > 0 and (current_epoch + 1) % save_period == 0:
                epoch_checkpoint_path = Path(self.weight_dir / f"epoch_{current_epoch + 1}.pt")
                torch.save(checkpoint, epoch_checkpoint_path.as_posix())

        except Exception as e:
            LOGGER.error(f"Error occurred while saving model: {e}")
            raise e

    def simplified_model(self, world_size: int):
        """
        精简化模型压缩模型文件大小并保存为指定格式，用于部署。
        该函数的主要功能是根据配置决定是否将模型转换为半精度（fp16），然后将模型的 `state_dict` 保存为一个检查点文件。
        如果配置中指定了 `FP16_pt` 为 `True`，则将模型的参数转换为 `float16` 格式，否则保持原格式。

        @param world_size: 分布式训练中的进程数量。
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
        获取当前设备的内存/显存使用情况。

        @return: 当前设备的内存使用量（单位：GB）。
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
        # 传递stop信号
        self.stop = True
        self.do_validate()
