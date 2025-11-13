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

from __future__ import annotations

import os
import gc
import sys
import time
import warnings
import psutil
import torch
import torch.distributed as dist

from typing import TYPE_CHECKING, List, Union
from datetime import timedelta
from pathlib import Path
from torch import autocast, optim, nn, Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from tinytrain.cfg import TTConfigManager, TTEngineRegistry
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.global_var import RANK, LOCAL_RANK, WORLD_SIZE
from tinytrain.metrics.base import TrainResult
from tinytrain.utils import LOGGER
from tinytrain.utils.progress_bar import TTProgressBar
from tinytrain.utils.any_utils import setup_torch_environment, create_iter_directory, maybe_limit_num_workers
from tinytrain.utils.callback import Callback, Events
from tinytrain.utils.checks import check_amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tinytrain.utils.train_utils import ModelEMA, EarlyStopping

if TYPE_CHECKING:
    from .model import TTBaseModel
    from .validator import TTBaseValidator


class TTBaseTrainer:
    """
    TTBaseTrainer 是一个通用、可扩展的深度学习训练框架基类，支持单机单卡、单机多卡（DDP）、多机多卡等多种训练模式。

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
    - 所有配置通过 `TTConfigManager` 统一管理，支持 TOML 文件。
    - 所有日志通过 `LOGGER` 输出，支持 rank 过滤。
    - 训练结果统一保存在 `save_dir`，包括权重、日志、配置、图表等。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: TTConfigManager, device: torch.device, model: TTBaseModel, callback: Callback):
        """
        初始化 TTBaseTrainer 类，配置训练所需的核心组件。

        Args:
            config_manager (TTConfigManager): 配置管理器，提供训练参数。
            model (TTBaseModel): 模型实例，用于训练和验证。
            callback (Callback): 回调函数，用于在训练过程中执行自定义操作。
        """
        self.config_manager: TTConfigManager = config_manager
        self.validator: TTBaseValidator | None = None

        # resume
        self.resume: bool = self.config_manager.core["resume"]

        # device
        self.device: torch.device = device

        # amp
        self.amp: bool | str = self.config_manager.core["amp"]
        self.scaler: torch.amp.GradScaler | None = None

        # seed
        setup_torch_environment(self.config_manager.core["seed"], self.config_manager.core["deterministic"])

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
        self.test_dataloader = None

        # model
        self.model: TTBaseModel = model
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
        self.intra_node_group = None  # DDP分组

    def train(self):
        """
        启动模型训练，支持单机单卡、单机多卡、多机多卡等多种训练方式。
        """
        self._perform_training(WORLD_SIZE)

    def _perform_training(self, world_size):
        """
        执行实际训练逻辑，包括 DDP 初始化、训练、异常处理和资源清理。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        try:
            if world_size > 1:
                LOGGER.info("Initializing DDP...")
                self.set_ddp()
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
            LOGGER.info("Done.")

    # ------------------------------------------------------------------
    # 2. 数据与预处理（子类可重写）
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

    def freeze_layers(self, model: TTBaseModel, world_size: int):
        """
        冻结模型的某些层，防止其参数在训练中被更新。

        Args:
            model (TTBaseModel): 模型实例
            world_size (int): 分布式训练中的进程数量。
        """
        pass

    def make_param_groups(self, model: TTBaseModel, lr, weight_decay, **kwargs) -> dict:
        """
        将模型参数划分为不同的优化组（parameter group），用于后续构造优化器。

        子类可通过重写本函数实现「主干网络与头部网络使用不同学习率 / 权重衰减」
        或「对某些层跳过权重衰减」等需求。

        必须返回一个 dict，每个 key 为组名，value 为 dict，且必须包含：
            - "name" : str, 组名（仅调试 / 打印）
            - "params": List[Tensor], 该组待优化参数
            - "lr"   : float, 该组学习率
            - "weight_decay": float, 该组权重衰减系数

        若返回空 dict，则 Trainer 会自动生成一个默认组，包含 model.parameters()。

        Args:
            model (TTBaseModel): 当前待训练模型。
            lr (float): 当前经过 world_size 与 accumulate 缩放后的初始学习率。
            weight_decay (float): 全局权重衰减系数。
            **kwargs: 预留扩展字段，例如 momentum、betas 等。

        Returns:
            dict: 优化器参数组字典；空 dict 表示使用默认分组。
        """
        return {}

    def add_extra_save_params(self, model: TTBaseModel) -> dict:
        """
        在保存 checkpoint 时，向 dict 中追加自定义字段。

        子类可重写本函数把「类别名列表、anchor 尺寸、tokenizer 配置」等
        与模型强相关的信息一并写入 *.pt，方便后续推理或断点续训。

        Args:
            model (TTBaseModel): 当前待保存的模型（可能是 EMA 或原始模型）。

        Returns:
            dict: 需要额外写入 checkpoint 的 key-value 字典；无额外内容时返回空 dict。
        """
        return {}

    def load_extra_save_params(self, model: TTBaseModel) -> None:
        """
        从 checkpoint 中读取并恢复由 add_extra_save_params 写入的自定义字段。

        本函数在 resume_training() 中被调用，此时 checkpoint 已加载到 CPU，
        可通过 self.config_manager.link["model"] 获取权重路径，再按 key 取出所需字段
        并赋值给 model 或 trainer 的相应属性。

        Args:
            model (TTBaseModel): 当前正在构建的模型实例。

        Returns:
            None
        """
        pass

    # ------------------------------------------------------------------
    # 3. 训练前准备
    # ------------------------------------------------------------------
    def _before_train(self, world_size: int):
        """
        训练前的准备工作，包括路径设置、模型初始化、优化器配置等。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        self.callbacks.run_callback(Events.ON_PREPARE_TRAIN_START, self)

        # set save dir
        self.set_save_dir(world_size)

        # set dataset dir
        self.set_dataset_dir()

        # set train result
        if RANK in {-1, 0}:
            self.train_result = TrainResult(config_manager=self.config_manager, save_dir=self.save_dir)

        # check batch size
        self.check_batch_size(world_size)

        # set dataloaders
        self.set_dataloaders(world_size)

        # set model
        self.set_model(world_size)

        # only val
        if self.config_manager.core["only_val"]:
            self.only_do_validate()
            sys.exit("Validation finished, exiting.")

        # set optimizer
        self.set_optimizer(world_size)

        # resume training
        self.resume_training()

        # set scheduler
        self.set_warmup_scheduler()
        self.set_standard_scheduler()

        # EarlyStopping
        self.early_stopping = EarlyStopping(self.config_manager.core["patience"])

        # save config file
        if RANK in {-1, 0}:
            self.save_config_file()

        # plot
        if RANK in {-1, 0}:
            self.plot_something_before_train()

        self.callbacks.run_callback(Events.ON_PREPARE_TRAIN_END, self)

        # DDP synchronize
        if world_size > 1:
            dist.barrier()

    def set_save_dir(self, world_size: int):
        """
        设置训练结果保存路径，支持多节点同步。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        LOGGER.info(f"Setting save directory...")
        # 节点内主进程（LOCAL_RANK==0）负责创建目录
        if LOCAL_RANK in {-1, 0}:
            config_core = self.config_manager.core
            save_dir = Path(config_core["save_dir"]).resolve()
            project_name = config_core["project_name"] or "default_project"
            config_core["project_name"] = project_name
            save_dir = save_dir / project_name / config_core["task"] / "train"
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

        LOGGER.info(f"Save directory setting completed. ✅")

    def set_dataset_dir(self) -> None:
        """
        设置训练、验证和测试数据集路径。
        统一返回 list，即使只有一个数据集。
        """
        LOGGER.info(f"Setting dataset directory...")

        def _get_dirs(dataset_dirs: Union[str, List[str]]) -> List[Path]:
            """辅助函数：根据 split_key 获取所有存在的路径列表"""
            dataset_dirs = [dataset_dirs] if isinstance(dataset_dirs, (str, Path)) else dataset_dirs
            dirs = [
                Path(dataset_dir).resolve()
                for dataset_dir in dataset_dirs
            ]
            for d in dirs:
                if not d.exists():
                    raise FileNotFoundError(f"Dataset path: {d} not found.")
            return dirs

        if self.config_manager.dataset["train"]:
            self.train_dir = _get_dirs(self.config_manager.dataset["train"])

        if self.config_manager.dataset["val"]:
            self.val_dir = _get_dirs(self.config_manager.dataset["val"])

        if self.config_manager.dataset.get("test"):
            self.test_dir = _get_dirs(self.config_manager.dataset["test"])

        LOGGER.info(f"Dataset directory setting completed ✅")

    def check_batch_size(self, world_size: int):
        """
        检查 batch_size 是否合理，是否支持 DDP。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        LOGGER.info(f"Checking batch size...")

        # 检查批量大小是否为 1
        if self.batch_size == 1:
            raise ValueError(f"Batch size {self.batch_size} cannot be 1.")

        # 检查批量大小是否为 16 的倍数
        if self.batch_size % 16 != 0:
            LOGGER.warning("Batch size is not a multiple of 16. It is recommended to set batch size as a multiple of 16 for better training performance.")

        # 打印批量大小信息
        if world_size > 1:
            LOGGER.info(f"{world_size} GPU(s) found. Each GPU has a batch size of {self.batch_size}.")
        else:
            LOGGER.info(f"Training batch size: {self.batch_size} ✅")

    def check_amp(self, world_size: int):
        """
        检查是否支持自动混合精度（AMP），并初始化 GradScaler。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        LOGGER.info(f"Checking AMP...")

        if isinstance(self.amp, str):
            assert self.amp == "force", f"amp must be [true、 false、'force'], got {self.amp}"

        if self.amp == "force":
            LOGGER.warning(f"Forcing AMP activation may result in precision loss."
                           f" It is recommended to set amp=True to check if AMP is available, ensuring no significant precision loss occurs.")
            LOGGER.info("AMP: force passed ✅")
            # 强制开启amp
            self.amp = True
        elif self.amp is True:
            do_amp = False
            if self.device.type == "cpu":
                LOGGER.warning("CPU does not support AMP. Disabling AMP.")
            elif self.device.type == "mps":
                LOGGER.warning("Apple MPS does not support AMP. Disabling AMP.")
            elif self.device.type == "cuda":
                do_amp = True
            else:
                LOGGER.warning(f"Device '{self.device.type}' AMP not verified, disabling.")

            if do_amp:
                LOGGER.info("AMP: running Automatic Mixed Precision (AMP) checks...")
                # 支持手动关闭 AMP 检查，如需关闭 AMP 检查，请注释下面这一行代码
                self.amp = check_amp(self)
                if self.amp:
                    LOGGER.info("AMP: checks passed ✅")
                else:
                    LOGGER.warning("AMP: checks failed ❌")

        # ---------- 同步 ----------
        if world_size > 1:
            amp_tensor = torch.tensor([self.amp], dtype=torch.uint8, device=self.device)
            if RANK != -1:
                dist.broadcast(amp_tensor, src=0)
            self.amp = bool(amp_tensor.item())
            if RANK == 0:
                LOGGER.info(f"AMP synchronised, final enabled={self.amp}")

        # 初始化 GradScaler
        try:
            # PyTorch ≥ 2.0 通用 GradScaler
            self.scaler = torch.amp.GradScaler(enabled=self.amp, growth_interval=100)
        except AttributeError:
            # 旧版只能走 cuda.amp
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp, growth_interval=100)

    def set_dataloaders(self, world_size: int):
        """
        构建训练与验证的 DataLoader。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        LOGGER.info(f"Initializing dataLoader. This may take a few moments for large datasets. Please wait patiently...")
        only_val = self.config_manager.core["only_val"]

        # train dataloader
        if self.train_dir and not only_val:
            self.train_dataloader = self.build_dataloader(world_size, mode="train")

        # validate dataloader
        if self.val_dir:
            self.val_dataloader = self.build_dataloader(world_size, mode="val")

        if self.val_dataloader is None:
            if only_val:
                raise RuntimeError("only_val is True, but valid directory is not provided.")
        else:
            self.validator = self.get_validator(world_size)
            if only_val:
                if self.validator is None:
                    raise RuntimeError("only_val is True, but validator engine is not bound.")
                return  # 仅验证模式下，初始化完 validator 就可以返回了

        # test dataloader
        if self.test_dir:
            self.test_dataloader = self.build_dataloader(world_size, mode="test")

        LOGGER.info(f"DataLoader initialization completed ✅")

    def build_dataloader(self, world_size: int, mode: str = "train", shuffle_val=False, shuffle_test=False, dist_val=False, dist_test=False):
        """
        构建 PyTorch 的 DataLoader，支持以下高级特性：
        1. 分布式训练（DDP）自动适配；
        2. 动态 batch_size 保证每张卡样本数合法；
        3. 动态 num_workers 调整，兼顾 Linux 共享内存限制；
        4. 可复现性：通过固定 Generator seed；
        5. 训练/验证/测试三种模式差异化配置；
        6. 自动处理 BatchNorm 对 batch_size 的最小样本数要求；
        7. 自动选择 pinned memory、prefetch_factor、multiprocessing_context 等性能优化参数。

        Args:
            world_size (int):
                当前分布式训练的总进程数（即 GPU 数）。单卡训练时传入 1。
            mode (str, optional):
                数据加载模式，可选值为：
                - "train" : 训练模式，默认 shuffle=True，drop_last=True；
                - "val"   : 验证模式，默认 shuffle 由 shuffle_val 控制；
                - "test"  : 测试模式，默认 shuffle 由 shuffle_test 控制。
            shuffle_val (bool, optional):
                仅在 mode="val" 时生效，控制是否打乱验证集。默认 False，保证验证结果稳定。
            shuffle_test (bool, optional):
                仅在 mode="test" 时生效，控制是否打乱测试集。默认 False，保证测试推理结果可复现。
            dist_val (bool, optional):
                是否在验证阶段启用分布式采样（DistributedSampler）。默认 False。
                当 world_size > 1 且 dist_val=True 时，每张卡只加载部分验证数据。
            dist_test (bool, optional):
                是否在测试阶段启用分布式采样。默认 False。
                当 world_size > 1 且 dist_test=True 时，每张卡只加载部分测试数据。

        Returns:
            torch.utils.data.DataLoader:
                配置完毕的数据加载器，可直接用于训练/验证/测试循环。

        Raises:
            AssertionError:
                若 mode 不在 ["train", "val", "test"] 范围内。
            ValueError:
                1. 当前 rank 分到的样本数小于 batch_size 时；
                2. 自动调整后的 batch_size < 2 时（BatchNorm 需要至少 2 个样本计算统计量）。

        Notes:
            - 当 world_size > 1 且模式满足分布式条件时，会自动使用 `DistributedSampler`，
              并强制 `drop_last=True`，确保所有卡样本数一致，防止 all_gather 永久阻塞。
            - 在 Linux 平台会自动检测 `/dev/shm` 共享内存剩余空间，
              若不足会自适应下调 num_workers，防止 DataLoader 因共享内存不足而崩溃。
            - 验证/测试阶段会自动将 batch_size 和 num_workers 减半，降低显存与 CPU 占用。
            - 最终 num_workers 还会再按 world_size 均摊一次，防止单节点进程数过多。
            - 使用 `spawn` 启动子进程（非 Windows）以兼容 CUDA 多进程要求。
            - 通过 `persistent_workers=True` 减少 epoch 之间 worker 重建开销。
            - 通过 `pin_memory=True` 与 `pin_memory_device` 加速 GPU 传输。
            - 通过 `prefetch_factor` 控制预加载批次数，根据 CPU 核心数与 world_size 自动调整。

        Examples:
            >>> # 单卡训练
            >>> train_loader = self.build_dataloader(world_size=1, mode="train")
            >>> # 4 卡分布式验证，启用分布式采样
            >>> val_loader = self.build_dataloader(world_size=4, mode="val", dist_val=True)
            >>> # 8 卡测试，不 shuffle，启用分布式采样
            >>> test_loader = self.build_dataloader(world_size=8, mode="test", dist_test=True)
        """

        assert mode in ["train", "val", "test"], "build dataloader mode must be 'train' or 'val' or 'test'"
        # 构建数据集
        dataset = self.build_dataset(mode=mode)

        # 判断是否shuffle
        if mode == "train":
            shuffle = True
        elif mode == "val":
            shuffle = shuffle_val
        elif mode == "test":
            shuffle = shuffle_test
        else:
            raise TypeError(f"build dataloader mode must be 'train' or 'val' or 'test'")

        # 计算每个 rank 实际分到的样本数
        if world_size > 1 and (mode == "train" or (mode == "val" and dist_val) or (mode == "val" and dist_test)):
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, drop_last=True)  # drop_last为True保证每张卡样本数量一致，避免all_gather永久阻塞
            effective_samples = len(sampler)

            if self.batch_size > effective_samples:
                raise ValueError(
                    f"Dataset too small for DDP: rank {RANK} has {effective_samples} samples, "
                    f"but batch_size is {self.batch_size}. Reduce batch_size or increase dataset samples."
                )
        else:
            sampler = None
            effective_samples = len(dataset)  # type: ignore[arg-type]

        # 确保 batch_size 不超过实际样本数
        batch_size = min(self.batch_size, effective_samples)
        if batch_size < 2:
            raise ValueError(
                f"current each device batch_size={batch_size}, but BatchNorm requires >=2 samples to compute statistics. "
                f"Please make sure the batch_size>=2."
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
            batch_size = max(1, batch_size // 2)
            num_workers = max(0, num_workers // 2)

        # DDP均摊 num_workers
        # if world_size > 1:
        #     num_workers = max(0, num_workers // world_size)

        # should pin memory
        can_pin_memory = self._should_pin_memory(world_size)

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,  # 由 sampler 控制 shuffle
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn"),  # 要求数据集必须实现collate_fn函数
            generator=generator,
            persistent_workers=num_workers > 0,
            drop_last=True,
            pin_memory=can_pin_memory,
            prefetch_factor=min(4, max(2, os.cpu_count() // max(world_size, 1))) if num_workers > 0 else None,
            multiprocessing_context='spawn' if os.name != 'nt' and num_workers > 0 else None,
        )
        return dataloader

    def set_model(self, world_size: int):
        """
        设置模型，包括移动到设备、启用 SyncBatchNorm、封装 DDP 和 EMA。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        LOGGER.info(f"Setting up model...")

        # freeze layers
        self.freeze_layers(self.model, world_size)

        # bind loss
        if self.model.criterion is None:
            self.model.criterion = self.model.init_criterion()

        # load extra save params
        if self.config_manager.link["model"].suffix in {".pt", ".pth"}:
            self.load_extra_save_params(self.model)

        # model -> device
        self.model = self.model.to(self.device)

        # check AMP
        if not self.config_manager.core["only_val"]:
            self.check_amp(world_size)

        # convert to DDP model
        self.convert_ddp_model(world_size)

        # EMA
        if self.config_manager.core["ema"]:
            self.ema = ModelEMA(self.get_model_instance(world_size))
            LOGGER.info("EMA(Exponential Moving Average) is enabled.")

        LOGGER.info(f"Model setup completed ✅")

    def convert_ddp_model(self, world_size: int) -> None:
        """
        将单卡模型转换为分布式训练模型（SyncBN + DDP）。

        当 world_size > 1 时，本函数会：
        1. 把普通 BN 层替换为 SyncBatchNorm；
        2. 用 DistributedDataParallel 封装模型，并绑定到 LOCAL_RANK。

        Args:
            world_size (int): 当前分布式进程总数。

        Returns:
            None
        """
        if world_size <= 1:
            return

        # 多卡情况下：先转 SyncBN，再封装 DDP
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  # 把普通 BN → SyncBatchNorm
        self.model = DDP(
            self.model,
            device_ids=[LOCAL_RANK],
            gradient_as_bucket_view=True
        )

    def set_optimizer(self, world_size: int, optimizer=None):
        """
        构建优化器，支持参数分组、学习率缩放、多种优化器选择。

        Args:
            world_size (int): 当前分布式进程总数。
            optimizer : 支持传入指定的优化器。
        """
        LOGGER.info(f"Setting optimizer...")

        if optimizer:
            self.optimizer = optimizer
            return

        # ---------------- 1. 超参 ----------------
        optimizer_name = self.config_manager.core["optimizer"]
        lr0 = self.config_manager.core["lr0"]
        momentum = self.config_manager.core["momentum"]
        weight_decay = self.config_manager.core["weight_decay"]

        # 线性缩放 LR
        if world_size > 1:
            lr_scaled = lr0 * max(world_size, 1) * self.accumulate
            LOGGER.info(f"DDP LR scaled from {lr0} -> {lr_scaled} ")
        else:
            lr_scaled = lr0

        # ---------------- 2. 默认分组骨架 ----------------
        groups = self.make_param_groups(model=self.model, lr=lr_scaled, weight_decay=weight_decay)
        if not groups:
            groups = {
                "default": {"params": self.model.parameters(), "weight_decay": weight_decay, "lr": lr_scaled},
            }

        # ---------------- 4. 构造优化器 ----------------
        param_groups = [v for v in groups.values() if v["params"]]  # 只保留非空组

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

        # 获取优化器类与默认 kwargs
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
            LOGGER.warning(f"Optimizer '{optimizer_name}' not supported, fallback to AdamW")
            optimizer_name = "AdamW"

        opt_cls, opt_kwargs = registry[optimizer_name]
        # 合并公共 lr（已写进分组）
        opt_kwargs.pop("lr", None)
        opt_kwargs.pop("weight_decay", None)

        # 构建骨架
        if "lr" in param_groups[0]:
            opt_kwargs["lr"] = param_groups[0]["lr"]
        if "weight_decay" in param_groups[0]:
            opt_kwargs["weight_decay"] = param_groups[0]["weight_decay"]
        self.optimizer = opt_cls(param_groups[0]["params"], **opt_kwargs)
        for param_group in param_groups[1:]:
            self.optimizer.add_param_group(param_group)

        # 提前初始化 Adagrad 的状态字典
        if isinstance(self.optimizer, torch.optim.Adagrad):
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["sum"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        LOGGER.info(f"Optimizer setting completed ✅")

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

    def set_warmup_scheduler(self, warmup_scheduler=None):
        """
        设置预热阶段的学习率调度器。

        Args:
            warmup_scheduler : 支持传入指定的warmup学习率调度器。
        """
        LOGGER.info(f"Setting warmup scheduler...")

        if self.warmup_epochs <= 0:
            return
        if self.start_epoch >= self.warmup_epochs:
            return

        if warmup_scheduler:
            self.warmup_scheduler = warmup_scheduler
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

        LOGGER.info(f"Warmup scheduler setting completed ✅")

    def set_standard_scheduler(self, standard_scheduler=None):
        """
        设置正式训练阶段的学习率调度器。

        Args:
            standard_scheduler : 支持传入指定的学习率调度器。
        """
        LOGGER.info(f"Setting standard scheduler...")

        if self.epochs <= self.warmup_epochs:
            return

        if standard_scheduler:
            self.scheduler = standard_scheduler
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
                end_factor=lr0 * lr1,
                total_iters=normal_epochs,
                last_epoch=last_epoch
            )
        elif scheduler_name == "CosineLR":
            t_max = normal_epochs if normal_epochs < self.config_manager.core["t_max"] else self.config_manager.core["t_max"]
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
                patience=self.config_manager.core["plateau_epochs"],
                factor=self.config_manager.core["plateau_factor"],
                threshold=self.config_manager.core["plateau_sensitivity"],
                threshold_mode="rel",
                cooldown=0,
                min_lr=lr0 * lr1,
            )
        else:
            LOGGER.warning(f"Unknown standard scheduler '{scheduler_name}', fallback to LinearLR.")
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr0 * lr1,
                total_iters=normal_epochs,
                last_epoch=last_epoch
            )

        LOGGER.info(f"Standard scheduler setting completed ✅")

    def save_config_file(self):
        """
        将配置保存为 TOML 文件到指定目录。
        """
        import toml

        LOGGER.info(f"Saving all config files...")

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

        LOGGER.info(f"All config files saved in directory -> {self.args_dir}")

    # ------------------------------------------------------------------
    # 4. 训练主循环
    # ------------------------------------------------------------------
    def _do_train(self, world_size: int):
        """
        执行模型训练主循环，包括前向、反向、验证、保存等。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        # 训练前检查
        self._before_train(world_size)

        current_epoch = self.start_epoch  # 一个epoch有几个batch
        num_batch = len(self.train_dataloader)  # 整个训练所有epoch一共有多少个batch
        last_opt_step = -1  # 记录上一次真正更新参数的 batch 序号

        # 先清一次显存,梯度归零
        self._clear_memory()
        self.optimizer.zero_grad()

        # 设置数据精度类型
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() and self.config_manager.core["bf16"] else torch.float16
        if world_size > 1:
            # 同步AMP dtype
            dtype_list = [dtype] if RANK == 0 else [None]
            dist.broadcast_object_list(dtype_list, src=0)
            dtype = dtype_list[0]

        # 正式训练
        LOGGER.info(f"start training...")
        train_time_start = time.time()
        self.callbacks.run_callback(Events.ON_TRAIN_START, self)
        while True:
            self.callbacks.run_callback(Events.ON_TRAIN_EPOCH_START, self)
            self.model.train()

            # dataloader sampler
            if world_size > 1:
                self.train_dataloader.sampler.set_epoch(current_epoch)

            # 设置打印进度条
            pbar = TTProgressBar(self.train_dataloader, total=num_batch)

            # 开启epoch循环
            smooth_loss_items = None
            for i, batch_samples in enumerate(pbar):
                self.callbacks.run_callback(Events.ON_TRAIN_BATCH_START, self)

                # 当前是第几个batch
                current_batch = current_epoch * num_batch + i

                # 判断本次是否真正执行 optimizer.step()
                is_last_accum_step = (current_batch - last_opt_step) == self.accumulate

                # move data to device
                batch_samples = self.preprocess_data(batch_samples)  # type: ignore[arg-type]

                # Forward
                with autocast(device_type=self.device.type, dtype=dtype, enabled=self.amp):
                    self.loss, self.loss_items = self.execute_forward(batch_samples)

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

                self.callbacks.run_callback(Events.ON_TRAIN_BATCH_END, self)

            if RANK in {-1, 0}:
                # train result
                self.train_result.add("lr", self.optimizer.param_groups[0]["lr"])
                for loss_name, loss_value in smooth_loss_items.items():
                    self.train_result.add(loss_name, loss_value.item())

            # validation
            if world_size > 1:
                dist.barrier()
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

            # save model
            self.save_model(world_size, current_epoch)
            self.callbacks.run_callback(Events.ON_MODEL_SAVE, self)

            # save train result to csv file
            if RANK in {-1, 0} and self.train_result:
                self.train_result.save_csv()

            self.callbacks.run_callback(Events.ON_TRAIN_EPOCH_END, self)

            if world_size > 1:
                dist.barrier()

            # stop training
            stop_flag = False
            if RANK in {-1, 0}:
                if self.config_manager.core["time"] > 0:
                    current_train_time = (time.time() - train_time_start) / 60
                    if current_train_time >= self.config_manager.core["time"]:
                        LOGGER.warning(f"train cost time already is over {self.config_manager.core['time']} minutes, training stopped.")
                        stop_flag = True
                if self.early_stopping(current_epoch, self.fitness):
                    LOGGER.warning(f"Early stopping training, training stopped.")
                    stop_flag = True
                if current_epoch + 1 == self.epochs:
                    stop_flag = True

            # --- 广播 stop 标志 ---
            stop_tensor = torch.tensor(stop_flag, device=self.device)
            if world_size > 1:
                dist.broadcast(stop_tensor, src=0)
            self.stop = stop_tensor.item()

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
        self.load_model_to_final_eval(world_size, checkpoint)

        if world_size > 1:
            dist.barrier()
        self.do_validate()

        # export simplified model
        self.simplified_model(world_size)

        self.callbacks.run_callback(Events.ON_TRAIN_END, self)
        LOGGER.info(f"train results saved at {self.save_dir}")

    def only_do_validate(self):
        """
        仅执行验证流程，不训练。
        """

        self.stop = True  # 传递stop信号
        self.do_validate()

    def execute_forward(self, batch_samples) -> tuple[Tensor, dict[str, Tensor]]:
        """
        执行一次前向传播，返回总损失及各分项损失字典。

        默认行为为调用 model(batch_samples)，子类可重写以实现「多模型级联、
        多任务损失加权、自定义 loss 计算」等逻辑。

        Args:
            batch_samples (BaseBatchDataInfo): 经过 preprocess_data 后的批次数据。

        Returns:
            tuple[Tensor, dict[str, Tensor]]:
                - 第一个元素：用于反向传播的总损失（已聚合各任务权重）。
                - 第二个元素：字典，key 为损失名，value 为对应分项损失（用于日志打印）。
           """
        return self.model(batch_samples)

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
        self.do_grad_clip(self.model)

        # 4. 安全 step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # 5. EMA 更新
        if LOCAL_RANK in {-1, 0} and self.ema is not None:
            self.ema.update(self.get_model_instance(WORLD_SIZE))

    def do_grad_clip(self, model: TTBaseModel) -> None:
        """
        对模型全部参数进行梯度范数裁剪（gradient norm clipping）。

        仅在 config_manager.core["grad_clip"] > 0 时生效，用于防止梯度爆炸。

        Args:
            model (TTBaseModel): 待裁剪的模型（DDP 下为 model.module）。

        Returns:
            None
        """

        grad_clip = self.config_manager.core.get("grad_clip", 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

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

    def save_model(self, world_size: int, current_epoch: int):
        """
        保存模型 checkpoint，包括 last.pt、best.pt、epoch_X.pt。

        Args:
            world_size (int): 分布式训练中的进程数量。
            current_epoch (int): 当前训练轮次。
        """

        if LOCAL_RANK not in {-1, 0}:
            return

        try:
            # 获取要保存的模型
            model = self.get_model_instance(world_size)
            model.eval()

            # 构建检查点
            checkpoint = {
                "current_epoch": current_epoch + 1,
                'model_name': self.config_manager.model["name"],
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "fitness": self.fitness,
                "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
                "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()}
            }

            extra_params: dict = self.add_extra_save_params(model)
            checkpoint.update(extra_params)

            # 保存最新的模型
            torch.save(checkpoint, self.last_pt.as_posix())

            # 保存最佳模型
            if self.best_fitness == self.fitness:
                torch.save(checkpoint, self.best_pt.as_posix())

            # 按周期保存模型
            save_period = self.config_manager.core["save_period"]
            if save_period > 0 and (current_epoch + 1) % save_period == 0:
                epoch_pt = self.weight_dir / f"epoch_{current_epoch + 1}.pt"
                torch.save(checkpoint, epoch_pt.as_posix())

        except Exception as e:
            LOGGER.error(f"Error occurred while saving model: {e}")
            raise e

    def simplified_model(self, world_size: int):
        """
        导出精简模型pt文件（如 fp16）用于部署。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        if LOCAL_RANK not in {-1, 0}:
            return

        LOGGER.info(f"start export simplified model...")

        # 获取要保存的模型
        model = self.get_model_instance(world_size)

        fp16_pt = self.config_manager.core["fp16_pt"]
        if fp16_pt:
            model = model.half()
            LOGGER.info("Simplified model converted to float16 (fp16) format.")

        # 将模型设置为评估模式
        model.eval()

        checkpoint = {
            'model_name': self.config_manager.model["name"],
            'model': model.state_dict(),
            "core_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.core.items()},
            "model_args": {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in self.config_manager.model.items()},
            "fp16": fp16_pt
        }

        torch.save(checkpoint, self.simplified_pt.as_posix())

    def get_model_instance(self, world_size: int) -> TTBaseModel:
        """
        获取用于保存的模型实例（EMA 或原始模型）。

        Args:
            world_size (int): 分布式训练中的进程数量。

        Returns:
            nn.Module: 模型实例。
        """
        return self.model.module if world_size > 1 else self.model

    def load_model_to_final_eval(self, world_size, checkpoint) -> None:
        """
        将 best.pt/last.pt 中的模型权重加载到当前网络。

        根据是否启用 DDP 自动决定加载 model 还是 model.module。

        Args:
            world_size (int): 分布式进程数，用于判断是否为 DDP 模式。
            checkpoint (dict): torch.load 返回的 checkpoint，必须包含 key="model"。

        Returns:
            None
        """
        if world_size > 1:
            self.model.module.load_model_state_dict(checkpoint["model"])
        else:
            self.model.load_model_state_dict(checkpoint["model"])

    # ------------------------------------------------------------------
    # 5. 验证/测试
    # ------------------------------------------------------------------
    def get_validator(self, world_size: int):
        """
        获取验证器实例。

        Args:
            world_size (int): 分布式训练中的进程数量。

        Returns:
            TTBaseValidator: 验证器实例。
        """

        validator_cls = TTEngineRegistry.get(self.config_manager, "validator")
        return validator_cls(self, world_size) if validator_cls else None

    def do_validate(self) -> float:
        """
        执行模型验证并返回 fitness 值。

        Returns:
            float: fitness 值。
        """
        return self.validator.validate() if self.validator else 0.

    # ------------------------------------------------------------------
    # 6. 分布式环境管理
    # ------------------------------------------------------------------
    @staticmethod
    def set_ddp():
        """
        初始化分布式训练环境（DDP）。

        Args:
            world_size (int): 分布式训练中的进程数量。
        """
        from torch.distributed import init_process_group

        init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=3600),  # 1 hours
            # rank=RANK,
            # world_size=WORLD_SIZE,
            # device_id=self.device,
        )

    @staticmethod
    def destroy_ddp():
        """
        销毁分布式训练环境（DDP）。
        """
        from torch.distributed import destroy_process_group
        destroy_process_group()

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

    # ------------------------------------------------------------------
    # 7. 内存与设备工具
    # ------------------------------------------------------------------
    def _get_memory(self) -> float:
        """
        获取当前设备显存/内存使用量。

        Returns:
            float: 使用量（单位：GB）。
        """
        if self.device.type == "cpu":
            memory = 0
        elif self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cuda":
            memory = torch.cuda.memory_reserved()
        else:
            raise NotImplementedError("not support device type: {}".format(self.device.type))
        return memory / 1e9

    def _clear_memory(self):
        """
        清理设备内存/显存。
        """
        gc.collect()
        if self.device.type == "cpu":
            pass
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError("not support device type: {}".format(self.device.type))

    def _should_pin_memory(self, world_size: int, fallback: bool = True) -> bool:
        """
        智能决定是否启用 pin_memory。
        规则：
        1. CPU 训练 -> False
        2. 没有 GPU -> False
        3. 容器/节点可用内存 < 安全阈值 -> False
        4. 其它 -> 按 fallback（默认 True）
        """
        if self.device.type == "cpu":
            return False
        elif self.device.type == "mps":
            return False
        elif self.device.type == "cuda":
            # 保守阈值：每 GPU 预留 1 GiB 给 pinned memory
            # 可根据实际调大/调小
            safety_per_gpu = 1024 ** 3  # 1 GiB
            needed = safety_per_gpu * world_size
            avail = psutil.virtual_memory().available
            if avail < needed:
                print(f"[WARN] available RAM {avail / 1024 ** 3:.1f} GiB < "
                      f"needed {needed / 1024 ** 3:.1f} GiB, disable pin_memory.")
                return False
            return fallback
        else:
            raise NotImplementedError("not support device type: {}".format(self.device.type))

    # ------------------------------------------------------------------
    # 8. 优雅退出
    # ------------------------------------------------------------------
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
