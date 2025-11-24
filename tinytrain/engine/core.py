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

import inspect
import os
import sys

import setproctitle
import torch

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Any, Dict

from tinytrain.cfg import TTConfigManager
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback
from tinytrain.utils.checks import check_file
from tinytrain.global_var import NUM_THREADS, LOCAL_RANK, WORLD_SIZE
from tinytrain import TTEngineRegistry, TTModuleRegistry
from tinytrain.utils.dist import DDPLauncher

if TYPE_CHECKING:
    from .model import TTBaseModel


class TTBaseCore:
    """
    TTBaseCore 是 TinyTrain 的核心门面类，负责把「配置、模型、训练器、推理器、导出器」
    等所有 engine 统一调度起来，对外暴露简洁的 train / predict / export 等接口。

    主要功能：
    1. 统一管理 TTConfigManager，支持链式配置（link 文件）。
    2. 根据场景自动绑定并实例化：
       - 训练：TTBaseTrainer
       - 推理：TTBasePredictor
       - 导出：TTBaseExporter
    3. 自动搜索 last.pt / best.pt 等权重文件。
    4. 支持进程名修改、回调钩子、DDP 启动路径保存等辅助特性。
    """

    # ------------------------------------------------------------------
    # 1. 构造与类级钩子
    # ------------------------------------------------------------------
    def __init__(self, link_file: str | Path, callback: Callback = None) -> None:
        """
        初始化 TTBaseCore，加载 link 配置并注册各 engine 的占位符。

        Args:
            link_file (str | Path, optional): link 配置文件路径（yaml / toml）。
        """
        # register manager
        self.config_manager = TTConfigManager(link_file=link_file)
        self.task: str = self.config_manager.core["task"]

        # register AI modules
        TTModuleRegistry.launch()

        # register components
        self.register_components()

        # register callback
        self.callback = callback if callback else Callback()

        # register engine
        self.device: torch.device | None = None
        self.model: TTBaseModel | None = None
        self.trainer = None
        self.predictor = None
        self.exporter = None
        self.tuner = None
        self.distiller = None

        # 保存当前主脚本路径（用于 DDP 启动）
        self.main_script_path = Path(inspect.stack()[-1].filename).resolve()

    @classmethod
    def register_components(cls):
        """
        类级钩子：一次性地把该 TTBaseCore 所支持的全部 (task, engine_type, backend) → 实现类的映射注册到 TTEngineRegistry。

        任何继承自 TTBaseCore 的子类 **必须** 实现此方法，否则在基类里会抛出NotImplementedError。

        示例：
        >>> class MyCore(TTBaseCore):
        ...     @classmethod
        ...     def register_components(cls):
        ...         TTEngineRegistry.register(cls, "classify", "model", MyClassificationModel)
        ...         TTEngineRegistry.register(cls, "detect", "model", MyDetectionModel)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 2. 运行时配置覆盖
    # ------------------------------------------------------------------
    def set_config_overrides(self, link_type: str = 'core', **kwargs):
        """
        在运行时动态覆盖指定配置段（link 除外）。

        Args:
            link_type (str): 配置段名称，如 "core"、"model"、"dataset" 等。
            **kwargs: 需要覆盖的键值对。

        Raises:
            KeyError: 试图覆盖 link 段。
            AttributeError: 不存在的配置段。
        """
        if link_type in {'link', "register_name"}:
            raise KeyError(f"The 'set_config_overrides' function does not support setting link_type: {link_type}. Please set them manually.")
        try:
            config = getattr(self.config_manager, link_type)
            update_dict = {}
            for k, v in kwargs.items():
                if k in config:
                    update_dict[k] = v
                else:
                    LOGGER.warning(f"{k} is not in {link_type} config. This parameter will be ignored!")
            config = {**config, **update_dict}
            setattr(self.config_manager, link_type, config)
        except AttributeError:
            raise AttributeError(f"Config type '{link_type}' is not supported.")

    @staticmethod
    def exclude_from_resume():
        """
        定义在恢复训练时需要排除的配置项列表。

        当从检查点恢复训练时，某些配置项不应该从保存的检查点中恢复，
        而是使用当前运行时的新配置值。这些配置项通常包括：
        - 训练相关的超参数（如batch_size、学习率策略等）
        - 设备相关的设置
        - 分布式训练配置
        - 其他可能影响训练稳定性的设置

        Returns:
            list[str]: 需要排除的配置项名称列表
        """
        exclude_list = [
            "launch_tb",
            "amp",
            "batch_size",
            "workers",
            "device",
            "accumulate",
            "patience",
            "save_period",
            "time",
            "seed",
            "deterministic",
            "ema",
            "grad_clip",
            "bf16",
            "fp16_pt",
            "l1_norm",

            # DDP
            "use_torchrun",
            "master_addr",
            "master_port",
            "nnodes",
            "node_rank",
        ]
        return exclude_list

    # ------------------------------------------------------------------
    # 3. 对外主要 API
    # ------------------------------------------------------------------
    def train(self, model_scale: str = None, model: str | Path = None, use_last_pt=False, use_best_pt=False, process_name: str = "defalut_train") -> None:
        """
        启动模型训练流程。

        Args:
            model_scale (str | None):
                模型规模标识，用于从配置文件中选择特定规模的模型架构。
                支持的值包括 "n", "s", "m", "l", "x" 等，具体取决于模型配置。
                如果为 None，则使用配置文件中定义的默认规模。

            model (str | Path | None):
                预训练权重文件路径（.pt 或 .pth 格式）。提供时将加载该权重继续训练。
                如果为 None，系统将根据 use_last_pt 或 use_best_pt 参数自动搜索权重文件。
                显式指定 model 参数时，model_scale、 use_last_pt 和 use_best_pt 参数将被忽略。

            use_last_pt (bool):
                是否自动搜索最近一次训练的 last.pt 权重文件继续训练。
                此参数仅在 model 为 None 时生效，且与 use_best_pt 互斥。

            use_best_pt (bool):
                是否自动搜索性能最佳的 best.pt 权重文件继续训练。
                此参数仅在 model 为 None 时生效，且与 use_last_pt 互斥。

            process_name (str):
                进程名称标识，用于在系统监控工具（如 htop、ps）中区分训练进程。
                默认为 "default_train"。

        Note:
            - 权重文件搜索优先级：显式 model 参数 > use_last_pt > use_best_pt
            - use_last_pt 和 use_best_pt 不能同时为 True
            - 训练过程支持单机单卡和多卡分布式训练（DDP），系统会自动检测并配置

        Raises:
            ValueError: 当 use_last_pt 和 use_best_pt 同时为 True 时抛出
            FileNotFoundError: 当指定的权重文件不存在时抛出
        """
        # 解析命令行参数
        self._auto_command_parser()

        # 单机或DDP执行训练
        if self._setup_ddp(process_name=process_name):
            # 根据优先级自动选择权重文件
            if model is None:
                model = self._find_pt_file(use_last_pt=use_last_pt, use_best_pt=use_best_pt)

            # bind model
            if self.model is None:
                self.get_model(model_scale, model)

            # bind trainer
            if self.trainer is None:
                self._bind_trainer()

            # train
            self.trainer.train()

    def predict(self, source, model: str | Path = None, backend: str = None, use_last_pt=False, use_best_pt=False, **kwargs) -> Generator[Any, None, None]:
        """
        启动推理。

        Args:
            source: 输入源（路径、URL、摄像头索引等）。
            model (str | Path | None): 权重或后端文件路径。
            backend (str | None): 后端名称（onnx / tensorrt / torchscript ...）。
            use_last_pt (bool): 当 model 为 None 时，是否自动寻找最近一次训练的 last.pt。
            use_best_pt (bool): 当 model 为 None 时，是否自动寻找最近一次训练的 best.pt。
            **kwargs: 透传给 predictor。

        Returns:
            Generator[Any, None, None]: 推理结果生成器。
        """
        # 解析命令行参数
        self._auto_command_parser()

        # 指定设备
        if self.device is None:
            self._set_device()

        # find best.pt file
        if model is None:
            model = self._find_pt_file(use_best_pt=use_best_pt, use_last_pt=use_last_pt)

        # bind predictor
        if self.predictor is None:
            self._bind_predictor(model, backend, **kwargs)

        yield from self.predictor.predict(source)

    def __call__(self, source, model: str | Path = None, backend: str = None, use_last_pt=False, use_best_pt=False, **kwargs) -> Generator[Any, None, None]:
        """
        允许 TTBaseCore 实例直接当函数用：core(source) 等价于 predict(source)。
        """
        yield from self.predict(source, model, backend, use_last_pt, use_best_pt, **kwargs)

    def export(self, backend: str, model: str | Path = None, use_last_pt=False, use_best_pt=False, **kwargs):
        """
        启动导出。

        Args:
            backend (str): 导出后端名称（onnx / tensorrt / torchscript ...）。
            model (str | Path | None): 权重文件路径。
            use_last_pt (bool): 当 model 为 None 时，是否自动寻找最近一次训练的 last.pt。
            use_best_pt (bool): 当 model 为 None 时，是否自动寻找最近一次训练的 best.pt。
            **kwargs: 透传给 exporter。
        """
        # 解析命令行参数
        self._auto_command_parser()

        # 指定设备
        if self.device is None:
            self._set_device()

        # find best.pt file
        if model is None:
            model = self._find_pt_file(use_best_pt=use_best_pt, use_last_pt=use_last_pt)

        # bind predictor
        if self.exporter is None:
            self._bind_exporter(backend=backend, model=model, **kwargs)

        self.exporter.export()

    def tune(self, model_scale: str = None, pop_size=40, generations=20) -> Dict[str, Any]:
        """
        启动遗传算法超参数调优，启动 GA 搜索并返回完整结果。
        使用注意：
            link.toml文件内的所有的配置文件路径必须使用完整的绝对路径！

        Args:
            model_scale (str | None, optional):
                模型规模标识，如 'n', 's', 'm', 'l', 'x'。
                传入后将覆盖配置文件中的默认值。若留空则使用配置值。
            pop_size (int, optional):
                遗传算法种群大小，默认 40。
            generations (int, optional):
                遗传算法迭代代数，默认 20。

        Returns:
            Dict[str, Any]:
                {
                    "history": pandas.DataFrame,  # 每代个体与适应度
                    "best_config": dict,          # 最优超参数组合
                }

        Raises:
            ValueError: 若任务未在注册表中注册对应 Tuner。
        """
        # 解析命令行参数
        self._auto_command_parser()

        # 指定设备
        if self.device is None:
            self._set_device()

        if self.tuner is None:
            self._bind_tuner(model_scale)

        return self.tuner.tune(pop_size=pop_size, generations=generations)

    def distill(self, teacher_model: str | Path | torch.nn.Module | TTBaseModel, student_model_scale: str = None, student_model: str | Path = None, process_name: str = "defalut_distill"):
        """
        执行知识蒸馏训练，将教师模型的知识迁移到学生模型。

        Args:
            teacher_model (str | Path | torch.nn.Module | TTBaseModel):
                教师模型定义，支持多种形式：
                - 权重文件路径（.pt/.pth）：加载预训练的教师模型
                - ONNX模型路径（.onnx）：直接使用ONNX模型作为教师
                - 模型实例：已初始化的PyTorch模型或TTBaseModel实例

            student_model_scale (str):
                学生模型规模标识，用于从配置文件中选择特定规模的学生模型架构。
                支持的值包括 "n", "s", "m", "l", "x" 等，具体取决于模型配置。

            student_model (str | Path | None):
                学生模型预训练权重文件路径（.pt 或 .pth 格式）。
                如果为 None，将根据 student_model_scale 创建新的学生模型。
                如果提供路径，将加载该权重初始化学生模型。

            process_name (str):
                进程名称标识，用于在系统监控工具中区分蒸馏训练进程。
                默认为 "default_distill"。

        Supported Teacher Model Types:
        - PyTorch模型文件 (.pt, .pth)
        - ONNX模型文件 (.onnx)
        - 已加载的PyTorch模型实例
        - TTBaseModel子类实例

        Raises:
            TypeError: 当教师模型类型不支持或文件格式不匹配时抛出
            FileNotFoundError: 当指定的模型文件不存在时抛出
            AttributeError: 当蒸馏器未在注册表中注册时抛出
        """
        # 解析命令行参数
        self._auto_command_parser()

        if self._setup_ddp(process_name=process_name):
            # bind student model
            if self.model is None:
                self.get_model(student_model_scale, student_model)

            # bind teacher model
            if isinstance(teacher_model, (str, Path)):  # str|Path:
                teacher_model = check_file(teacher_model)

                # 加载和学生模型同款模型，但不同尺寸
                if Path(teacher_model).suffix in {".pt", ".pth"}:
                    checkpoint = torch.load(teacher_model.as_posix(), map_location="cpu", weights_only=False)
                    teacher_cfg = deepcopy(self.config_manager)
                    teacher_cfg.model = checkpoint["model_args"]
                    teacher_model_ins = TTEngineRegistry.get(teacher_cfg, "model")(teacher_cfg, self.device)
                    teacher_model_ins.load_model_state_dict(checkpoint["model"], force_load=True)
                elif Path(teacher_model).suffix in {".onnx"}:
                    import onnxruntime as ort
                    teacher_model_ins = ort.InferenceSession(teacher_model)
                else:
                    raise TypeError(f"{Path(teacher_model).suffix} is not supported")
            elif isinstance(student_model, (TTBaseModel, torch.nn.Module)):  # model instance
                teacher_model_ins = teacher_model
            else:
                raise TypeError(f"Not supported teacher model!")

            # bind distiller
            if self.distiller is None:
                self._bind_distiller(teacher_model_ins)

            # distill
            self.distiller.distill()

    # ------------------------------------------------------------------
    # 4. 模型获取
    # ------------------------------------------------------------------
    def get_model(self, model_scale: str = None, model: str | Path = None, force_load=True) -> TTBaseModel:
        """
        根据权重或配置文件获取模型。

        Args:
            model_scale (str | None): 规模标识，仅新建模型时生效。
            model (str | Path | None): 权重文件（.pt/.pth）。
            force_load (bool): 是否强制形状匹配。
        """
        # 提前获取resume，防止加载参数后被修改
        resume = self.config_manager.core["resume"]

        # load exist model
        if model is not None:
            assert Path(model).suffix in {".pt", ".pth"}, f"{Path(model).suffix} is not supported"

            model = check_file(model)
            checkpoint = torch.load(model.as_posix(), map_location="cpu", weights_only=False)
            self.config_manager.link["model"] = Path(model)

            if resume:
                for k, v in checkpoint["core_args"].items():
                    if k not in self.exclude_from_resume():
                        self.config_manager.core[k] = v
                # self.config_manager.core = checkpoint["core_args"]
                # 要覆盖resume成指定的信息
                self.config_manager.core["resume"] = resume

            self.config_manager.model = checkpoint["model_args"]
            self.model = TTEngineRegistry.get(self.config_manager, "model")(self.config_manager, self.device)
            self.model.load_model_state_dict(checkpoint["model"], force_load)

            LOGGER.info(f"load pt model: {model}")
        # create new model
        else:
            if resume:
                raise KeyError("Error: Detected resume=True, but no valid .pt or .pth file was provided.")

            scales = list(self.config_manager.model["scales"].keys())  # 添加model的scale
            if not scales:
                raise KeyError("Error: Model.toml no scales were provided.")

            if model_scale:
                if model_scale not in scales:
                    raise KeyError(f"{self.config_manager.link['model']} not support scale:{self.config_manager.model['scale']}")

                self.config_manager.model["scale"] = model_scale
            else:
                self.config_manager.model["scale"] = scales[0]

            self.model = TTEngineRegistry.get(self.config_manager, "model")(self.config_manager, self.device)

        return self.model

    # ------------------------------------------------------------------
    # 5. 设备与 DDP 相关（内部工具）
    # ------------------------------------------------------------------
    def _check_device(self) -> torch.device:
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

    def _set_device(self):
        """
        根据配置文件及当前硬件环境，为 TTBaseCore 实例设置 self.device。

        步骤：
        1. 调用 _check_device() 获取最合适的 torch.device（cpu / cuda / mps）。
        2. 若使用多卡训练（WORLD_SIZE > 1），将当前进程绑到 LOCAL_RANK 对应的卡上，
           并更新 self.device 为 cuda:LOCAL_RANK。
        3. 日志仅在主进程（LOCAL_RANK 未设置）打印，避免 DDP 子进程重复输出。

        注意：
        - 本函数只会被显式调用一次，之后所有 engine 均复用 self.device。
        - 环境变量 CUDA_VISIBLE_DEVICES 已在 _check_device() 中按配置做好映射。
        """
        self.device = self._check_device()

        if self.device.type == "cuda" and WORLD_SIZE > 1:
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device("cuda", LOCAL_RANK)

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

    def _setup_ddp(self, process_name: str = None):
        # 指定设备
        if self.device is None:
            self._set_device()

        # 修改进程名，从而避免与其他脚本混淆
        if process_name:
            setproctitle.setproctitle(f"{process_name}-local_rank:[{LOCAL_RANK}]")

        nproc_per_node = self._get_nproc_per_node()
        # 智能设置线程数
        self._setup_optimal_threads(nproc_per_node)

        # 如果已经在DDP环境中，直接执行
        if self._is_ddp_environment():
            LOGGER.info(f"DDP environment detected - Local Rank: {LOCAL_RANK}, World Size: {WORLD_SIZE}")
            return True
        # 如果需要启动DDP
        elif nproc_per_node > 1:
            LOGGER.info(f"Launching DDP training with {nproc_per_node} GPUs")
            self._launch_ddp(nproc_per_node)
            return False  # 主进程不执行
        # 单机单卡
        else:
            LOGGER.info("Single GPU/CPU training mode")
            return True

    def _launch_ddp(self, nproc_per_node: int):
        """
        使用 torchrun 启动分布式训练。

        Args:
            nproc_per_node (int): 每个节点的 GPU 数量。
        """
        launcher = DDPLauncher(
            main_script=self.main_script_path,
            nproc_per_node=nproc_per_node,
            nnodes=self.config_manager.core["nnodes"],
            node_rank=self.config_manager.core["node_rank"],
            master_addr=self.config_manager.core["master_addr"],
            master_port=self.config_manager.core["master_port"],
        )
        try:
            LOGGER.info("Starting DDP training...")
            launcher.run(use_torchrun=self.config_manager.core["use_torchrun"])
        except Exception as e:
            LOGGER.error(f"DDP launch failed: {e}")
            raise
        finally:
            LOGGER.info("Releasing memory...")

    @staticmethod
    def _setup_optimal_threads(nproc_per_node: int):
        """
        根据系统资源设置最优线程数

        Args:
            nproc_per_node (int): 每个节点的 GPU 数量。
        """
        cpu_cores = os.cpu_count()

        if not cpu_cores:
            return

        if nproc_per_node > 1:
            # 分布式训练，每个进程分配的核心数
            cores_per_process = max(1, cpu_cores // nproc_per_node)
        else:
            cores_per_process = max(1, cpu_cores // 2)  # 单机使用一半核心

        os.environ['OMP_NUM_THREADS'] = str(cores_per_process)
        os.environ['MKL_NUM_THREADS'] = str(cores_per_process)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cores_per_process)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cores_per_process)

        return cores_per_process

    @staticmethod
    def _is_ddp_environment():
        """检查是否在DDP环境中"""
        return all(var in os.environ for var in ["LOCAL_RANK", "RANK", "WORLD_SIZE"])

    # ------------------------------------------------------------------
    # 6. 各引擎绑定（内部工具）
    # ------------------------------------------------------------------
    def _bind_trainer(self) -> None:
        """
        实例化并绑定 trainer。
        """
        self.trainer = TTEngineRegistry.get(self.config_manager, "trainer")(
            config_manager=self.config_manager,
            device=self.device,
            model=self.model,
            callback=self.callback
        )

    def _bind_predictor(self, model, backend: str = None, **kwargs):
        """
        实例化并绑定 predictor，支持三种场景：
        1. 训练完直接预测；
        2. 加载 .pt/.pth 后预测；
        3. 使用第三方后端文件直接推理。
        """

        # 场景 1：训练完直接预测
        if self.model is not None:
            self.predictor = TTEngineRegistry.get(self.config_manager, "predictor")(
                config_manager=self.config_manager,
                device=self.device,
                model=self.model,
                callback=self.callback,
                **kwargs
            )
            return

        # 场景 2 & 3：外部文件预测
        if model is None:
            raise ValueError("model path must be provided when no training model exists")

        model_path = Path(check_file(model))
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist")

        # 场景 2：权重文件 → 先绑定模型
        if model_path.suffix in {".pt", ".pth"}:
            self.get_model(model=model_path, force_load=False)
            self.predictor = TTEngineRegistry.get(self.config_manager, "predictor")(
                config_manager=self.config_manager,
                device=self.device,
                model=self.model,
                callback=self.callback,
                **kwargs
            )
            return

        # 场景 3：直接推理后端文件（onnx、engine、tensorrt...）
        if backend is not None:
            self.predictor = TTEngineRegistry.get(self.config_manager, "predictor")(
                config_manager=self.config_manager,
                device=self.device,
                model=model_path,  # 直接把文件路径传进去
                callback=self.callback,
                backend=backend,
                **kwargs
            )

    def _bind_exporter(self, backend: str, model: str | Path = None, **kwargs):
        """
        实例化并绑定 exporter，支持：
        1. 训练完直接导出；
        2. 加载 .pt/.pth 后导出。
        """

        # 场景 1：训练完直接导出
        if self.model is not None:
            self.exporter = TTEngineRegistry.get(self.config_manager, "exporter")(
                config_manager=self.config_manager,
                device=self.device,
                model=self.model,
                callback=self.callback,
                backend=backend,
                **kwargs
            )
            return

        # 场景 2：外部权重文件导出

        # 没有传递模型权重文件，此时导出将是未训练的模型
        if model is None:
            # 构建新模型，默认使用最小的scale
            self.get_model()
            LOGGER.warning("No model weights file provided. Exporting an untrained model.")
        else:
            model_path = Path(check_file(model))
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} does not exist")

            if model_path.suffix not in {".pt", ".pth"}:
                raise TypeError(f"export only supports '.pt' or '.pth' model files, got {model_path.suffix}")

            # 加载权重并绑定模型
            self.get_model(model=model_path, force_load=False)

        self.exporter = TTEngineRegistry.get(self.config_manager, "exporter")(
            config_manager=self.config_manager,
            device=self.device,
            model=self.model,
            callback=self.callback,
            backend=backend,
            **kwargs
        )

    def _bind_tuner(self, model_scale: str = None) -> None:
        """
        实例化并绑定 tuner。
        """
        self.tuner = TTEngineRegistry.get(self.config_manager, "tuner")(core=self, model_scale=model_scale)

    def _bind_distiller(self, teacher_model) -> None:
        """
        实例化知识蒸馏器（Distiller）并绑定到 self.distiller。

        参数：
            teacher_model: 已加载权重的教师模型实例（TTBaseModel）。

        注册表查询 key 为 "distiller"，传入下列参数：
        - config_manager: 全局配置
        - device: 当前进程设备
        - student_model: 已绑定到 self.model 的学生模型
        - teacher_model: 教师模型
        - callback: 回调钩子
        - main_script_path: 主脚本绝对路径，用于 DDP 重启

        注意：
        - 教师模型必须提前加载并传参，本函数不再负责权重加载。
        - 若注册表未找到对应蒸馏器，将抛出 AttributeError。
        """
        trainer = TTEngineRegistry.get(self.config_manager, "trainer")(
            config_manager=self.config_manager,
            device=self.device,
            model=self.model,
            callback=self.callback
        )

        self.distiller = TTEngineRegistry.get(self.config_manager, "distiller")(
            trainer=trainer,
            teacher_model=teacher_model
        )

    # ------------------------------------------------------------------
    # 7. 权重文件搜索（内部工具）
    # ------------------------------------------------------------------
    def _find_last_pt_file(self):
        """
        自动搜索最近一次训练生成的 last.pt。

        Returns:
            Path: last.pt 的绝对路径。

        Raises:
            FileNotFoundError: 如果任何 run 目录下都找不到 last.pt。
        """

        save_dir = Path(self.config_manager.core["save_dir"]).resolve()
        project_name = self.config_manager.core["project_name"]
        if project_name == "":
            project_name = self.config_manager.core["project_name"] = "default_project"

        task_dir = save_dir / project_name / self.config_manager.core["task"] / "train"
        if not task_dir.exists():
            raise FileNotFoundError(f"{task_dir} 不存在，无法加载 last.pt")
        # 按时间升序排列所有 run 目录（可根据需要改成按名称排序）
        run_dirs = sorted(
            [d for d in task_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,  # 也可以改为按目录名排序
            reverse=True  # 最新的在前
        )

        pt_model = None
        for run_dir in run_dirs:
            candidate = run_dir / "weights" / "last.pt"
            if candidate.exists() and candidate.is_file():
                pt_model = candidate
                break

        if pt_model is None:
            raise FileNotFoundError(
                f"No usable last.pt found in {task_dir} or its subdirectories"
            )
        return pt_model

    def _find_best_pt_file(self):
        """
        自动搜索最近一次训练生成的 best.pt（逻辑与 _find_last_pt_file 一致）。

        Returns:
            Path: best.pt 的绝对路径。

        Raises:
            FileNotFoundError: 如果任何 run 目录下都找不到 best.pt。
        """

        save_dir = Path(self.config_manager.core["save_dir"]).resolve()
        project_name = self.config_manager.core["project_name"]
        if project_name == "":
            project_name = self.config_manager.core["project_name"] = "default_project"

        task_dir = save_dir / project_name / self.config_manager.core["task"] / "train"
        if not task_dir.exists():
            raise FileNotFoundError(f"{task_dir} 不存在，无法加载 best.pt")
        # 按时间升序排列所有 run 目录（可根据需要改成按名称排序）
        run_dirs = sorted(
            [d for d in task_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,  # 也可以改为按目录名排序
            reverse=True  # 最新的在前
        )

        pt_model = None
        for run_dir in run_dirs:
            candidate = run_dir / "weights" / "best.pt"
            if candidate.exists() and candidate.is_file():
                pt_model = candidate
                break

        if pt_model is None:
            raise FileNotFoundError(
                f"在 {task_dir} 及其子目录中均未找到可用的 best.pt"
            )
        return pt_model

    def _find_pt_file(self, use_last_pt, use_best_pt):
        """
        根据优先级自动查找并返回权重文件路径。

        权重文件查找优先级：
        1. 如果 use_last_pt 为 True，查找最新的 last.pt 文件
        2. 如果 use_best_pt 为 True，查找最新的 best.pt 文件
        3. 如果都为 False，返回 None

        Args:
            use_last_pt (bool): 是否查找最近一次训练的 last.pt 文件
            use_best_pt (bool): 是否查找性能最好的 best.pt 文件

        Returns:
            Path | None: 找到的权重文件路径，如果未找到且不需要查找则返回 None

        Raises:
            ValueError: 当 use_last_pt 和 use_best_pt 同时为 True 时抛出
            FileNotFoundError: 当需要查找但找不到对应的权重文件时抛出

        注意:
            - use_last_pt 和 use_best_pt 是互斥的，不能同时为 True
            - 查找范围基于配置中的 save_dir、project_name 和 task 设置
            - 按目录修改时间倒序查找，返回找到的第一个有效文件
        """
        # 检查 use_last_pt 和 use_best_pt 互斥性
        if use_last_pt and use_best_pt:
            raise ValueError("参数 use_last_pt 和 use_best_pt 不能同时为 True")

        # 根据优先级自动选择权重文件
        if use_last_pt:
            return self._find_last_pt_file()
        elif use_best_pt:
            return self._find_best_pt_file()  # 假设有对应的查找最佳权重文件的方法
        else:
            return None

    # ------------------------------------------------------------------
    # 8. 命令行解析（内部工具）
    # ------------------------------------------------------------------
    def _auto_command_parser(self):
        """
        自动解析命令行参数并更新配置管理器中的参数覆盖。

        该方法解析命令行参数，支持以下格式：
        - 组指定：以单个 '-' 开头（如：-data）
        - 参数指定：以 '--' 开头（如：--device, --batch_size）
        - 参数值格式：
            - 键值对：--key=value
            - 空格分隔：--key value
            - 布尔标志：--flag（无值参数，自动设为 True）
            - 列表值：支持逗号分隔的值（如：--device=0,1,2,3）

        命令行参数结构示例：
            python script.py -core --device 0,1,2 --batch_size 32 -model --name MyNet

        解析流程：
        1. 检查帮助参数 --help，如果存在则显示帮助信息并退出
        2. 遍历命令行参数，识别组和参数
        3. 将参数按组分类存储
        4. 应用参数覆盖到配置管理器

        参数分组机制：
        - 每个参数必须属于一个组
        - 组以 '-' 开头（如：-data）
        - 参数以 '--' 开头（如：--device）
        - 参数会应用到其前面的组中

        异常处理：
        - 未知组名：抛出 ValueError
        - 未分组的参数：显示错误信息并退出
        - 无效参数：显示帮助信息并退出

        返回值：
            None

        异常：
            ValueError: 当遇到未知的组名或参数应用失败时抛出

        示例：
            >>> # 命令行：python script.py -core --device 0,1,2 --batch_size 32
            >>> # 结果：core 组的 device 参数设置为 [0, 1, 2]，batch_size 参数设置为 32

        注意：
            - 参数值会自动进行类型转换（int, float, bool, list, str）
            - 逗号分隔的值会自动转换为列表
            - 布尔标志参数只需指定参数名，值自动设为 True
        """
        # 检查是否是 --help
        if len(sys.argv) > 1 and sys.argv[1] == '--help':
            self._show_help()
            sys.exit(0)

        # 允许无参数
        if len(sys.argv) <= 1:
            return

        groups = {key: {} for key in self.config_manager.link.keys()}  # 空字典
        current_group = None

        i = 1  # 跳过脚本名
        while i < len(sys.argv):
            arg = sys.argv[i]

            # 检查是否是 --help（可能在中间位置）
            if arg == '--help':
                self._show_help()
                sys.exit(0)

            # 以 - 开头（但不是 --）的是组
            if arg.startswith('-') and not arg.startswith('--'):
                group_name = arg[1:]  # 去掉开头的 -
                current_group = group_name

                # 初始化组
                if group_name not in groups:
                    raise ValueError(f"Unknown group name: {group_name}, available groups: {list(groups.keys())}")

                i += 1
                continue

            # 以 -- 开头的是参数
            elif arg.startswith('--'):
                if current_group is None:
                    print(f"Error: Parameter '{arg}' is not preceded by a group specification!")
                    self._show_help()
                    sys.exit(1)

                # 处理 --key=value 格式
                if '=' in arg:
                    key_value = arg[2:]  # 去掉开头的 --
                    key, value = key_value.split('=', 1)
                    groups[current_group][key] = self._convert_value(value)
                    i += 1
                else:
                    # 处理 --key value 格式
                    key = arg[2:]  # 去掉开头的 --

                    # 检查下一个参数是否是值（不是组且不是参数）
                    if i + 1 < len(sys.argv):
                        next_arg = sys.argv[i + 1]
                        if not next_arg.startswith('-'):
                            value = next_arg
                            groups[current_group][key] = self._convert_value(value)
                            i += 2
                        else:
                            # 下一个参数是组或参数，当前参数作为布尔标志
                            groups[current_group][key] = True
                            i += 1
                    else:
                        # 最后一个参数，作为布尔标志
                        groups[current_group][key] = True
                        i += 1
            else:
                print(f"Error: Ungrouped parameter '{arg}'!")
                print("All parameters must belong to a group (starting with -)")
                self._show_help()
                sys.exit(1)

        # 应用参数覆盖到配置管理器
        for group_name, params in groups.items():
            if params:  # 只覆盖有参数的组
                try:
                    self.set_config_overrides(link_type=group_name, **params)
                    LOGGER.info(f"command override {group_name} group params: {params}")
                except Exception as e:
                    raise ValueError(f"command override {group_name} group params error: {e}")

        # 验证至少有一个组有参数（如果有命令行参数的话）
        has_any_params = any(params for params in groups.values())
        if len(sys.argv) > 1 and not has_any_params:
            print("Error: No valid group parameters provided in command line arguments!")
            self._show_help()
            sys.exit(1)

    @staticmethod
    def _convert_value(value):
        """
        转换参数值，支持将逗号分隔的字符串转换为列表

        Args:
            value: 原始字符串值

        Returns:
            转换后的值，可能是基本类型或列表
        """
        # 如果是逗号分隔的字符串，转换为列表
        if isinstance(value, str) and ',' in value:
            # 分割字符串并去除每个元素的首尾空格
            parts = [part.strip() for part in value.split(',')]

            # 尝试将每个部分转换为适当的数据类型
            converted_parts = []
            for part in parts:
                if part:
                    converted_parts.append(TTBaseCore._convert_single_value(part))

            return converted_parts

        # 单个值的正常转换
        return TTBaseCore._convert_single_value(value)

    @staticmethod
    def _convert_single_value(value):
        """
        转换单个参数值

        Args:
            value: 原始字符串值

        Returns:
            转换后的值（int, float, bool 或 str）
        """
        if not isinstance(value, str):
            return value

        # 布尔值转换
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 整数转换
        try:
            return int(value)
        except ValueError:
            pass

        # 浮点数转换
        try:
            return float(value)
        except ValueError:
            pass

        # 保持原字符串
        return value

    @staticmethod
    def _show_help():
        """显示帮助信息"""
        print("用法: python script.py [--help] [组] [参数] [组] [参数] ...")
        print("\n选项:")
        print("  --help                显示此帮助信息")
        print("\n规则:")
        print("  - 以 - 开头的为组（如: -core, -dataset）")
        print("  - 以 -- 开头的为参数（如: --project_name, --workers=4）")
        print("  - 布尔类型值支持: true/false、yes/no、on/off，（如： -core --ema false）")
        print("  - 参数必须跟在组后面")
        print("\n示例:")
        print("  python script.py --help")
        print("  python script.py -core --epochs 10 --batch_size 16 -dataset --cache")
        print("  python script.py -core --epochs=10 --batch_size=16 -dataset --cache=false")
        print("  python script.py                      # 无参数运行")
