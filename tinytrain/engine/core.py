from __future__ import annotations

import inspect
import os
import setproctitle
import torch

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Any, Dict

from tinytrain.cfg import TTEngineRegistry, TTConfigManager
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback
from tinytrain.utils.checks import check_file
from tinytrain.global_var import NUM_THREADS, LOCAL_RANK, WORLD_SIZE
from ..utils.dist import DDPLauncher

if TYPE_CHECKING:
    from .model import TTBaseModel


class Core:
    """
    Core 是 TinyTrain 的核心门面类，负责把「配置、模型、训练器、推理器、导出器」
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
    def __init__(self, link_file: str | Path, callback: Callback = None, *args, **kwargs) -> None:
        """
        初始化 Core，加载 link 配置并注册各 engine 的占位符。

        Args:
            link_file (str | Path, optional): link 配置文件路径（yaml / toml）。
        """
        # register manager
        self.config_manager = TTConfigManager(link_file=link_file)
        self.task: str = self.config_manager.core["task"]

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
        frame = inspect.stack()[-1]
        self.main_script_path = Path(frame.filename).resolve()

    @classmethod
    def register_components(cls):
        """
        类级钩子：一次性地把该 Core 所支持的全部 (task, engine_type, backend) → 实现类的映射注册到 TTEngineRegistry。

        任何继承自 Core 的子类 **必须** 实现此方法，否则在基类里会抛出NotImplementedError。

        示例：
        >>> class MyCore(Core):
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
        if link_type == 'link':
            raise KeyError("The 'set_config_overrides' function does not support setting link files. Please set them manually.")
        try:
            config = getattr(self.config_manager, link_type)
            update_dict = {}
            for k, v in kwargs.items():
                if k in config:
                    update_dict[k] = v
                else:
                    LOGGER.warning(f"{k} is not in {link_type} config. Please use this key with caution!")
            config = {**config, **update_dict}
            setattr(self.config_manager, link_type, config)
        except AttributeError:
            raise AttributeError(f"Config type '{link_type}' is not supported.")

    # ------------------------------------------------------------------
    # 3. 对外主要 API
    # ------------------------------------------------------------------
    def train(self, model_scale: str = None, model: str | Path = None, use_last_pt=False, process_name: str = None, **kwargs) -> None:
        """
        启动训练。

        Args:
            model_scale (str | None):
                模型规模标识（如 "n", "s", "m", "l", "x"），将覆盖配置文件。
            model (str | Path | None):
                权重文件路径（.pt/.pth）。为 None 时根据 use_last_pt 自动搜索。
            use_last_pt (bool):
                是否自动寻找最新的 last.pt，优先级低于显式 model。
            process_name (str | None):
                进程名，便于在系统监控中区分。
        """
        # 指定设备
        if self.device is None:
            self._set_device()

        # 修改进程名，从而避免与其他脚本混淆
        if process_name:
            setproctitle.setproctitle(process_name)

        nproc_per_node = self._get_nproc_per_node()
        # 判断是否已由 torchrun 启动（DDP 子进程）
        if "LOCAL_RANK" in os.environ:
            LOGGER.info("Detected DDP environment (torchrun), skipping subprocess launch.")
            self._launch_training(model_scale, model, use_last_pt)
        elif nproc_per_node > 1:
            self._launch_ddp(nproc_per_node)
        else:
            self._launch_training(model_scale, model, use_last_pt)

        # import gc
        # LOGGER.info("Training completed. Waiting for garbage collection...")
        # gc.collect()

    def predict(self, source, model: str | Path | None = None, backend: str | None = None, use_best_pt=False, **kwargs) -> Generator[Any, None, None]:
        """
        启动推理。

        Args:
            source: 输入源（路径、URL、摄像头索引等）。
            model (str | Path | None): 权重或后端文件路径。
            backend (str | None): 后端名称（onnx / tensorrt / torchscript ...）。
            use_best_pt (bool): 是否自动寻找 best.pt。
            **kwargs: 透传给 predictor。

        Returns:
            Generator[Any, None, None]: 推理结果生成器。
        """
        # 指定设备
        if self.device is None:
            self._set_device()

        # find best.pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
        if self.predictor is None:
            self._bind_predictor(model, backend, **kwargs)

        yield from self.predictor.predict(source)

    def __call__(self, source, model: str | Path | None = None, **kwargs) -> Generator[Any, None, None]:
        """
        允许 Core 实例直接当函数用：core(source) 等价于 predict(source)。
        """
        yield from self.predict(source, model, **kwargs)

    def export(self, backend: str, model: str | Path | None = None, export_dir=None, use_best_pt=False, **kwargs):
        """
        启动导出。

        Args:
            backend (str): 导出后端名称（onnx / tensorrt / torchscript ...）。
            model (str | Path | None): 权重文件路径。
            export_dir (str | Path | None): 导出目录，默认为配置中的 save_dir。
            use_best_pt (bool): 是否自动寻找 best.pt。
            **kwargs: 透传给 exporter。
        """
        # 指定设备
        if self.device is None:
            self._set_device()

        # find best.pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
        if self.exporter is None:
            self._bind_exporter(backend=backend, model=model, **kwargs)

        self.exporter.export(export_dir=export_dir)

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
        # 指定设备
        if self.device is None:
            self._set_device()

        if self.tuner is None:
            self._bind_tuner(model_scale)

        return self.tuner.tune(pop_size=pop_size, generations=generations)

    def distill(self, teacher_model: str | Path, student_model_scale: str, student_model: str | Path = None, process_name: str = None):
        """
        对外暴露的“一站式知识蒸馏”接口。

        参数：
            teacher_model: 教师权重文件路径（.pt/.pth）。
            student_model_scale: 学生模型规模（n/s/m/l/x），用于新建学生模型。
            student_model: 学生权重文件路径（可选）。若提供，则直接加载；否则按 scale 新建。
            process_name: 进程名，方便在 top / htop 中识别。

        执行流程：
        1. 若设备未初始化，先调用 _set_device()。
        2. 若指定进程名，调用 setproctitle 修改。
        3. 若 self.model 尚未绑定，调用 _bind_model() 加载或创建学生模型。
        4. 加载教师权重，重建教师模型实例（deepcopy 配置防止冲突）。
        5. 调用 _bind_distiller() 实例化蒸馏器。
        6. 启动蒸馏训练：self.distiller.train()。
        7. 训练结束手动触发 gc.collect()，释放显存与内存。

        异常：
        - 教师权重文件后缀非法将触发 AssertionError。
        - 蒸馏器未注册将触发 AttributeError。
        """
        # 指定设备
        if self.device is None:
            self._set_device()

        # 修改进程名，从而避免与其他脚本混淆
        if process_name:
            setproctitle.setproctitle(process_name)

        # bind student model
        if self.model is None:
            self._bind_model(student_model_scale, student_model)

        # bind distiller
        if self.distiller is None:
            # load teacher model
            assert Path(teacher_model).suffix in {".pt", ".pth"}, f"{Path(teacher_model).suffix} is not supported"
            teacher_model = check_file(teacher_model)
            checkpoint = torch.load(teacher_model.as_posix(), map_location="cpu", weights_only=False)
            teacher_config_manager = deepcopy(self.config_manager)
            teacher_config_manager.model = checkpoint["model_args"]
            teacher_model = TTEngineRegistry.get(teacher_config_manager, "teacher_model")(teacher_config_manager)
            teacher_model.load_model_state_dict(checkpoint["model"], True)

            self._bind_distiller(teacher_model)

        # train
        self.distiller.train()

        import gc
        LOGGER.info("Training completed. Waiting for garbage collection...")
        gc.collect()

    # ------------------------------------------------------------------
    # 4. 设备与 DDP 相关（内部工具）
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
        根据配置文件及当前硬件环境，为 Core 实例设置 self.device。

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

    def _launch_training(self, model_scale: str = None, model: str | Path = None, use_last_pt=False):
        """
        单进程 / DDP 子进程内部真正执行训练的入口函数。

        参数：
            model_scale: 模型规模（n/s/m/l/x），新建权重时使用。
            model: 权重文件路径（.pt/.pth），优先级高于 use_last_pt。
            use_last_pt: 当 model 为 None 时，是否自动寻找最近一次训练的 last.pt。

        步骤：
        1. 若 use_last_pt 为真，调用 _find_last_pt_file() 自动定位权重。
        2. 若 self.model 尚未实例化，调用 _bind_model() 加载或新建模型。
        3. 若 self.trainer 尚未实例化，调用 _bind_trainer() 绑定训练器。
        4. 执行 self.trainer.train() 开始训练迭代。

        注意：
        - 该函数由 train() 在“非 torchrun 启动”或“DDP 子进程”路径下调用。
        - 所有异常直接抛出，由外层 train() 统一捕获并记录。
        """

        # find last pt file
        if use_last_pt and model is None:
            model = self._find_last_pt_file()

        # bind model
        if self.model is None:
            self._bind_model(model_scale, model)

        # bind trainer
        if self.trainer is None:
            self._bind_trainer()

        # train
        self.trainer.train()

    # ------------------------------------------------------------------
    # 5. 模型绑定（内部工具）
    # ------------------------------------------------------------------
    def _bind_model(self, model_scale: str | None = None, model: str | Path = None, force_load=True) -> None:
        """
        根据权重或配置文件绑定模型。

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
                self.config_manager.core = checkpoint["core_args"]
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

    def _bind_predictor(self, model, backend: str | None = None, **kwargs):
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
            self._bind_model(model=model_path, force_load=False)
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

    def _bind_exporter(self, backend: str, model: str | Path | None = None, **kwargs):
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
            self._bind_model()
            LOGGER.warning("No model weights file provided. Exporting an untrained model.")
        else:
            model_path = Path(check_file(model))
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} does not exist")

            if model_path.suffix not in {".pt", ".pth"}:
                raise TypeError(f"export only supports '.pt' or '.pth' model files, got {model_path.suffix}")

            # 加载权重并绑定模型
            self._bind_model(model=model_path, force_load=False)

        self.exporter = TTEngineRegistry.get(self.config_manager, "exporter")(
            config_manager=self.config_manager,
            device=self.device,
            model=self.model,
            callback=self.callback,
            backend=backend,
            **kwargs
        )

    def _bind_tuner(self, model_scale: str | None = None) -> None:
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
        self.distiller = TTEngineRegistry.get(self.config_manager, "distiller")(
            config_manager=self.config_manager,
            device=self.device,
            student_model=self.model,
            teacher_model=teacher_model,
            callback=self.callback,
            main_script_path=self.main_script_path
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
                f"在 {task_dir} 及其子目录中均未找到可用的 last.pt"
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
