from __future__ import annotations

import inspect
import setproctitle

from pathlib import Path
from typing import TYPE_CHECKING, Generator, Any, Dict

from tinytrain.cfg.TT_register import TTEngineRegistry
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback
from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.checks import check_file

if TYPE_CHECKING:
    from .model import BaseModel


class Core:
    """
    Core 是 TinyTrain 的核心门面类，负责把「配置、模型、训练器、推理器、导出器」
    等所有 engine 统一调度起来，对外暴露简洁的 train / predict / export 等接口。

    主要功能：
    1. 统一管理 ConfigManager，支持链式配置（link 文件）。
    2. 根据场景自动绑定并实例化：
       - 训练：BaseTrainer
       - 推理：BasePredictor
       - 导出：BaseExporter
    3. 自动搜索 last.pt / best.pt 等权重文件。
    4. 支持进程名修改、回调钩子、DDP 启动路径保存等辅助特性。
    """

    def __init__(self, link_file: str | Path):
        """
        初始化 Core，加载 link 配置并注册各 engine 的占位符。

        Args:
            link_file (str | Path, optional): link 配置文件路径（yaml / toml）。
        """
        # register manager and components
        self.config_manager = ConfigManager(link_file=link_file)
        # self.config_manager.register_name = self.__class__.__name__
        self.register_components()

        self.task: str | None = None

        # register callback
        self.callbacks = Callback()

        # register engine
        self.model: BaseModel | None = None
        self.trainer = None
        self.predictor = None
        self.exporter = None
        self.tuner = None

        # 保存当前主脚本路径（用于 DDP 启动）
        frame = inspect.stack()[-1]
        self.main_script_path = Path(frame.filename).resolve()

    # ------------------------------------------------------------------
    # 外部调用主要 API
    # ------------------------------------------------------------------
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

    def train(self, model_scale: str = None, model: str | Path = None, use_last_pt=False, process_name: str = None):
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

        # 修改进程名，从而避免与其他脚本混淆
        if process_name:
            setproctitle.setproctitle(process_name)

        # find last pt file
        if use_last_pt and model is None:
            model = self._find_last_pt_file()

        # bind model
        self._bind_model(model_scale, model)

        # bind trainer
        self._bind_trainer()

        # train
        self.trainer.train()

        import gc
        LOGGER.info("Training completed. Waiting for garbage collection...")
        gc.collect()

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
        # find best.pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
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

        # find best.pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
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

        self._bind_tuner(model_scale)
        return self.tuner.tune(pop_size=pop_size, generations=generations)

    # ------------------------------------------------------------------
    # 受保护函数
    # ------------------------------------------------------------------
    def _bind_model(self, model_scale: str | None = None, model: str | Path = None, force_load=True) -> None:
        """
        根据权重或配置文件绑定模型。

        Args:
            model_scale (str | None): 规模标识，仅新建模型时生效。
            model (str | Path | None): 权重文件（.pt/.pth）。
            force_load (bool): 是否强制形状匹配。
        """
        import torch

        # 提前获取resume，防止加载参数后被修改
        resume = self.config_manager.core["resume"]

        # load exist model
        if model is not None:
            assert Path(model).suffix in {".pt", ".pth"}, f"{Path(model).suffix} is not supported"

            model = check_file(model)
            checkpoint = torch.load(model.as_posix(), map_location="cpu", weights_only=False)
            if resume:
                self.config_manager.link["model"] = Path(model)
                self.config_manager.core = checkpoint["core_args"]
                # 要覆盖resume成指定的信息
                self.config_manager.core["resume"] = resume

            self.config_manager.model = checkpoint["model_args"]
            self.model = TTEngineRegistry.get(self.config_manager, "model")(self.config_manager)
            self.model.load_model_state_dict(checkpoint["model"], force_load)

            LOGGER.info(f"load pt model: {model}")
        # create new model
        else:
            if resume:
                raise KeyError("Error: Detected resume=True, but no valid .pt or .pth file was provided.")

            scales = self.config_manager.model["scales"].keys()  # 添加model的scale
            if model_scale:
                self.config_manager.model["scale"] = model_scale

            if self.config_manager.model["scale"] not in scales:
                raise KeyError(f"{self.config_manager.link['model']} not support scale:{self.config_manager.model['scale']}")

            self.model = TTEngineRegistry.get(self.config_manager, "model")(self.config_manager)

    def _bind_trainer(self) -> None:
        """
        实例化并绑定 trainer。
        """
        self.trainer = TTEngineRegistry.get(self.config_manager, "trainer")(
            config_manager=self.config_manager,
            model=self.model,
            callback=self.callbacks,
            main_script_path=self.main_script_path
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
                model=self.model,
                callback=self.callbacks,
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
                model=self.model,
                callback=self.callbacks,
                **kwargs
            )
            return

        # 场景 3：直接推理后端文件（onnx、engine、tensorrt...）
        if backend is not None:
            self.predictor = TTEngineRegistry.get(self.config_manager, "predictor")(
                config_manager=self.config_manager,
                model=model_path,  # 直接把文件路径传进去
                callback=self.callbacks,
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
                model=self.model,
                callback=self.callbacks,
                backend=backend,
                **kwargs
            )
            return

        # 场景 2：外部权重文件导出
        if model is None:
            raise ValueError("model path must be provided when no training model exists")

        model_path = Path(check_file(model))
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist")

        if model_path.suffix not in {".pt", ".pth"}:
            raise TypeError(f"export only supports '.pt' or '.pth' model files, got {model_path.suffix}")

        # 加载权重并绑定模型
        self._bind_model(model=model_path, force_load=False)

        self.exporter = TTEngineRegistry.get(self.config_manager, "exporter")(
            config_manager=self.config_manager,
            model=self.model,
            callback=self.callbacks,
            backend=backend,
            **kwargs
        )

    def _bind_tuner(self, model_scale: str | None = None) -> None:
        """
        实例化并绑定 tuner。
        """
        self.tuner = TTEngineRegistry.get(self.config_manager, "tuner")(core=self, model_scale=model_scale)

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

        task_dir = save_dir / project_name / self.config_manager.core["task"]
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

        task_dir = save_dir / project_name / self.config_manager.core["task"]
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
