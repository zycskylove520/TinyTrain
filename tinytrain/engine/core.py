from __future__ import annotations

import inspect
import setproctitle

from pathlib import Path
from typing import TYPE_CHECKING, Generator, Any

from tinytrain.utils.register import TTRegistry
from tinytrain.global_var import DEFAULT_CORE_CONFIG_FILE
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback
from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.checks import check_file

if TYPE_CHECKING:
    from .model import BaseModel


class Core:
    """
    Core类用于集成所有的engine
    """

    def __init__(self, link_file: str | Path = DEFAULT_CORE_CONFIG_FILE):
        """
        初始化所有Core需要的附加engine和manager
        :param link_file: link配置文件,支持yaml格式和toml格式
        """
        self.task: str | None = None

        # register callback
        self.callbacks = Callback()

        # register engine
        self.model: BaseModel | None = None
        self.trainer = None
        self.predictor = None
        self.exporter = None
        self.tuner = None

        # register manager
        self.config_manager = ConfigManager(link_file=link_file)

        # 保存当前主脚本路径（用于 DDP 启动）
        frame = inspect.stack()[-1]
        self.main_script_path = Path(frame.filename).resolve()

    def __call__(self, source, model: str | Path | None = None, **kwargs) -> Generator[Any, None, None]:
        yield from self.predict(source, model, **kwargs)

    def set_config_overrides(self, link_type: str = 'core', **kwargs):
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
        调用该函数将启动模型的训练。
        @param model_scale:
        @param model:
        @param use_last_pt:
        @param process_name:
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
        # find last pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
        self._bind_predictor(model, backend, **kwargs)

        # predict
        yield from self.predictor.predict(source)

    def export(self, backend: str, model: str | Path | None = None, export_dir=None, use_best_pt=False, **kwargs):
        # find last pt file
        if use_best_pt and model is None:
            model = self._find_best_pt_file()

        # bind predictor
        self._bind_exporter(backend=backend, model=model, **kwargs)

        self.exporter.export(export_dir=export_dir)

    def _bind_model(self, model_scale: str | None = None, model: str | Path = None, force_load=True) -> None:
        """
        绑定模型，支持toml类型的模型配置文件和pt/pth类型的模型参数文件
        @param model_scale: 模型的大小。比如"n","s"等，由model的toml文件里指定的scales决定，这里可进行覆盖。
        @return:
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
            self.task = self.config_manager.core["task"]
            self.model = TTRegistry.get(self.task, "model")(self.config_manager)
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
                raise KeyError(f"{self.config_manager.link["model"]} not support scale:{self.config_manager.model["scale"]}")

            self.task = self.config_manager.core["task"]
            self.model = TTRegistry.get(self.task, "model")(self.config_manager)

    def _bind_trainer(self) -> None:
        """
        为Core绑定trainer，以便于进行模型训练。
        @return:
        """
        self.trainer = TTRegistry.get(self.task, "trainer")(
            config_manager=self.config_manager,
            model=self.model,
            callback=self.callbacks,
            main_script_path=self.main_script_path
        )

    def _bind_predictor(self, model, backend: str | None = None, **kwargs):
        """
       为 Core 绑定 predictor。
       支持三种场景：
       1. 直接加载 .pt/.pth 后预测；
       2. 传入非权重文件（如 onnx）后预测；
       3. 训练完直接预测（self.model 已存在）。
       """
        # 场景 1：训练完直接预测
        if self.model is not None:
            self.predictor = TTRegistry.get(self.task, "predictor")(
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
            self.predictor = TTRegistry.get(self.task, "predictor")(
                config_manager=self.config_manager,
                model=self.model,
                callback=self.callbacks,
                **kwargs
            )
            return

        # 场景 3：直接推理后端文件（onnx、engine、tensorrt...）
        if backend is not None:
            self.task = self.config_manager.core["task"]
            self.predictor = TTRegistry.get(self.task, "predictor")(
                config_manager=self.config_manager,
                model=model_path,  # 直接把文件路径传进去
                callback=self.callbacks,
                backend=backend,
                **kwargs
            )

    def _bind_exporter(self, backend: str, model: str | Path | None = None, **kwargs):
        """
        为 Core 绑定 exporter。
        支持两种场景：
        1. 训练完直接导出；
        2. 加载 .pt/.pth 后导出。
        """
        # 场景 1：训练完直接导出
        if self.model is not None:
            self.exporter = TTRegistry.get(self.task, "exporter")(
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

        self.exporter = TTRegistry.get(self.task, "exporter")(
            config_manager=self.config_manager,
            model=self.model,
            callback=self.callbacks,
            backend=backend,
            **kwargs
        )

    def _find_last_pt_file(self):
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