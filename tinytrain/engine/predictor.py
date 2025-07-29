from __future__ import annotations

import threading
import torch

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Generator, Union

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data import BaseDataInfo
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import create_iter_directory
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from torch import nn
    from tinytrain.server.inference_server import InferenceServerCore
    from tinytrain.utils.source_loader import SourceParser, SourceParserHub


class BasePredictor:
    def __init__(self,
                 config_manager: ConfigManager,
                 model,
                 callback: Callback,
                 backend: str | None = None,
                 max_qsize: int = 64,
                 max_workers: int = 4,
                 batch_size: int = 1,
                 **kwargs
                 ):
        self.config_manager = config_manager
        self.backend = backend

        # device
        from tinytrain.utils.checks import check_device_mini
        self.device = check_device_mini(self.config_manager.core["device"])

        # model
        self.model = self._setup_inference_server(model, **kwargs)

        # callback
        self.callback = callback

        # register parser
        self.register_parsers()

        # source queue
        self.source = None
        self.data_stream: "Queue[Any]" = Queue(max_qsize)
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # result
        self.preprocess_result = None
        self.inference_result = None
        self.postprocess_result = None

        # save dir
        save_dir = Path(config_manager.core["save_dir"]).resolve()
        project_name = config_manager.core["project_name"] or "default_project"
        save_dir = save_dir / project_name
        self.output_dir = create_iter_directory(save_dir, start_string="predict_")

    def predict(self, source) -> Generator[Any, None, None]:
        from tinytrain.utils.source_loader import SourceParserHub

        self.callback.run_callback(self, "on_predict_start")
        try:
            # 1. 选择解析器并启动线程
            parser = SourceParserHub.auto(source)
            threading.Thread(target=self._produce, args=(parser, source), daemon=True).start()

            # 2. 消费队列
            with torch.inference_mode():
                yield from self._consume()
        finally:
            self.callback.run_callback(self, "on_predict_end")

    def _produce(self, parser: SourceParser, source):
        try:
            for item in parser.stream(source):
                # 🔒 强制类型检查
                if item is not None and not isinstance(item, BaseDataInfo):
                    raise TypeError(
                        f"SourceParser 返回的数据必须是 BaseDataInfo 子类，实际收到 {type(item)}"
                    )
                self.data_stream.put(item)
        except Exception as e:
            LOGGER.exception(e)
            self.data_stream.put(None)

    def _consume(self) -> Generator[Any, None, None]:
        """消费者：批量或逐条推理"""
        batch = []
        while True:
            item = self.data_stream.get()
            if item is None:
                if batch:  # 处理剩余
                    yield from self._infer_batch(batch)
                break
            batch.append(item)
            if len(batch) == self.batch_size:
                yield from self._infer_batch(batch)
                batch.clear()

    def _infer_batch(self, batch: list[Any]) -> Generator[Any, None, None]:
        """通用推理：可被子类覆盖做真正的 batch / async"""
        for data_info in batch:
            self.callback.run_callback(self, "on_predict_batch_start")
            self.preprocess_result = self.preprocess(data_info)
            self.callback.run_callback(self, "on_predict_preprocess_end")
            self.inference_result = self.model.inference(self.preprocess_result)
            self.callback.run_callback(self, "on_predict_inference_end")
            self.postprocess_result = self.postprocess(data_info, self.inference_result)
            self.callback.run_callback(self, "on_predict_batch_end")
            yield self.show(data_info, self.postprocess_result)

    def register_parsers(self) -> None:
        # 注册解析器
        pass

    def preprocess(self, data_info: BaseDataInfo) -> Any:
        return data_info

    def postprocess(self, data_info: BaseDataInfo, inference_result: list[torch.Tensor]):
        return inference_result[0]

    def show(self, data_info: BaseDataInfo, postprocess_result) -> Any:
        return data_info

    def _setup_inference_server(self, model: Union[nn.Module, str, Path], **kwargs) -> Union[nn.Module, InferenceServerCore]:
        from torch import nn
        from tinytrain.server.inference_server import InferenceServerCore

        if isinstance(model, nn.Module):
            model.to(self.device)
            model.eval()
            return model
        elif isinstance(model, (str, Path)):
            return InferenceServerCore(config_manager=self.config_manager, model_file=model, device=self.device, backend=self.backend, **kwargs)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
