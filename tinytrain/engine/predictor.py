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

import threading
import torch

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Generator, Union

from tinytrain.cfg import TTConfigManager, TTEngineRegistry
from tinytrain.data.data_format import BaseDataInfo
from tinytrain.global_var import TIMER_ENABLED
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import create_iter_directory
from tinytrain.utils.callback import Callback, Events
from tinytrain.utils.time_recorder import TimeRecorder

if TYPE_CHECKING:
    from torch import nn
    from tinytrain.server.inference_server import TTBaseInferenceServer
    from tinytrain.utils.source_loader import SourceParser, SourceParserHub


class TTBasePredictor:
    """
    TTBasePredictor 是一个通用推理基类，支持多种输入源（视频/图片/目录/摄像头/网络流等）
    的实时或批量推理。特性包括：
    - 支持本地 torch.nn.Module 或远程推理服务器（TTBaseInferenceServer）两种后端。
    - 内置线程安全的数据流队列，支持多线程异步生产-消费模式。
    - 支持批量推理与逐条推理，可配置 batch_size、最大队列长度、线程池大小。
    - 提供完整的生命周期钩子（on_predict_start、on_predict_batch_* 等）。
    - 所有输出统一以生成器形式返回，方便下游处理与流式传输。
    - 输出目录自动编号，避免覆盖历史结果。

    使用方式：
    1. 继承并重写 `preprocess`、`postprocess`、`show` 等钩子，实现自定义逻辑。
    2. 可选重写 `_infer_batch` 以支持真正的批量推理或异步调用。
    3. 调用 `predict(source)` 即可获得结果生成器。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self,
                 config_manager: TTConfigManager,
                 device: torch.device,
                 model,
                 callback: Callback,
                 backend: str | None = None,
                 max_qsize: int = 64,
                 max_workers: int = 4,
                 batch_size: int = 1,
                 **kwargs
                 ):
        """
        初始化推理器。

        Args:
            config_manager (TTConfigManager): 全局配置管理器。
            model (nn.Module | str | Path):
                - torch.nn.Module：本地模型实例。
                - str / Path：模型文件路径或远程推理服务配置，将使用 TTBaseInferenceServer。
            callback (Callback): 回调对象，用于插入生命周期钩子。
            backend (str | None, optional): 推理后端名称，仅对远程模型生效。默认 None。
            max_qsize (int, optional): 数据流队列最大长度。默认 64。
            max_workers (int, optional): 生产端线程池最大线程数。默认 4。
            batch_size (int, optional): 每批推理的样本数。默认 1。
            **kwargs: 透传给 TTBaseInferenceServer 的额外参数。
        """
        self.config_manager = config_manager
        self.backend = backend

        # device
        self.device = device

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
        self.output_dir = None

        # time recorder
        self.preprocess_recoder = TimeRecorder(self.device, "preprocess")
        self.inference_recoder = TimeRecorder(self.device, "inference")
        self.postprocess_recoder = TimeRecorder(self.device, "postprocess")

    # ------------------------------------------------------------------
    # 2. 唯一公开主链
    # ------------------------------------------------------------------
    def predict(self, source) -> Union[Generator[Any, None, None], list[Any]]:
        """
        启动推理流程，返回结果生成器。

        Args:
            source: 输入源，可为文件夹、视频、摄像头索引、URL 等，由 SourceParserHub 自动选择解析器。

        Returns:
            Generator[Any, None, None] | list[Any]: 推理结果生成器或列表。
        """
        from tinytrain.utils.source_loader import SourceParserHub

        core = self.config_manager.core
        save_dir = Path(core["save_dir"]).resolve() / (core["project_name"] or "default_project") / core["task"] / "predict"
        self.output_dir = create_iter_directory(save_dir, start_string="predict_")

        self.callback.run_callback(Events.ON_PREDICT_START ,self)
        try:
            # 1. 选择解析器并启动线程
            parser = SourceParserHub.auto(source)
            threading.Thread(target=self._produce, args=(parser, source), daemon=True).start()

            # 2. 消费队列
            yield from self._consume()
        finally:
            self.callback.run_callback(Events.ON_PREDICT_END, self)

        LOGGER.info(f"Predict result saved in directory -> {self.output_dir}")

    # ------------------------------------------------------------------
    # 3. 后端初始化（内部工具）
    # ------------------------------------------------------------------
    def _setup_inference_server(self, model: Union[nn.Module, str, Path], **kwargs) -> Union[nn.Module, TTBaseInferenceServer]:
        """
        根据输入类型初始化推理后端。

        Args:
            model (nn.Module | str | Path):
                - nn.Module：本地模型，直接加载到 self.device。
                - str / Path：模型文件路径或远程配置，交由 TTBaseInferenceServer 处理。
            **kwargs: 透传给 TTBaseInferenceServer 的额外参数。

        Returns:
            nn.Module | TTBaseInferenceServer: 已就绪的推理后端。

        Raises:
            TypeError: 不支持的模型类型。
        """
        from torch import nn

        if isinstance(model, nn.Module):
            model.to(self.device)
            model.eval()
            return model
        elif isinstance(model, (str, Path)):
            return TTEngineRegistry.get(self.config_manager, "inference_server", self.backend)(model_file=model, device=self.device, **kwargs)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    # ------------------------------------------------------------------
    # 4. 数据流水（子类可重写）
    # ------------------------------------------------------------------
    def register_parsers(self) -> None:
        """
        注册自定义 SourceParser（子类重写以添加解析器）。
        """
        pass

    def preprocess(self, data_info: BaseDataInfo) -> Any:
        """
        对输入数据进行预处理，如 resize、归一化、to(device) 等。

        Args:
            data_info (BaseDataInfo): 原始数据。

        Returns:
            Any: 预处理后的数据，可直接送入模型。
        """
        return data_info

    def inference(self, data: Any) -> Any:
        return self.model.inference(data)

    def postprocess(self, data_info: BaseDataInfo, inference_result: list[torch.Tensor]) -> Any:
        """
        对模型输出进行后处理，如 softmax、NMS、阈值过滤、解码等。

        Args:
            data_info (BaseDataInfo): 对应输入数据。
            inference_result (list[torch.Tensor]): 模型原始输出。

        Returns:
            Any: 后处理结果，通常是结构化的预测对象。
        """
        return inference_result[0]

    def show(self, data_info: BaseDataInfo, postprocess_result) -> Any:
        """
        可视化或序列化最终结果，如绘制框、保存图片、生成 JSON 等。

        Args:
            data_info (BaseDataInfo): 输入数据（含原始路径、尺寸等信息）。
            postprocess_result: 后处理后的结果。

        Returns:
            Any: 最终输出，供业务侧消费。
        """
        return data_info

    # ------------------------------------------------------------------
    # 5. 异步生产-消费（内部协奏）
    # ------------------------------------------------------------------
    def _produce(self, parser: SourceParser, source):
        """
        生产者线程：将解析后的数据项放入队列。

        Args:
            parser (SourceParser): 输入解析器。
            source: 原始输入源。
        """
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
        """
        消费者逻辑：批量或逐条推理。

        Yields:
            Any: 单条推理结果。
        """
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

    @torch.inference_mode()
    def _infer_batch(self, batch: list[Any]) -> Generator[Any, None, None]:
        """
        执行真正的推理逻辑（逐条或批量）。
        默认逐条处理，子类可重写以实现真正的批处理、异步调用或缓存合并。

        Args:
            batch (list[Any]): BaseDataInfo 列表。

        Yields:
            Any: show() 处理后的单条结果。
        """
        for data_info in batch:
            self.callback.run_callback(Events.ON_PREDICT_BATCH_START, self)

            with self.preprocess_recoder:
                self.preprocess_result = self.preprocess(data_info)
            self.callback.run_callback(Events.ON_PREDICT_PREPROCESS_END, self)

            with self.inference_recoder:
                self.inference_result = self.inference(self.preprocess_result)
            self.callback.run_callback(Events.ON_PREDICT_INFERENCE_END, self)

            with self.postprocess_recoder:
                self.postprocess_result = self.postprocess(data_info, self.inference_result)
            self.callback.run_callback(Events.ON_PREDICT_BATCH_END, self)

            yield self.show(data_info, self.postprocess_result)
