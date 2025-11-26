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

import onnx
import torch
import onnxruntime as ort

from pathlib import Path
from tabulate import tabulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tinytrain.utils import LOGGER

from .base_inference_server import TTBaseInferenceServer

if TYPE_CHECKING:
    import numpy as np


class TTBaseOnnxInferenceServer(TTBaseInferenceServer):
    """
    通用 ONNX 推理服务器，基于 onnxruntime 实现。

    特性
    ----
    1. 自动设备检测和最优provider选择
    2. 性能优化配置（线程、内存、执行模式）
    3. 异步推理支持
    4. 详细的性能监控和统计
    5. 输入验证和错误处理
    """

    def __init__(self, model_file: str, device: torch.device, **kwargs):
        """
        初始化 ONNX 推理服务器。

        Args
        ----
        model_file : str
            已导出的 ONNX 模型路径。
        device : torch.device
            期望运行设备；自动选择最优的ExecutionProvider。
        **kwargs
            推理配置参数：
            - session_options : Dict[str, Any]
                ONNX Runtime会话选项
            - providers : List[str]
                自定义provider列表
            - enable_profiling : bool, default False
                是否启用性能分析
            - warm_up_iters : int, default 10
                预热迭代次数
            - use_io_binding : bool, default True
                是否使用IO绑定优化
            - async_enabled : bool, default False
                是否启用异步推理
        """
        super().__init__(model_file, device)

        # 参数提取和验证
        self.validate_and_set_kwargs(kwargs)

        # 版本检查和日志
        LOGGER.info(
            f"ONNX Runtime {ort.__version__} | "
            f"Device: {device} | "
            f"Model: {Path(model_file).name}"
        )

        self.prepare()

        self.ort_session = None
        self.create_inference_session()

    def validate_and_set_kwargs(self, kwargs):
        """验证和设置推理参数"""

        # 会话配置
        self.session_options = kwargs.pop("session_options", {})
        self.custom_providers = kwargs.pop("providers", None)
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.warm_up_iters = kwargs.pop("warm_up_iters", 10)
        self.use_io_binding = kwargs.pop("use_io_binding", True)
        self.async_enabled = kwargs.pop("async_enabled", False)

    def prepare(self):
        """
        模型加载前准备工作：验证模型和配置优化选项。
        """
        LOGGER.info("Preparing ONNX model for inference...")

        # 验证 ONNX 模型
        self._validate_onnx_model()

        # 创建优化会话选项
        self._setup_session_options()

        # 选择最优provider
        self.providers = self._select_optimal_providers()

    def _validate_onnx_model(self):
        """验证ONNX模型文件"""
        try:
            LOGGER.info("Validating ONNX model...")
            onnx_model = onnx.load(self.model_file)
            onnx.checker.check_model(onnx_model)
            LOGGER.info("ONNX model validation passed!")

            # 记录模型信息
            self._log_model_info(onnx_model)

        except Exception as e:
            LOGGER.exception("ONNX model validation failed")
            raise RuntimeError(f"Invalid ONNX model: {e}") from e

    def _log_model_info(self, onnx_model):
        """记录模型结构信息"""
        input_info = []
        output_info = []
        node_count = 0

        for _ in onnx_model.graph.node:
            node_count += 1

        for _input in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else -1
                     for dim in _input.type.tensor_type.shape.dim]
            input_info.append([_input.name, shape])

        for _output in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else -1
                     for dim in _output.type.tensor_type.shape.dim]
            output_info.append([_output.name, shape])

        LOGGER.info(f"ONNX Model Structure: {node_count} nodes, "
                    f"{len(input_info)} inputs, {len(output_info)} outputs")

        print("Inputs:")
        print(tabulate(input_info, headers=["Name", "Shape"], tablefmt="grid"))
        print("Outputs:")
        print(tabulate(output_info, headers=["Name", "Shape"], tablefmt="grid"))

    def _setup_session_options(self):
        """设置会话优化选项"""
        self.session_options = ort.SessionOptions()

        # 性能优化配置
        self.session_options.intra_op_num_threads = 4
        self.session_options.inter_op_num_threads = 2
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 内存优化
        self.session_options.enable_mem_pattern = True
        self.session_options.enable_mem_reuse = True

        # 性能分析
        if self.enable_profiling:
            self.session_options.enable_profiling = True
            LOGGER.info("Performance profiling enabled")

    def _select_optimal_providers(self) -> List[str]:
        """选择最优的ExecutionProvider"""
        if self.custom_providers:
            LOGGER.info(f"Using custom providers: {self.custom_providers}")
            return self.custom_providers

        available_providers = ort.get_available_providers()
        LOGGER.info(f"Available providers: {available_providers}")

        if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            LOGGER.info("Selected CUDAExecutionProvider as primary")
        else:
            providers = ['CPUExecutionProvider']
            LOGGER.info("Selected CPUExecutionProvider")

        return providers

    def create_inference_session(self):
        """创建推理会话"""
        try:
            LOGGER.info("Creating ONNX Runtime inference session...")

            self.ort_session = ort.InferenceSession(
                self.model_file,
                sess_options=self.session_options,
                providers=self.providers
            )

            # 缓存输入输出信息
            self._cache_io_info()

            LOGGER.info("Inference session created successfully")

        except Exception as e:
            LOGGER.exception("Failed to create inference session")
            raise RuntimeError(f"Inference session creation failed: {e}") from e

    def _cache_io_info(self):
        """缓存输入输出信息以提升性能"""
        self.input_info = self.ort_session.get_inputs()
        self.output_info = self.ort_session.get_outputs()

        self.input_names = [info.name for info in self.input_info]
        self.output_names = [info.name for info in self.output_info]

        LOGGER.info(f"Cached IO info - Inputs: {self.input_names}, Outputs: {self.output_names}")

    def inference(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        执行一次前向推理。

        Args
        ----
        data : torch.Tensor
            输入张量，形状需与 ONNX 模型输入一致。

        Returns
        -------
        List[torch.Tensor]
            输出张量列表，每个元素对应 ONNX 模型的一个输出节点。
        """

        # 输入验证
        self.validate_input(data)

        try:
            # 数据预处理
            input_dict = self.prepare_inputs(data)

            # 执行推理
            ort_outputs = self.ort_session.run(self.output_names, input_dict)

            # 后处理
            outputs = self.process_outputs(ort_outputs)

            return outputs

        except Exception as e:
            raise RuntimeError(f"Inference execution failed: {e}") from e

    def validate_input(self, data: torch.Tensor):
        """验证输入数据"""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")

        if data.ndim == 0:
            raise ValueError("Input tensor must have at least one dimension")

        # 检查形状兼容性（支持动态形状）
        if len(self.input_info) == 1:
            expected_shape = self.input_info[0].shape
            actual_shape = data.shape

            # 动态形状验证（跳过动态维度）
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if not isinstance(expected, str) and expected > 0 and expected != actual:
                    LOGGER.warning(
                        f"Shape mismatch at dimension {i}: expected {expected}, got {actual}. "
                        f"This might cause issues if the model doesn't support dynamic shapes."
                    )

    def prepare_inputs(self, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """准备输入数据"""
        if len(self.input_info) == 1:
            # 单输入情况
            input_name = self.input_info[0].name
            input_data = data.to("cpu").numpy()
            return {input_name: input_data}
        else:
            # 多输入情况（需要根据具体模型调整）
            raise NotImplementedError("Multi-input inference not yet implemented")

    def process_outputs(self, ort_outputs: List[np.ndarray]) -> List[torch.Tensor]:
        """处理输出数据"""
        return [torch.from_numpy(output) for output in ort_outputs]

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'ort_session'):
            del self.ort_session
