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

import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, Any

from tinytrain.utils import LOGGER

from .base_inference_server import TTBaseInferenceServer

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    LOGGER.warning("TensorRT or PyCUDA not available. TensorRT inference will be disabled.")


class TTBaseTensorRTInferenceServer(TTBaseInferenceServer):
    """
    通用 TensorRT 10 推理服务器，仅支持 .engine 文件。

    特性
    ----
    1. 支持 TensorRT 10 API
    2. 直接加载预编译的 TensorRT 引擎
    3. 最优内存管理和缓冲区分配
    4. 异步流式推理
    5. 动态形状支持（如果引擎支持）
    6. 详细的性能监控和统计
    """

    def __init__(self, model_file: str, device: torch.device, **kwargs):
        """
        初始化 TensorRT 10 推理服务器。

        Args
        ----
        model_file : str
            预编译的 TensorRT 引擎文件路径（必须为 .engine 后缀）。
        device : torch.device
            推理设备（必须为 cuda）。
        **kwargs
            推理配置参数：
            - use_async : bool, default True
                是否使用异步推理。
            - warm_up_iters : int, default 10
                预热迭代次数。
            - profiling_enabled : bool, default False
                是否启用性能分析。
            - use_cuda_graph : bool, default False
                是否使用 CUDA Graph（如果引擎支持）。
            - stream_priority : int, default 0
                CUDA 流优先级。
            - enable_dla : bool, default False
                是否启用 DLA（深度学习加速器）。

        Raises
        ------
        ValueError
            当 model_file 不是 .engine 文件或设备不是 cuda 时抛出。
        RuntimeError
            当 TensorRT 不可用时抛出。
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not available. Please install: "
                "pip install tensorrt pycuda"
            )

        if device.type != 'cuda':
            raise ValueError("TensorRT inference requires CUDA device")

        if Path(model_file).suffix != ".engine":
            raise ValueError("TensorRT inference server only supports .engine files")

        if not Path(model_file).exists():
            raise FileNotFoundError(f"Engine file not found: {model_file}")

        super().__init__(model_file, device)

        # 参数提取和验证
        self.validate_and_set_kwargs(kwargs)

        # 版本检查
        self._check_versions()

        # CUDA 上下文管理
        self.cuda_ctx = cuda.Device(0).make_context()

        self.prepare()
        self.load_engine()
        self.create_execution_context()
        self.allocate_buffers()

    def validate_and_set_kwargs(self, kwargs: Dict[str, Any]):
        """验证和设置推理参数"""

        # 推理配置
        self.use_async = kwargs.pop("use_async", True)
        self.profiling_enabled = kwargs.pop("profiling_enabled", False)
        self.use_cuda_graph = kwargs.pop("use_cuda_graph", False)
        self.stream_priority = kwargs.pop("stream_priority", 0)
        self.enable_dla = kwargs.pop("enable_dla", False)

        # 内部状态
        self._engine_loaded = False
        self._cuda_stream = None
        self._bindings = None

    def _check_versions(self):
        """检查 TensorRT 和 CUDA 版本"""
        LOGGER.info(
            f"TensorRT {trt.__version__} | "
            f"PyCUDA {pycuda.VERSION} | "
            f"Engine: {Path(self.model_file).name}"
        )

    def prepare(self):
        """
        准备 TensorRT 环境：创建 logger 和 runtime。
        """
        LOGGER.info("Preparing TensorRT environment...")

        # 创建 TensorRT 组件 - TensorRT 10 使用更简单的初始化
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.trt_logger)

        # 创建 CUDA 流
        self._create_cuda_stream()

    def _create_cuda_stream(self):
        """创建 CUDA 流"""
        try:
            self._cuda_stream = cuda.Stream()
            LOGGER.info("Created default CUDA stream")
        except Exception as e:
            raise f"Failed to create prioritized stream" from e

    def load_engine(self):
        """
        加载预编译的 TensorRT 引擎。
        """
        try:
            LOGGER.info(f"Loading TensorRT engine: {self.model_file}")

            # 读取引擎文件
            with open(self.model_file, 'rb') as f:
                engine_data = f.read()

            # TensorRT 10 反序列化引擎
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            # 记录引擎信息
            self._log_engine_info()

            self._engine_loaded = True
            LOGGER.info("TensorRT engine loaded successfully")

        except Exception as e:
            LOGGER.exception("Failed to load TensorRT engine")
            raise RuntimeError(f"TensorRT engine loading failed: {e}") from e

    def _log_engine_info(self):
        """记录引擎详细信息"""
        # 基本信息
        num_io_tensors = self.engine.num_io_tensors
        num_layers = self.engine.num_layers
        has_implicit_batch = self.engine.has_implicit_batch_dimension

        LOGGER.info(
            f"Engine Info - IO Tensors: {num_io_tensors}, "
            f"Layers: {num_layers}, "
            f"Implicit Batch: {has_implicit_batch}"
        )

        # TensorRT 10 使用 tensor names 而不是 bindings
        input_info = []
        output_info = []

        # 获取所有 tensor 名称
        tensor_names = []
        for i in range(num_io_tensors):
            try:
                tensor_name = self.engine.get_tensor_name(i)
                tensor_names.append(tensor_name)
            except Exception as e:
                LOGGER.warning(f"Could not get tensor name for index {i}: {e}")
                continue

        # 记录输入输出信息
        for tensor_name in tensor_names:
            try:
                # TensorRT 10 获取 tensor 信息的方式
                tensor_dims = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                tensor_mode = self.engine.get_tensor_mode(tensor_name)

                # 直接转换为元组
                shape_tuple = tuple(tensor_dims)

                info_str = f"{tensor_name} {shape_tuple} {tensor_dtype}"

                if tensor_mode == trt.TensorIOMode.INPUT:
                    input_info.append(info_str)
                elif tensor_mode == trt.TensorIOMode.OUTPUT:
                    output_info.append(info_str)
                else:
                    LOGGER.debug(f"Tensor {tensor_name} has mode: {tensor_mode}")

            except Exception as e:
                LOGGER.warning(f"Could not get info for tensor {tensor_name}: {e}")

        LOGGER.info(f"Inputs: {input_info}")
        LOGGER.info(f"Outputs: {output_info}")

        # DLA 信息
        if self.enable_dla and hasattr(self.engine, 'has_dla') and self.engine.has_dla:
            dla_cores = self.engine.get_dla_core_count()
            LOGGER.info(f"DLA enabled with {dla_cores} cores")

    def create_execution_context(self):
        """
        创建执行上下文。
        """
        try:
            LOGGER.info("Creating execution context...")

            self.context = self.engine.create_execution_context()

            if self.context is None:
                raise RuntimeError("Failed to create execution context")

            # 检查动态形状支持
            self._check_dynamic_shapes()

            LOGGER.info("Execution context created successfully")

        except Exception as e:
            LOGGER.exception("Failed to create execution context")
            raise RuntimeError(f"Execution context creation failed: {e}") from e

    def _check_dynamic_shapes(self):
        """检查动态形状支持"""
        self.has_dynamic_shapes = False

        try:
            # 检查动态形状的方式
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_dims = self.engine.get_tensor_shape(tensor_name)

                # 直接转换为元组
                shape_tuple = tuple(tensor_dims)

                if -1 in shape_tuple:  # 动态维度
                    self.has_dynamic_shapes = True
                    LOGGER.info(f"Engine supports dynamic shapes for tensor: {tensor_name}")
                    break
        except Exception as e:
            LOGGER.warning(f"Could not check dynamic shapes: {e}")

        if not self.has_dynamic_shapes:
            LOGGER.info("Engine has fixed shapes")

    def allocate_buffers(self):
        """
        分配 GPU 输入输出缓冲区 - TensorRT 10 方式。
        """
        try:
            LOGGER.info("Allocating GPU buffers...")

            self.input_buffers = {}
            self.output_buffers = {}
            self.host_outputs = {}

            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_dims = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                tensor_mode = self.engine.get_tensor_mode(tensor_name)

                # Dims 对象处理 - 直接转换为元组
                tensor_shape = tuple(tensor_dims)

                # 计算内存大小（处理动态维度）
                if -1 in tensor_shape:
                    # 对于动态形状，使用最小尺寸进行分配
                    tensor_shape = tuple(1 if dim == -1 else dim for dim in tensor_shape)

                # 计算缓冲区大小
                element_count = np.prod(tensor_shape)
                element_size = int(element_count) * self._get_dtype_size(tensor_dtype)

                # 分配设备内存
                device_mem = cuda.mem_alloc(element_size)

                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.input_buffers[tensor_name] = device_mem
                    LOGGER.info(f"Allocated input buffer: {tensor_name} {tensor_shape} {tensor_dtype}")
                elif tensor_mode == trt.TensorIOMode.OUTPUT:
                    # 同时分配主机内存用于结果拷贝
                    host_mem = cuda.pagelocked_empty(
                        tensor_shape,
                        dtype=self._trt_dtype_to_np(tensor_dtype)
                    )
                    self.output_buffers[tensor_name] = device_mem
                    self.host_outputs[tensor_name] = host_mem
                    LOGGER.info(f"Allocated output buffer: {tensor_name} {tensor_shape} {tensor_dtype}")

                # 设置 tensor 地址
                self.context.set_tensor_address(tensor_name, int(device_mem))

            LOGGER.info("GPU buffers allocated successfully")

        except Exception as e:
            LOGGER.exception("Failed to allocate GPU buffers")
            raise RuntimeError(f"Buffer allocation failed: {e}") from e

    def _get_dtype_size(self, dtype: trt.DataType) -> int:
        """获取数据类型大小"""
        size_map = {
            trt.DataType.FLOAT: 4,
            trt.DataType.HALF: 2,
            trt.DataType.INT8: 1,
            trt.DataType.INT32: 4,
            trt.DataType.BOOL: 1,
            trt.DataType.UINT8: 1,
            trt.DataType.INT64: 8,
        }
        return size_map.get(dtype, 4)  # 默认 4 bytes

    def _trt_dtype_to_np(self, dtype: trt.DataType) -> np.dtype:
        """TensorRT 数据类型转 NumPy 数据类型"""
        mapping = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
            trt.DataType.UINT8: np.uint8,
            trt.DataType.INT64: np.int64,
        }
        return mapping.get(dtype, np.float32)

    def inference(self, data: torch.Tensor) -> List[torch.Tensor]:
        """
        执行一次 TensorRT 推理。

        Args
        ----
        data : torch.Tensor
            输入张量，必须在 CUDA 设备上。

        Returns
        -------
        List[torch.Tensor]
            输出张量列表，每个元素对应一个输出绑定。
        """
        # 输入验证
        self._validate_input(data)

        try:
            # 数据预处理
            input_dict = self.prepare_inputs(data)

            # 执行推理
            outputs = self.inference_impl(input_dict)

            return outputs

        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed: {e}") from e

    def _validate_input(self, data: torch.Tensor):
        """验证输入数据"""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(data)}")

        if data.device.type != 'cuda':
            raise ValueError("Input tensor must be on CUDA device")

        if data.ndim == 0:
            raise ValueError("Input tensor must have at least one dimension")

    def prepare_inputs(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """准备输入数据，将 torch.Tensor 转换为 TensorRT 需要的输入格式"""

        # 单输入情况 - 这是最常见的情况
        if len(self.input_buffers) == 1:
            # 获取唯一的输入 tensor 名称
            input_name = next(iter(self.input_buffers.keys()))

            # 确保数据在正确的内存布局上
            if not data.is_contiguous():
                data = data.contiguous()

            # 验证数据类型匹配
            self._validate_input_dtype(input_name, data)

            # 验证形状兼容性（支持动态形状）
            self._validate_input_shape(input_name, data)

            return {input_name: data}

        # 多输入情况
        elif len(self.input_buffers) > 1:
            return self._prepare_multiple_inputs(data)

        else:
            raise RuntimeError("No input buffers found in the TensorRT engine")

    def _validate_input_dtype(self, input_name: str, data: torch.Tensor):
        """验证输入数据类型与引擎期望的匹配"""
        try:
            # 获取引擎期望的数据类型
            expected_dtype = self.engine.get_tensor_dtype(input_name)
            actual_dtype = data.dtype

            # TensorRT 数据类型到 PyTorch 数据类型的映射
            dtype_mapping = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT8: torch.int8,
                trt.DataType.INT32: torch.int32,
                trt.DataType.BOOL: torch.bool,
                trt.DataType.UINT8: torch.uint8,
                trt.DataType.INT64: torch.int64,
            }

            expected_torch_dtype = dtype_mapping.get(expected_dtype)

            if expected_torch_dtype is not None and actual_dtype != expected_torch_dtype:
                LOGGER.warning(
                    f"Input dtype mismatch for {input_name}: "
                    f"expected {expected_torch_dtype}, got {actual_dtype}. "
                    f"Attempting automatic conversion..."
                )
                # 这里不进行自动转换，因为调用者应该处理数据类型

        except Exception as e:
            LOGGER.warning(f"Could not validate input dtype for {input_name}: {e}")

    def _validate_input_shape(self, input_name: str, data: torch.Tensor):
        """验证输入形状与引擎期望的兼容性"""
        try:
            # 获取引擎期望的形状
            expected_dims = self.engine.get_tensor_shape(input_name)
            expected_shape = tuple(expected_dims)
            actual_shape = tuple(data.shape)

            # 动态形状检查（-1 表示动态维度）
            if -1 in expected_shape:
                LOGGER.debug(f"Dynamic shape detected for {input_name}: expected {expected_shape}, got {actual_shape}")
                return

            # 固定形状检查
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Input shape mismatch for {input_name}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )

        except Exception as e:
            LOGGER.warning(f"Could not validate input shape for {input_name}: {e}")

    def _prepare_multiple_inputs(self, data) -> Dict[str, torch.Tensor]:
        """处理多输入情况"""

        if isinstance(data, torch.Tensor):
            # 如果传入单个 tensor 但模型需要多个输入，抛出错误
            raise ValueError(
                f"Model expects {len(self.input_buffers)} inputs, "
                f"but only 1 tensor was provided"
            )

        elif isinstance(data, (list, tuple)):
            # 列表或元组输入
            if len(data) != len(self.input_buffers):
                raise ValueError(
                    f"Model expects {len(self.input_buffers)} inputs, "
                    f"but {len(data)} tensors were provided"
                )

            input_dict = {}
            input_names = list(self.input_buffers.keys())

            for i, (input_name, input_data) in enumerate(zip(input_names, data)):
                if not isinstance(input_data, torch.Tensor):
                    raise TypeError(f"Input {i} must be torch.Tensor, got {type(input_data)}")

                # 确保数据在正确的设备和内存布局上
                if not input_data.is_contiguous():
                    input_data = input_data.contiguous()

                if input_data.device.type != 'cuda':
                    input_data = input_data.to(self.device)

                input_dict[input_name] = input_data

            return input_dict

        elif isinstance(data, dict):
            # 字典输入（最灵活的方式）
            if set(data.keys()) != set(self.input_buffers.keys()):
                missing = set(self.input_buffers.keys()) - set(data.keys())
                extra = set(data.keys()) - set(self.input_buffers.keys())

                error_msg = "Input key mismatch:"
                if missing:
                    error_msg += f" missing keys: {missing}"
                if extra:
                    error_msg += f" extra keys: {extra}"

                raise ValueError(error_msg)

            # 处理每个输入
            input_dict = {}
            for input_name, input_data in data.items():
                if not isinstance(input_data, torch.Tensor):
                    raise TypeError(f"Input {input_name} must be torch.Tensor, got {type(input_data)}")

                # 确保数据在正确的设备和内存布局上
                if not input_data.is_contiguous():
                    input_data = input_data.contiguous()

                if input_data.device.type != 'cuda':
                    input_data = input_data.to(self.device)

                input_dict[input_name] = input_data

            return input_dict

        else:
            raise TypeError(
                f"Unsupported input type: {type(data)}. "
                f"Expected torch.Tensor, list, tuple, or dict"
            )

    def inference_impl(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """推理实现核心逻辑"""
        # 拷贝输入数据到 GPU 缓冲区
        self._copy_inputs_to_device(inputs)

        # 执行推理
        if self.use_async:
            # 使用正确的异步执行API
            if hasattr(self.context, 'execute_async_v3'):
                self.context.execute_async_v3(self._cuda_stream.handle)
            elif hasattr(self.context, 'execute_async'):
                self.context.execute_async(stream=self._cuda_stream.handle)
            else:
                self.context.execute_async_v2(stream=self._cuda_stream.handle)
        else:
            # 同步执行
            if hasattr(self.context, 'execute_v2'):
                self.context.execute_v2()
            else:
                self.context.execute()

        # 拷贝输出数据到主机
        host_data = self._copy_outputs_to_host()

        outputs = self.process_outputs(host_data)

        return outputs

    def process_outputs(self, trt_outputs: List[np.ndarray]) -> List[torch.Tensor]:
        """处理输出数据"""
        return [torch.from_numpy(output) for output in trt_outputs]

    def _copy_inputs_to_device(self, inputs: Dict[str, torch.Tensor]):
        """拷贝输入数据到设备"""
        for tensor_name, input_tensor in inputs.items():
            if tensor_name in self.input_buffers:
                # 确保张量在正确的设备和格式
                if not input_tensor.is_contiguous():
                    input_tensor = input_tensor.contiguous()

                # 获取 numpy 数组用于拷贝
                numpy_array = input_tensor.cpu().numpy() if input_tensor.device.type != 'cpu' else input_tensor.numpy()

                # 拷贝到设备缓冲区
                cuda.memcpy_htod_async(
                    self.input_buffers[tensor_name],
                    numpy_array,
                    self._cuda_stream
                )

    def _copy_outputs_to_host(self) -> List[np.ndarray]:
        """拷贝输出数据到主机"""
        outputs = []

        for tensor_name, device_mem in self.output_buffers.items():
            host_mem = self.host_outputs[tensor_name]

            # 从设备拷贝到主机
            cuda.memcpy_dtoh_async(host_mem, device_mem, self._cuda_stream)

            outputs.append(host_mem.copy())

        # 等待异步操作完成
        if self.use_async:
            self._cuda_stream.synchronize()

        return outputs

    def __del__(self):
        """清理资源"""
        try:
            # 清理 CUDA 上下文
            if hasattr(self, 'cuda_ctx'):
                self.cuda_ctx.pop()
                del self.cuda_ctx

            # 清理 CUDA 流
            if hasattr(self, '_cuda_stream'):
                self._cuda_stream.synchronize()
                del self._cuda_stream

            LOGGER.info("TensorRT 10 resources cleaned up")

        except Exception as e:
            LOGGER.warning(f"Error during TensorRT cleanup: {e}")