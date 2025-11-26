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

import tensorrt as trt

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from tensorrt_bindings import ICudaEngine

from tinytrain.utils import LOGGER

from .base_export_server import TTBaseExportServer
from .onnx_export_server import TTBaseOnnxExportServer

if TYPE_CHECKING:
    import torch


class TTBaseTensorRTExportServer(TTBaseExportServer):
    def __init__(self, model: torch.nn.Module, device: torch.device, export_dir: Path, **kwargs):
        super().__init__(model, device, export_dir)

        # 检查设备是否为CPU
        if device.type != 'cuda':
            raise ValueError(
                f"TensorRT export requires GPU device, but got {device.type}. "
                "Please set export device to CUDA."
            )

        # 检查TensorRT版本是否 >= 8.0
        self._check_tensorrt_version()

        # 版本检查
        LOGGER.info(
            f"TensorRT {trt.__version__} | "
            f"Device: {device}"
        )

        self.trt_model_path = self.export_dir / "model.engine"
        self.engine: Optional[trt.IHostMemory] = None

        # 参数验证和设置
        self.validate_and_set_kwargs(kwargs)

        # 构建ONNX导出服务
        self.onnx_export_server = self.get_onnx_export_server(model, device, export_dir, **kwargs)

    def _check_tensorrt_version(self):
        """检查TensorRT版本是否 >= 8.0"""
        version_str = trt.__version__

        # 解析版本号
        try:
            # 版本号格式可能是 "8.6.1.6" 或 "10.0.0.1" 等
            major_version = int(version_str.split('.')[0])

            if major_version < 8:
                raise RuntimeError(
                    f"TensorRT version {version_str} is not supported. "
                    f"This class requires TensorRT >= 8.0. "
                    f"Please upgrade your TensorRT installation."
                )

            LOGGER.info(f"TensorRT version check passed: {version_str} >= 8.0")

        except (ValueError, IndexError) as e:
            LOGGER.warning(f"Could not parse TensorRT version '{version_str}'. Proceeding with caution. Error: {e}")

    def validate_and_set_kwargs(self, kwargs):
        """验证和设置TensorRT导出参数"""
        # TensorRT 配置参数
        self.engine_name = kwargs.pop("engine_name", None)
        self.precision_mode = kwargs.pop("precision_mode", "fp32")
        self.workspace_size = kwargs.pop("workspace_size", 1 << 30)  # 1GB
        self.max_batch_size = kwargs.pop("max_batch_size", 1)
        self.min_batch_size = kwargs.pop("min_batch_size", 1)
        self.opt_batch_size = kwargs.pop("opt_batch_size", 1)
        self.builder_optimization_level = kwargs.pop("builder_optimization_level", 3)
        self.sparsity_flag = kwargs.pop("sparsity_flag", False)
        self.tf32_flag = kwargs.pop("tf32_flag", False)
        self.refittable = kwargs.pop("refittable", False)
        self.profiling_verbosity = kwargs.pop("profiling_verbosity", trt.ProfilingVerbosity.LAYER_NAMES_ONLY)

        # 动态形状配置
        self.dynamic_shapes = kwargs.pop("dynamic_shapes", {})

        # 性能调优参数
        self.builder_timeout = kwargs.pop("builder_timeout", None)
        self.max_aux_streams = kwargs.pop("max_aux_streams", None)
        self.hardware_compatibility_level = kwargs.pop("hardware_compatibility_level", None)

        # 验证精度模式
        valid_precision_modes = ["fp32", "fp16", "int8"]
        if self.precision_mode not in valid_precision_modes:
            raise ValueError(f"precision_mode must be one of {valid_precision_modes}, got {self.precision_mode}")

    def get_onnx_export_server(self, model, device, export_dir, **kwargs) -> TTBaseOnnxExportServer:
        """创建ONNX导出服务器实例，确保动态形状配置同步"""
        # 确保ONNX导出使用正确的配置
        onnx_kwargs = kwargs.copy()

        # 为TensorRT优化ONNX导出配置
        onnx_kwargs.setdefault("opset_version", 11)  # TensorRT兼容的opset
        onnx_kwargs.setdefault("do_constant_folding", True)
        onnx_kwargs.setdefault("use_onnxslim", False)  # TensorRT构建时自己会优化

        # 移除可能导致问题的配置
        if "jit_export" in onnx_kwargs and onnx_kwargs["jit_export"] is True:
            LOGGER.warning("JIT export may cause compatibility issues with TensorRT, disabling")
            onnx_kwargs["jit_export"] = False

        # 同步动态形状配置到ONNX导出
        if self.dynamic_shapes:
            onnx_kwargs["dynamic_shapes"] = True
            # 自动生成dynamic_axes配置
            dynamic_axes = self._generate_dynamic_axes_from_shapes()
            if dynamic_axes:
                onnx_kwargs["dynamic_axes"] = dynamic_axes

        return TTBaseOnnxExportServer(model, device, export_dir, **onnx_kwargs)

    def _generate_dynamic_axes_from_shapes(self):
        """从dynamic_shapes配置生成ONNX的dynamic_axes配置"""
        if not self.dynamic_shapes:
            return None

        dynamic_axes = {}

        # 获取输入名称（从ONNX配置或使用默认名称）
        input_names = getattr(self.onnx_export_server, 'input_names', None)
        if input_names is None:
            # 如果没有输入名称，根据dynamic_shapes的键生成
            input_names = list(self.dynamic_shapes.keys())

        # 为每个输入创建动态轴配置
        for i, input_name in enumerate(input_names):
            if input_name in self.dynamic_shapes:
                # 使用用户提供的具体配置
                shapes = self.dynamic_shapes[input_name]
                min_shape = shapes.get("min", [])
                dynamic_axes[input_name] = {}

                # 为每个可变维度创建动态轴
                for dim_idx in range(len(min_shape)):
                    dynamic_axes[input_name][dim_idx] = f"{input_name}_dim_{dim_idx}"
            else:
                # 默认配置：只将批量维度设为动态
                dynamic_axes[input_name] = {0: 'batch_size'}

        LOGGER.info(f"Generated dynamic_axes for ONNX: {dynamic_axes}")
        return dynamic_axes

    def setup_dynamic_shapes(self, builder, network, config):
        """设置动态形状配置"""
        if not self.dynamic_shapes:
            # 如果没有提供动态形状配置，使用批量大小配置
            profile = builder.create_optimization_profile()

            # 获取所有输入
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                input_name = input_tensor.name

                # 获取静态形状
                static_shape = list(input_tensor.shape)

                # 设置动态批量维度
                if static_shape[0] == -1:
                    min_shape = [self.min_batch_size] + static_shape[1:]
                    opt_shape = [self.opt_batch_size] + static_shape[1:]
                    max_shape = [self.max_batch_size] + static_shape[1:]
                else:
                    # 如果已经是静态批量大小，保持原样
                    min_shape = static_shape
                    opt_shape = static_shape
                    max_shape = static_shape

                # 验证形状有效性
                if any(dim <= 0 for dim in min_shape + opt_shape + max_shape):
                    LOGGER.warning(f"Invalid shape for {input_name}, using static shape")
                    min_shape = [abs(dim) for dim in min_shape]
                    opt_shape = [abs(dim) for dim in opt_shape]
                    max_shape = [abs(dim) for dim in max_shape]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                LOGGER.info(f"Set dynamic shape for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

            config.add_optimization_profile(profile)
        else:
            # 使用用户提供的动态形状配置
            profile = builder.create_optimization_profile()

            for input_name, shapes in self.dynamic_shapes.items():
                min_shape = shapes.get("min")
                opt_shape = shapes.get("opt")
                max_shape = shapes.get("max")

                if not all([min_shape, opt_shape, max_shape]):
                    LOGGER.warning(f"Incomplete dynamic shapes for {input_name}, skipping")
                    continue

                # 验证形状
                if len(min_shape) != len(opt_shape) or len(opt_shape) != len(max_shape):
                    raise ValueError(f"Shape dimensions mismatch for {input_name}")

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                LOGGER.info(f"Set custom dynamic shape for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

            config.add_optimization_profile(profile)

    def setup_builder_config(self, builder):
        """配置TensorRT构建器 - 仅使用TensorRT 8.0+ API"""
        config = builder.create_builder_config()

        # 设置工作空间大小 - 使用TensorRT 8.0+的memory_pool_limits
        if hasattr(config, 'set_memory_pool_limit'):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
            LOGGER.info(f"Set workspace memory pool limit: {self.workspace_size / 1024 / 1024:.2f} MB")
        else:
            LOGGER.warning("Cannot set workspace size - unsupported TensorRT version")

        config.builder_optimization_level = self.builder_optimization_level
        config.profiling_verbosity = self.profiling_verbosity

        # 设置超时（如果提供）
        if self.builder_timeout:
            config.set_timeout(self.builder_timeout)

        # 设置辅助流（如果提供）
        if self.max_aux_streams is not None:
            config.max_aux_streams = self.max_aux_streams

        # 设置硬件兼容性（如果提供）
        if self.hardware_compatibility_level:
            config.hardware_compatibility_level = self.hardware_compatibility_level

        # 设置精度模式
        if self.precision_mode == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                LOGGER.info("FP16 precision enabled")
            else:
                LOGGER.warning("Platform doesn't support fast FP16, using FP32")
        elif self.precision_mode == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                LOGGER.info("INT8 precision enabled")
            else:
                LOGGER.warning("Platform doesn't support fast INT8, using FP32")

        # 设置其他优化标志
        if self.sparsity_flag:
            # TensorRT 8.0+ 结构化稀疏性支持 [citation:1]
            if hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
                config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
                LOGGER.info("Sparse weights enabled - requires NVIDIA Ampere GPU or newer")
            else:
                LOGGER.warning("Sparse weights not supported in this TensorRT version")

        if self.tf32_flag:
            config.set_flag(trt.BuilderFlag.TF32)
            LOGGER.info("TF32 precision enabled")

        if self.refittable:
            if hasattr(trt.BuilderFlag, 'REFIT'):
                config.set_flag(trt.BuilderFlag.REFIT)
                LOGGER.info("Refit enabled")
            else:
                LOGGER.warning("Refit not supported in this TensorRT version")

        return config

    def build_engine(self):
        """构建TensorRT引擎 - 返回序列化引擎"""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # 创建网络 - 显式批处理
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network: trt.INetworkDefinition = builder.create_network(network_flags)

        # 设置网络名称
        if self.engine_name:
            network.name = self.engine_name

        parser = trt.OnnxParser(network, logger)

        # 导出ONNX文件
        LOGGER.info("Exporting ONNX model for TensorRT...")
        self.onnx_export_server.export()
        onnx_model_path = self.onnx_export_server.onnx_model_path

        # 解析ONNX模型
        LOGGER.info(f"Parsing ONNX model: {onnx_model_path}")
        with open(onnx_model_path, 'rb') as f:
            onnx_data = f.read()

        if not parser.parse(onnx_data):
            error_msg = "Failed to parse ONNX model:\n"
            for i in range(parser.num_errors):
                error_msg += f"  Error {i}: {parser.get_error(i)}\n"
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

        # 获取网络信息
        try:
            num_layers = network.num_layers
            num_inputs = network.num_inputs
            num_outputs = network.num_outputs
        except:
            num_layers = sum(1 for _ in network)
            num_inputs = sum(1 for i in range(100) if network.get_input(i) is not None)
            num_outputs = sum(1 for i in range(100) if network.get_output(i) is not None)

        LOGGER.info(f"ONNX model parsed successfully: {num_layers} layers, {num_inputs} inputs, {num_outputs} outputs")

        # 配置构建器
        config = self.setup_builder_config(builder)

        # 配置动态形状
        self.setup_dynamic_shapes(builder, network, config)

        # 构建引擎 - 仅支持TensorRT 8.0+的API
        try:
            LOGGER.info("Building TensorRT engine...")

            # TensorRT 8.0+ 使用 build_serialized_network，直接返回序列化引擎
            serialized_engine: trt.IHostMemory = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build serialized TensorRT engine")

            LOGGER.info("TensorRT engine built and serialized successfully")
            return serialized_engine

        except Exception as e:
            LOGGER.exception("Error building TensorRT engine")
            raise RuntimeError(f"Error building TensorRT engine: {e}") from e

    def export(self) -> None:
        """导出TensorRT模型"""
        try:
            LOGGER.info("Starting TensorRT export process...")

            # 构建引擎（返回序列化引擎）
            if self.engine is None:
                self.engine = self.build_engine()

            if self.engine is None:
                raise RuntimeError("TensorRT engine is None after build")

            # 直接保存序列化引擎到文件
            with open(self.trt_model_path, 'wb') as f:
                f.write(self.engine)  # self.engine 已经是序列化数据

            LOGGER.info(f"TensorRT model exported to: {self.trt_model_path}")

            # 输出引擎详细信息
            self._log_engine_info()

        except Exception as e:
            LOGGER.exception("Error exporting TensorRT model")
            raise RuntimeError(f"Error exporting TensorRT model: {e}") from e

    def _log_engine_info(self):
        """记录引擎详细信息 - 使用新的 TensorRT API"""
        if self.engine is None:
            LOGGER.warning("No serialized engine available for info logging")
            return

        try:
            # 获取文件大小
            engine_size = self.trt_model_path.stat().st_size

            # 创建runtime并反序列化引擎来获取详细信息
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            # 从序列化数据反序列化引擎
            deserialized_engine: ICudaEngine = runtime.deserialize_cuda_engine(self.engine)
            if deserialized_engine is None:
                LOGGER.warning("Failed to deserialize engine for info logging")
                return

            # 获取绑定信息
            num_io_tensors = deserialized_engine.num_io_tensors
            binding_info = []

            # 获取所有张量名称
            tensor_names = []
            for i in range(num_io_tensors):
                try:
                    tensor_name = deserialized_engine.get_tensor_name(i)
                    tensor_names.append(tensor_name)
                except Exception as e:
                    LOGGER.debug(f"Could not get tensor name for index {i}: {e}")
                    break

            # 为每个张量获取详细信息
            for tensor_name in tensor_names:
                try:
                    # 获取张量模式（输入/输出）
                    tensor_mode = deserialized_engine.get_tensor_mode(tensor_name)
                    tensor_mode_str = "input" if tensor_mode == trt.TensorIOMode.INPUT else "output"

                    # 获取张量数据类型
                    tensor_dtype = deserialized_engine.get_tensor_dtype(tensor_name)

                    # 获取张量形状
                    tensor_shape = deserialized_engine.get_tensor_shape(tensor_name)

                    binding_info.append(f"{tensor_name} ({tensor_mode_str}): shape={list(tensor_shape)}, dtype={tensor_dtype}")

                except Exception as e:
                    LOGGER.debug(f"Could not get tensor info for {tensor_name}: {e}")
                    binding_info.append(f"{tensor_name} (unknown): failed to get info")

            # 获取其他引擎信息
            num_layers = deserialized_engine.num_layers
            num_profiles = deserialized_engine.num_optimization_profiles
            device_memory_size = deserialized_engine.device_memory_size

            LOGGER.info(f"TensorRT Engine Info: {engine_size / 1024 / 1024:.2f}MB, "
                        f"{len(tensor_names)} IO tensors, {num_profiles} profiles, {num_layers} layers, "
                        f"device memory: {device_memory_size} bytes")

            LOGGER.info("Engine tensors:")
            for info in binding_info:
                LOGGER.info(f"  {info}")

            # 记录优化配置信息
            try:
                if num_profiles > 0:
                    LOGGER.info("Optimization profiles:")
                    for profile_idx in range(num_profiles):
                        for tensor_name in tensor_names:
                            try:
                                tensor_mode = deserialized_engine.get_tensor_mode(tensor_name)
                                if tensor_mode == trt.TensorIOMode.INPUT:
                                    # 获取输入张量的形状范围
                                    min_shape = deserialized_engine.get_tensor_profile_shape(tensor_name, profile_idx)[0]
                                    opt_shape = deserialized_engine.get_tensor_profile_shape(tensor_name, profile_idx)[1]
                                    max_shape = deserialized_engine.get_tensor_profile_shape(tensor_name, profile_idx)[2]
                                    LOGGER.info(f"  Profile {profile_idx} - {tensor_name}: min={list(min_shape)}, opt={list(opt_shape)}, max={list(max_shape)}")
                            except Exception as e:
                                LOGGER.debug(f"Could not get shape info for {tensor_name} in profile {profile_idx}: {e}")
            except Exception as e:
                LOGGER.debug(f"Could not get optimization profile info: {e}")

            # 记录其他引擎属性
            LOGGER.info(f"Engine name: {deserialized_engine.name}")
            LOGGER.info(f"Engine refittable: {deserialized_engine.refittable}")

            # 清理反序列化的引擎
            del deserialized_engine

        except Exception as e:
            LOGGER.warning(f"Failed to log engine info: {e}")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                del self.engine
            except:
                pass
