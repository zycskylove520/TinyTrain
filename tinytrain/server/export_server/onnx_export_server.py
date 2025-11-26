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
import onnxslim
import torch

from pathlib import Path
from typing import TYPE_CHECKING

from tinytrain.utils import LOGGER

from .base_export_server import TTBaseExportServer

if TYPE_CHECKING:
    from torch import nn


class TTBaseOnnxExportServer(TTBaseExportServer):
    """
    通用 ONNX 导出服务器，支持 torch → ONNX → onnxslim 完整链路。
    """

    def __init__(self, model: nn.Module, device: torch.device, export_dir: Path, **kwargs):
        """
        初始化 ONNX 导出服务器。

        Args
        ----
        model : nn.Module
            已加载权重的 PyTorch 模型。
        device : torch.device
            当前模型所在的计算设备（cpu / cuda）。
        export_dir : Path
            导出目录。
        **kwargs
            导出超参与开关：
            - input_shapes : list[tuple[int, ...]]
                用于 tracing 的 dummy 输入形状（必填）。
            - use_onnxslim : bool, default True
                是否使用 onnxslim 精简与优化模型。
            - jit_export : bool, default False
                是否先用 torch.jit.trace 后再导出。
            - input_names / output_names : list[str] | None
                输入/输出节点名称，用于动态轴。
            - do_constant_folding : bool, default True
                是否折叠常量节点。
            - opset_version : int | None
                导出 ONNX 的 opset 版本。
            - dynamic_axes : dict[str, dict[int, str]] | None
                动态轴描述。
            - dynamic_shapes : bool | None
                是否启用动态 shape（与 dynamic_axes 二选一即可）。
        Raises
        ------
        ValueError
            未提供 `input_shapes` 时抛出。
        """
        super().__init__(model, device, export_dir)

        # 版本检查和日志
        LOGGER.info(
            f"PyTorch {torch.__version__} | "
            f"ONNX {onnx.__version__} | "
            f"onnxslim {onnxslim.__version__} | "
            f"Device: {device}"
        )

        self.onnx_model_path = self.export_dir / "model.onnx"
        self.dummy_inputs: tuple | None = None

        # 参数提取和验证
        self.validate_and_set_kwargs(kwargs)

        self.prepare()

    def validate_and_set_kwargs(self, kwargs):
        """验证和设置导出参数"""

        # 必需参数检查
        if "input_shapes" not in kwargs:
            raise ValueError("Missing required argument: input_shapes. Please pass it via kwargs, e.g., input_shapes=[(1, 3, 224, 224)]")

        self.input_shapes = kwargs.pop("input_shapes")

        # 验证输入形状
        if not isinstance(self.input_shapes, (list, tuple)):
            raise TypeError(f"input_shapes must be list or tuple, got {type(self.input_shapes)}")

        # 可选参数设置
        self.use_onnxslim = kwargs.pop("use_onnxslim", True)
        self.input_names = kwargs.pop("input_names", None)
        self.output_names = kwargs.pop("output_names", None)
        self.do_constant_folding = kwargs.pop("do_constant_folding", True)
        self.opset_version = kwargs.pop("opset_version", 17)  # 提供默认值
        self.dynamic_axes = kwargs.pop("dynamic_axes", None)
        self.dynamic_shapes = kwargs.pop("dynamic_shapes", None)
        self.export_params = kwargs.pop("export_params", True)
        self.external_data = kwargs.pop("external_data", False)
        self.jit_export = kwargs.pop("jit_export", False)
        self.verbose = kwargs.pop("verbose", False)

        # 自动推断输入输出名称（如果未提供）
        if self.input_names is None:
            self.input_names = [f"input_{i}" for i in range(len(self.input_shapes))]

        # 处理动态轴
        self.setup_dynamic_axes()

    def setup_dynamic_axes(self):
        """设置动态轴配置"""
        if self.dynamic_shapes and self.dynamic_axes is None:
            # 如果提供了dynamic_shapes但未提供dynamic_axes，自动生成
            self.dynamic_axes = {}
            for i, input_name in enumerate(self.input_names):
                self.dynamic_axes[input_name] = {
                    0: 'batch_size',
                    # 可以根据需要添加其他动态维度
                }
            LOGGER.info(f"Auto-generated dynamic_axes: {self.dynamic_axes}")

    def prepare(self):
        """
        根据 input_shapes 生成 dummy 输入张量，支持多种数据类型。
        """
        import torch

        # 支持不同的数据类型
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'int32': torch.int32,
            'int64': torch.int64,
            'bool': torch.bool
        }

        input_dtypes = getattr(self, 'input_dtypes', ['float32'] * len(self.input_shapes))

        self.dummy_inputs = []
        for i, (shape, dtype_str) in enumerate(zip(self.input_shapes, input_dtypes)):
            dtype = dtype_map.get(dtype_str, torch.float32)

            # 根据数据类型生成合适的虚拟数据
            if dtype in [torch.int32, torch.int64]:
                dummy_tensor = torch.randint(0, 100, shape, dtype=dtype, device=self.device)
            elif dtype == torch.bool:
                dummy_tensor = torch.randint(0, 2, shape, dtype=dtype, device=self.device)
            else:
                dummy_tensor = torch.randn(shape, dtype=dtype, device=self.device)

            self.dummy_inputs.append(dummy_tensor)

        self.dummy_inputs = tuple(self.dummy_inputs)

        # 记录输入信息
        input_info = []
        for i, (tensor, name) in enumerate(zip(self.dummy_inputs, self.input_names)):
            input_info.append(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

        LOGGER.info(f"Prepared dummy inputs: {', '.join(input_info)}")

    def export(self):
        """
        执行 torch → ONNX → onnxslim 的完整导出流程，支持更多优化选项。
        """
        import tempfile
        import shutil

        # 确保导出目录存在
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # JIT trace（如果启用）
        if self.jit_export:
            self._jit_trace_model()

        LOGGER.info("Starting ONNX export...")
        LOGGER.info(f"Export config: opset={self.opset_version}, "
                    f"constant_folding={self.do_constant_folding}, "
                    f"dynamic_axes={self.dynamic_axes is not None}")

        try:
            # ONNX导出
            torch.onnx.export(
                model=self.model,
                args=self.dummy_inputs,
                f=str(self.onnx_model_path),
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=self.do_constant_folding,
                opset_version=self.opset_version,
                dynamic_axes=self.dynamic_axes,
                export_params=self.export_params,
                verbose=self.verbose,
                training=torch.onnx.TrainingMode.EVAL,  # 明确设置为推理模式
                external_data=self.external_data,
            )

            LOGGER.info(f"ONNX model successfully exported to: {self.onnx_model_path}")

            # 验证和优化
            self._post_process_onnx()

        except Exception as e:
            LOGGER.exception("ONNX export failed")
            raise RuntimeError(f"Failed to export ONNX model: {e}") from e

    def _jit_trace_model(self):
        """JIT trace模型"""
        try:
            LOGGER.info("Starting JIT trace...")

            # 确保模型在eval模式
            self.model.eval()

            # 使用更严格的trace设置
            with torch.no_grad():
                if hasattr(self.model, 'get_trace_config'):
                    # 如果模型有自定义trace配置，使用它
                    trace_config = self.model.get_trace_config()
                    self.model = torch.jit.trace(self.model, self.dummy_inputs, **trace_config)
                else:
                    self.model = torch.jit.trace(self.model, self.dummy_inputs, strict=False, check_trace=True)

            LOGGER.info("JIT trace completed successfully")

        except Exception as e:
            LOGGER.exception("JIT trace failed")
            raise RuntimeError("JIT trace failed") from e

    def _post_process_onnx(self):
        """ONNX后处理：验证和优化"""
        try:
            # 验证ONNX模型
            LOGGER.info("Validating ONNX model...")
            onnx_model = onnx.load(self.onnx_model_path)
            onnx.checker.check_model(onnx_model)

            # 输出模型信息
            self._log_onnx_info(onnx_model)

            LOGGER.info("ONNX model validation passed!")

            # 优化模型
            if self.use_onnxslim:
                self._optimize_with_onnxslim(onnx_model)

        except Exception as e:
            LOGGER.exception("ONNX post-processing failed")
            raise RuntimeError(f"ONNX post-processing failed: {e}") from e

    def _log_onnx_info(self, onnx_model):
        """记录ONNX模型信息"""

        input_info = []
        output_info = []
        node_count = 0

        for _ in onnx_model.graph.node:
            node_count += 1

        for input in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else -1
                     for dim in input.type.tensor_type.shape.dim]
            input_info.append(f"{input.name} {shape}")

        for output in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else -1
                     for dim in output.type.tensor_type.shape.dim]
            output_info.append(f"{output.name} {shape}")

        LOGGER.info(f"ONNX Model Info: {node_count} nodes, "
                    f"{len(input_info)} inputs, {len(output_info)} outputs")
        LOGGER.info(f"Inputs: {input_info}")
        LOGGER.info(f"Outputs: {output_info}")

    def _optimize_with_onnxslim(self, onnx_model):
        """使用onnxslim优化模型"""
        try:
            LOGGER.info("Starting ONNX optimization with onnxslim...")

            # 保存原始模型大小
            original_size = self.onnx_model_path.stat().st_size

            # 优化模型
            model_optimized = onnxslim.slim(onnx_model)

            # 保存优化后的模型
            optimized_path = self.export_dir / "model_slim.onnx"
            onnx.save(model_optimized, optimized_path)

            # 记录优化结果
            optimized_size = optimized_path.stat().st_size
            compression_ratio = optimized_size / original_size

            LOGGER.info(f"ONNX optimization completed: "
                        f"original={original_size / 1024 / 1024:.2f}MB, "
                        f"optimized={optimized_size / 1024 / 1024:.2f}MB, "
                        f"ratio={compression_ratio:.2f}")

        except Exception as e:
            LOGGER.exception("ONNX optimization failed")
            # 优化失败不应该影响原始导出
            LOGGER.warning("Continuing with unoptimized model")
