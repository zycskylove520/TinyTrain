from __future__ import annotations

import onnx
import onnxslim
import torch

from pathlib import Path
from typing import TYPE_CHECKING

from tinytrain.utils import LOGGER

from .base_export_server import BaseExportServer

if TYPE_CHECKING:
    from torch import nn


class BaseOnnxExportServer(BaseExportServer):
    """
    通用 ONNX 导出服务器，支持 torch → ONNX → onnxslim 完整链路。
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 **kwargs
                 ):
        """
        初始化 ONNX 导出服务器。

        Args
        ----
        model : nn.Module
            已加载权重的 PyTorch 模型。
        device : torch.device
            当前模型所在的计算设备（cpu / cuda）。
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
        super().__init__(model, device)
        LOGGER.info(
            f"PyTorch {torch.__version__} | "
            f"ONNX {onnx.__version__} | "
            f"onnxslim {onnxslim.__version__}"
        )

        self.dummy_inputs: tuple | None = None
        self.input_shapes = kwargs.get("input_shapes")
        if self.input_shapes is None:
            raise ValueError("Missing required argument: input_shapes. Please pass it via kwargs, e.g., input_shapes=[(1, 3, 224, 224)]")

        self.use_onnxslim = kwargs.get("use_onnxslim", True)
        self.jit_export = kwargs.get("jit_export", False)
        self.input_names = kwargs.get("input_names")
        self.output_names = kwargs.get("output_names")
        self.do_constant_folding = kwargs.get("do_constant_folding", True)
        self.opset_version = kwargs.get("opset_version")
        self.dynamic_axes = kwargs.get("dynamic_axes")
        self.dynamic_shapes = kwargs.get("dynamic_shapes")

        self.prepare()

    def prepare(self):
        """
        根据 input_shapes 生成 dummy 输入张量，用于 tracing 与验证。
        所有张量默认移动到与模型相同的 device，且无需梯度。
        """
        self.dummy_inputs = tuple(torch.rand(input_shape, dtype=torch.float, requires_grad=False).to(self.device) for input_shape in self.input_shapes)

    def export(self, export_dir: str | Path = None):
        """
        执行 torch → ONNX → onnxslim 的完整导出流程。

        Args
        ----
        export_dir : str | Path | None
            导出目录，若为空则保存到当前目录下的 model.onnx。

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            torch.onnx.export 或 onnxslim 任意环节失败时抛出。
        """
        onnx_model_path = Path(export_dir) / "model.onnx" if export_dir else Path("model.onnx")
        onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

        # jit trace（如果启用）
        if self.jit_export:
            try:
                self.model = torch.jit.trace(self.model, self.dummy_inputs, strict=False)
            except Exception as e:
                LOGGER.exception("torch.jit.trace failed")
                raise

        LOGGER.info("Start exporting ONNX model...")

        try:
            torch.onnx.export(
                model=self.model,
                args=self.dummy_inputs,
                f=str(onnx_model_path),  # torch.onnx.export 只接受 str
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=self.do_constant_folding,
                opset_version=self.opset_version,
                dynamic_axes=self.dynamic_axes,
            )
        except Exception as e:
            LOGGER.exception("ONNX export failed")
            raise RuntimeError("Failed to export ONNX model") from e

        LOGGER.info("Finish exporting ONNX model.")

        # 优化模型
        if self.use_onnxslim:
            try:
                LOGGER.info("Using onnxslim...")

                # 验证 ONNX 模型
                onnx_model = onnx.load(onnx_model_path)
                LOGGER.info("check onnx model...")
                onnx.checker.check_model(onnx_model)
                LOGGER.info("onnx model check passed!")

                # 优化模型
                LOGGER.info("start onnx slim...")
                model_onnx = onnxslim.slim(onnx_model)
                onnx.save(model_onnx, onnx_model_path)
                LOGGER.info("onnx onnx slim completed!")
            except Exception as e:
                LOGGER.exception("ONNX slim failed")
                raise RuntimeError("Failed to export ONNX slim model") from e

        LOGGER.info(f"Model has been successfully converted to ONNX format and saved to {onnx_model_path}")
