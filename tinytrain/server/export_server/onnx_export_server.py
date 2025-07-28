from __future__ import annotations

import onnx
import onnxslim
import torch

from pathlib import Path
from typing import TYPE_CHECKING

from .base_export_server import BaseExportServer
from tinytrain.utils import LOGGER

if TYPE_CHECKING:
    from torch import nn


class BaseOnnxExportServer(BaseExportServer):
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 **kwargs
                 ):
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
        # 准备一个示例输入，用于模型转换和推理
        self.dummy_inputs = tuple(torch.rand(input_shape, dtype=torch.float, requires_grad=False).to(self.device) for input_shape in self.input_shapes)

    def export(self, export_dir: str | Path = None):
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
