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

import onnx
import torch
import onnxruntime as ort

from tabulate import tabulate

from tinytrain.utils import LOGGER

from .base_inference_server import TTBaseInferenceServer


class TTBaseOnnxInferenceServer(TTBaseInferenceServer):
    """
    ONNX 通用推理服务器，基于 onnxruntime 实现。

    特性
    ----
    1. 加载并校验 ONNX 模型文件。
    2. 打印输入/输出节点信息，便于调试。
    3. 支持 CPU / CUDA EP，自动适配 device。
    4. 统一返回 `list[torch.Tensor]`，方便下游组件。
    """

    def __init__(self, model_file, device, **kwargs):
        """
        初始化 ONNX 推理服务器。

        Args
        ----
        model_file : str
            已导出的 ONNX 模型路径。
        device : torch.device
            期望运行设备；若 cuda 可用，onnxruntime-gpu 会被自动调用。
        **kwargs
            透传给 `ort.InferenceSession` 的 session_options / providers 等。
        """
        super().__init__(model_file, device)
        self.prepare()
        # 创建onnxruntime session
        self.ort_session = ort.InferenceSession(self.model_file)

    def prepare(self):
        """
        模型加载与校验流程。

        步骤
        ----
        1. 使用 onnx.checker 检查模型合法性。
        2. 打印输入/输出节点名称与形状，方便调试。
        """

        # 验证 ONNX 模型
        LOGGER.info("check onnx model...")
        onnx_model = onnx.load(self.model_file)
        onnx.checker.check_model(onnx_model)
        LOGGER.info("onnx model check passed!")

        # 检查模型结构
        input_info = [[i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]] for i in onnx_model.graph.input]
        output_info = [[o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]] for o in onnx_model.graph.output]

        print("Inputs:")
        print(tabulate(input_info, headers=["Name", "Shape"], tablefmt="grid"))

        print("Outputs:")
        print(tabulate(output_info, headers=["Name", "Shape"], tablefmt="grid"))

    def inference(self, data: torch.Tensor) -> list[torch.Tensor]:
        """
        执行一次前向推理。

        Args
        ----
        data : torch.Tensor
            输入张量，形状需与 ONNX 模型输入一致，通常为 (B, C, H, W)。

        Returns
        -------
        list[torch.Tensor]
            输出张量列表，每个元素对应 ONNX 模型的一个输出节点。
        """
        data = data.to("cpu").numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: data}
        ort_outs: list = self.ort_session.run(None, ort_inputs)
        return [torch.tensor(x) for x in ort_outs]
