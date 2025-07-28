import onnx
import onnxruntime as ort
import torch

from tabulate import tabulate

from .base_inference_server import BaseInferenceServer
from ...utils import LOGGER


class BaseOnnxInferenceServer(BaseInferenceServer):
    def __init__(self, model_file, device, **kwargs):
        super().__init__(model_file, device)
        self.prepare()
        # 创建onnxruntime session
        self.ort_session = ort.InferenceSession(self.model_file)

    def prepare(self):
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
        data = data.to("cpu").numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: data}
        ort_outs: list = self.ort_session.run(None, ort_inputs)
        return [torch.tensor(x) for x in ort_outs]
