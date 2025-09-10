import torch

from tinytrain.server.inference_server import BaseOnnxInferenceServer


class YOLOClassificationOnnxInferenceServer(BaseOnnxInferenceServer):
    def inference(self, data: torch.Tensor):
        """
        单张或多张图片的 ONNX 前向推理。

        参数
        ----
        data : torch.Tensor
            输入张量，形状 `(B,C,H,W)`，已在 predictor 中完成归一化与 resize。

        返回
        ----
        torch.Tensor
            分类 logits，形状 `(B,num_classes)`。
        """
        # predictor中data为[b,c,h,w]
        data = data.to("cpu").numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: data}
        ort_outs: list = self.ort_session.run(None, ort_inputs)
        return torch.tensor(ort_outs[0])
