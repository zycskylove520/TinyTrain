import torch

from TinyTrain.server.inference_server import BaseOnnxInferenceServer


class YOLOClassificationOnnxInferenceServer(BaseOnnxInferenceServer):
    def inference(self, data: torch.Tensor) -> list[torch.Tensor]:
        # predictor中data为[b,c,h,w]
        data = data.to("cpu").numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: data}
        ort_outs: list = self.ort_session.run(None, ort_inputs)
        return [torch.tensor(x) for x in ort_outs]
