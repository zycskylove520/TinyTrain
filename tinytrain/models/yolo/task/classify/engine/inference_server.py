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

import torch

from tinytrain.server.inference_server import TTBaseOnnxInferenceServer


class YOLOClassificationOnnxInferenceServer(TTBaseOnnxInferenceServer):
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
