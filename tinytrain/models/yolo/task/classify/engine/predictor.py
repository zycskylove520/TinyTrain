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
import cv2

from PIL import Image
from torchvision import transforms

from tinytrain.data.data_format import ClassifyDataInfo
from tinytrain.engine.predictor import TTBasePredictor
from tinytrain.utils.source_loader import ImageParser, VideoParser, SourceParserHub


class YOLOClassificationPredictor(TTBasePredictor):
    """
    图像分类预测器（兼容通用 TTBasePredictor）
    输入：单张图片 / 视频 / 目录 / URL / 文本清单 … 均可
    输出：每张图片的 logits 或 softmax 概率
    """

    def __init__(self,
                 config_manager,
                 device,
                 model,
                 callback,
                 backend=None,
                 **kwargs
                 ):
        super().__init__(config_manager=config_manager, device=device, model=model, callback=callback, backend=backend, **kwargs)
        self.img_shape = kwargs.get("img_shape")

        if self.img_shape is None:
            raise ValueError("img_shape must be set")

    def register_parsers(self) -> None:
        # ---------- 注册专用解析器 ----------
        SourceParserHub.register("jpg", ImageParser)
        SourceParserHub.register("jpeg", ImageParser)
        SourceParserHub.register("png", ImageParser)
        SourceParserHub.register("bmp", ImageParser)
        SourceParserHub.register("mp4", VideoParser)
        SourceParserHub.register("avi", VideoParser)
        SourceParserHub.register("mov", VideoParser)

    # ---------- 数据前处理 ----------
    def preprocess(self, data_info: ClassifyDataInfo) -> torch.Tensor:
        """
        sample: 由 SourceParser 给出的任意对象
                这里约定为 np.ndarray [H,W,3] BGR
        """
        rgb_img = cv2.cvtColor(data_info.img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img)

        tf_list = [transforms.ToTensor(), transforms.Normalize(mean=0, std=1)]
        if self.img_shape is not None:
            tf_list.insert(0, transforms.Resize(self.img_shape))
        transform = transforms.Compose(tf_list)

        tensor = transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    # ---------- 后处理 ----------
    def postprocess(self, data_info: ClassifyDataInfo, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        preds: list[Tensor] 来自推理后端
        返回: [num_classes] 的 logits
        """
        logits = preds[0].squeeze(0)  # [B, num_classes] -> [num_classes]
        return logits

    # ---------- 可视化 ----------
    def show(self, data_info: ClassifyDataInfo, result: torch.Tensor):
        prob = torch.softmax(result, dim=0)
        pred_cls = int(prob.argmax())
        return {"class_idx": pred_cls, "per_class_probability": prob.tolist()}
