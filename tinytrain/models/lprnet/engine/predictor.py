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

from tinytrain.models.lprnet.data_format import LPRDataInfo
from tinytrain.engine.predictor import TTBasePredictor


class LPRPredictor(TTBasePredictor):
    """
    图像分类预测器（兼容通用 TTBasePredictor）
    输入：单张图片 / 视频 / 目录 / URL / 文本清单 … 均可
    输出：每张图片的 logits 或 softmax 概率
    """

    def __init__(self,config_manager,device,model,callback,backend=None,**kwargs):
        super().__init__(config_manager=config_manager, device=device, model=model, callback=callback, backend=backend, **kwargs)
        self.chars_dict = {int(idx): char for idx, char in self.config_manager.dataset["names"].items()}

    # ---------- 数据前处理 ----------
    def preprocess(self, data_info: LPRDataInfo) -> torch.Tensor:
        """
        sample: 由 SourceParser 给出的任意对象
                这里约定为 np.ndarray [H,W,3] BGR
        """
        rgb_img = cv2.cvtColor(data_info.img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img)

        transform = transforms.Compose([
            transforms.Resize((24, 94)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1)
        ])

        tensor = transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    # ---------- 后处理 ----------
    def postprocess(self, data_info: LPRDataInfo, preds: list[torch.Tensor]) -> str:
        """
        preds: list[Tensor] 来自推理后端
        返回: [num_classes] 的 logits
        """
        logits = preds[0].squeeze(0)  # [T, C]
        plate_str = self.ctc_greedy_decode(logits, blank_id=len(self.chars_dict) - 1)
        return plate_str

    # ---------- 可视化 ----------
    def show(self, data_info: LPRDataInfo, result: str):
        return {"plate": result}

    def ctc_greedy_decode(self, logits: torch.Tensor, blank_id: int) -> str:
        """
        logits: (num_classes, T)  未归一化得分
        return: 字符串
        """
        best_path = torch.argmax(logits, dim=0)  # T
        dedup = []
        prev = blank_id
        for label in best_path:
            if label == blank_id:  # 去 blank
                prev = blank_id
                continue
            if label == prev:  # 去重复
                continue
            dedup.append(label.cpu().item())
            prev = label
        return ''.join([self.chars_dict[i] for i in dedup])
