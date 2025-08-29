from typing import Any

import cv2
import torch

from PIL import Image
from torchvision import transforms

from tinytrain.data import AnyDataInfo
from tinytrain.engine import BasePredictor, BaseModel
from tinytrain.utils.data_utils import cv_imread


class FaceRecognitionPredictor(BasePredictor):
    def __init__(self,
                 config_manager,
                 model,
                 callback,
                 backend=None,
                 **kwargs):
        super().__init__(config_manager=config_manager, model=model, callback=callback, backend=backend, **kwargs)

        # 设置模型head层为export模式
        if isinstance(self.model, BaseModel):
            self.model.module_list[-1].export = True
        self.img_shape = kwargs.get("img_shape")
        if self.img_shape is None:
            raise ValueError("img_shape must be set")

    def preprocess(self, data_info: AnyDataInfo) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample: 由 SourceParser 给出的任意对象
                这里约定为 np.ndarray [H,W,3] BGR
        """
        img_list = data_info.data
        img1 = cv_imread(img_list[0])
        img2 = cv_imread(img_list[1])

        rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb_img1 = Image.fromarray(rgb_img1)

        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        rgb_img2 = Image.fromarray(rgb_img2)

        tf_list = [transforms.ToTensor(), transforms.Normalize(mean=0, std=1)]
        if self.img_shape is not None:
            tf_list.insert(0, transforms.Resize(self.img_shape))
        transform = transforms.Compose(tf_list)

        tensor1 = transform(rgb_img1).unsqueeze(0).to(self.device)  # [1,C,H,W]
        tensor2 = transform(rgb_img2).unsqueeze(0).to(self.device)
        return tensor1, tensor2

    def inference(self, data: Any) -> Any:
        inference_result = []
        for tensor in data:
            result = self.model.inference(tensor)[0]  # 拿推理后的第一个输出
            inference_result.append(result)
        return inference_result

    # ---------- 后处理 ----------
    def postprocess(self, data_info: AnyDataInfo, preds: list[torch.Tensor]):
        # 计算余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(preds[0], preds[1], dim=1).item()
        return cosine_similarity, preds

    # ---------- 可视化 ----------
    def show(self, data_info: AnyDataInfo, result):
        return {"cosine_similarity": result[0], "img1_feature_vector": result[1][0], "img2_feature_vector": result[1][1]}
