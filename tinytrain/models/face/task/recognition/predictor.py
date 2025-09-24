import cv2
import torch
from typing import Any
from PIL import Image
from torchvision import transforms

from tinytrain.data.data_format import AnyDataInfo
from tinytrain.engine import BasePredictor
from tinytrain.utils.data_utils import cv_imread


class FaceRecognitionPredictor(BasePredictor):
    """
    人脸比对预测器。

    负责完成「两张人脸图像 → 特征向量 → 余弦相似度」的完整流水线：
    1. 预处理：读取、缩放、归一化、转 Tensor
    2. 推理：分别提取两张图的人脸特征
    3. 后处理：计算余弦相似度
    4. 可视化：返回结构化结果
    """

    def __init__(self, config_manager, device, model, callback, backend=None, **kwargs):
        """
        Args
        ----
        config_manager : ConfigManager
            全局配置管理器
        device : torch.device
            模型运行的设备
        model : BaseModel
            已加载的人脸特征提取模型
        callback : Callable
            推理完成后的回调函数
        backend : str, optional
            推理后端标识，默认 None
        **kwargs
            其余关键字参数，必须包含 img_shape
        """
        super().__init__(config_manager=config_manager, device=device, model=model, callback=callback, backend=backend, **kwargs)

        self.img_shape = kwargs.get("img_shape")
        if self.img_shape is None:
            raise ValueError("img_shape must be set")

    def preprocess(self, data_info: AnyDataInfo) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将两张人脸图像转换为模型所需的 Tensor 格式。

        Args
        ----
        data_info : AnyDataInfo
            data_info.data 约定为长度为 2 的列表，
            每个元素为图像路径或 ndarray，颜色顺序 BGR。

        Returns
        ----
        tuple[Tensor, Tensor]
            (tensor1, tensor2)，形状均为 [1, C, H, W]，已放置到 self.device
        """
        img_list = data_info.data
        img1 = cv_imread(img_list[0])
        img2 = cv_imread(img_list[1])

        rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb_img1 = Image.fromarray(rgb_img1)

        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        rgb_img2 = Image.fromarray(rgb_img2)

        tf_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ]
        if self.img_shape is not None:
            tf_list.insert(0, transforms.Resize(self.img_shape))
        transform = transforms.Compose(tf_list)

        tensor1 = transform(rgb_img1).unsqueeze(0).to(self.device)  # [1,C,H,W]
        tensor2 = transform(rgb_img2).unsqueeze(0).to(self.device)
        return tensor1, tensor2

    def inference(self, data: Any) -> Any:
        """
        分别对两张图像提取人脸特征向量。

        Args
        ----
        data : tuple[Tensor, Tensor]
            preprocess 返回的两个 Tensor

        Returns
        ----
        list[Tensor]
            两张图对应的特征向量，每个形状 [1, D]
        """
        inference_result = []
        for tensor in data:
            result = self.model.inference(tensor)[0]  # 拿推理后的第一个输出
            inference_result.append(result)
        return inference_result

    # ---------- 后处理 ----------
    def postprocess(self, data_info: AnyDataInfo, preds: list[torch.Tensor]):
        """
        计算两张人脸特征的余弦相似度。

        Args
        ----
        data_info : AnyDataInfo
            原始数据信息，预留接口
        preds : list[Tensor]
            inference 返回的两个特征向量，形状均为 [1, D]

        Returns
        ----
        tuple[float, list[Tensor]]
            (cosine_similarity, preds)
        """
        # 计算余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(preds[0], preds[1], dim=1).item()
        return cosine_similarity, preds

    # ---------- 可视化 ----------
    def show(self, data_info: AnyDataInfo, result):
        """
        将比对结果封装为易读的字典。

        Args
        ----
        data_info : AnyDataInfo
            原始数据信息，预留接口
        result : tuple[float, list[Tensor]]
            postprocess 返回的 (cosine_similarity, preds)

        Returns
        ----
        dict
            {
                "cosine_similarity": float,
                "img1_feature_vector": Tensor,
                "img2_feature_vector": Tensor
            }
        """
        return {"cosine_similarity": result[0], "img1_feature_vector": result[1][0], "img2_feature_vector": result[1][1]}
