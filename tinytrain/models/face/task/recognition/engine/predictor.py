import cv2
import torch
from typing import Any
from PIL import Image
from torchvision import transforms

from tinytrain.data.data_format import AnyDataInfo
from tinytrain.engine import TTBasePredictor
from tinytrain.utils.data_utils import cv_imread


class FaceRecognitionPredictor(TTBasePredictor):
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
        config_manager : TTConfigManager
            全局配置管理器
        device : torch.device
            模型运行的设备
        model : TTBaseModel
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

    def inference(self, data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        分别对两张图像提取人脸特征向量。

        Args
        ----
        data : tuple[torch.Tensor, torch.Tensor]
            preprocess 返回的两个 Tensor

        Returns
        ----
        tuple[torch.Tensor, torch.Tensor]
            两张图对应的特征向量，每个形状 [1, D]
        """
        x = torch.cat(data, dim=0)  # 2b
        result = self.model.inference(x)[0]  # 拿推理后的第一个输出
        pred1, pred2 = torch.chunk(result, 2, dim=0)
        return pred1, pred2

    # ---------- 后处理 ----------
    def postprocess(self, data_info: AnyDataInfo, preds: tuple[torch.Tensor, torch.Tensor]):
        """
        计算两张人脸特征的欧氏距离。

        Args
        ----
        data_info : AnyDataInfo
            原始数据信息，预留接口
        preds : list[Tensor]
            inference 返回的两个特征向量，形状均为 [1, D]

        Returns
        ----
        tuple[float, list[Tensor]]
            (distance, preds)
        """
        # 计算欧氏距离
        diff = preds[0] - preds[1]
        distance = torch.sum(diff * diff, dim=1).item()  # L2 距离
        return distance, preds

    # ---------- 可视化 ----------
    def show(self, data_info: AnyDataInfo, result):
        """
        将比对结果封装为易读的字典。

        Args
        ----
        data_info : AnyDataInfo
            原始数据信息，预留接口
        result : tuple[float, list[Tensor]]
            postprocess 返回的 (l2_distance, preds)

        Returns
        ----
        dict
            {
                "l2_distance": float,
                "img1_feature_vector": Tensor,
                "img2_feature_vector": Tensor
            }
        """
        return {"l2_distance": result[0], "img1_feature_vector": result[1][0], "img2_feature_vector": result[1][1]}
