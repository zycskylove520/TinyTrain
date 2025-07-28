import torch

from typing import Any
from PIL import Image
from torchvision import transforms

from tinytrain.data import ClassifyDataInfo
from tinytrain.engine.predictor import BasePredictor
from tinytrain.utils.source_loader import ImageParser, VideoParser, SourceParserHub


class YOLOClassificationPredictor(BasePredictor):
    """
    图像分类预测器（兼容通用 BasePredictor）
    输入：单张图片 / 视频 / 目录 / URL / 文本清单 … 均可
    输出：每张图片的 logits 或 softmax 概率
    """

    def __init__(self,
                 config_manager,
                 model,
                 callback,
                 backend=None,
                 **kwargs
                 ):
        super().__init__(config_manager=config_manager, model=model, callback=callback, backend=backend, **kwargs)
        self.img_shape = kwargs.get("img_shape")

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
                这里约定为 np.ndarray [H,W,3] RGB
        """
        img = Image.fromarray(data_info.img)

        tf_list = [transforms.ToTensor(), transforms.Normalize(mean=0, std=1)]
        if self.img_shape is not None:
            tf_list.insert(0, transforms.Resize(self.img_shape))
        transform = transforms.Compose(tf_list)

        tensor = transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    # ---------- 后处理 ----------
    def postprocess(self, sample: Any, preds: list[torch.Tensor]) -> torch.Tensor:
        """
        preds: list[Tensor] 来自推理后端
        返回: [num_classes] 的 logits
        """
        logits = preds[0].squeeze(0)  # [B, num_classes] -> [num_classes]
        return logits

    # ---------- 可视化 ----------
    def show(self, sample: Any, result: torch.Tensor):
        prob = torch.softmax(result, dim=0)
        pred_cls = int(prob.argmax())
        return {"class_idx": pred_cls, "per_class_probability": prob.tolist()}
