import torch

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data import ClassifyBatchDataInfo
from tinytrain.engine import BaseModel
from tinytrain.loss.loss import FocalLoss


class FaceRecognitionModel(BaseModel):
    """
    人脸识别模型。
    """

    def __init__(self, config_manager: ConfigManager, *args, **kwargs):
        self.features = None
        super().__init__(config_manager, *args, **kwargs)
        self.head = self.module_list[-1]

    def init_criterion(self):
        """
        返回人脸识别损失实例。

        Returns
        -------
            损失权重由配置文件 `loss.cls_loss_gain` 控制。
        """
        cls_loss_gain = self.config_manager.loss["cls_loss_gain"]
        return FocalLoss(cls_loss_gain=cls_loss_gain)

    def loss(self, preds: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, module_info):
        nc = self.config_manager.dataset["nc"]

        if module_info["module"] in {"ArcFace", "CosFace", "SphereFace"}:
            self.features = module_info["args"]["nc"]= nc
