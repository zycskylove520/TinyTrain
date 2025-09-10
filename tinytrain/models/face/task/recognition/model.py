from typing import Mapping, Any

import torch

from tinytrain.cfg import ConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.engine import BaseModel
from tinytrain.global_var import WORLD_SIZE
from tinytrain.models.face.face_loss import PartialFCLoss
from tinytrain.models.face.task.recognition.margin import CombinedMargin
from tinytrain.utils import LOGGER


class FaceRecognitionModel(BaseModel):
    """
    人脸识别模型。
    """

    def __init__(self, config_manager: ConfigManager, device, *args, **kwargs):
        self.embedding_size = None
        super().__init__(config_manager=config_manager, device=device, *args, **kwargs)

    def init_criterion(self):
        """
        返回人脸识别损失实例。
        """
        margin_loss = CombinedMargin(
            s=self.config_manager.loss["s"],
            m_arc=self.config_manager.loss["m_arc"],
            m_cos=self.config_manager.loss["m_cos"],
            interclass_filtering_threshold=self.config_manager.loss["interclass_filtering_threshold"],
        )
        return PartialFCLoss(
            margin_loss=margin_loss,
            device=self.device,
            embedding_size=self.embedding_size,
            num_classes=self.config_manager.dataset["nc"],
            sample_rate=self.config_manager.loss["sample_rate"],
            cls_loss_gain=self.config_manager.loss["cls_loss_gain"])

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, module_info):
        if module_info["type"] == "head" and module_info["module"] == "Conv2Linear":
            self.embedding_size = module_info["args"]["out_channels"]