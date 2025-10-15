import torch

from tinytrain.engine import TTBaseModel
from tinytrain.loss import LPRCTCLoss
from tinytrain.models.ocr.ocr_data_format import LPRBatchDataInfo


class LPRModel(TTBaseModel):
    """LPRNet 车牌识别模型"""

    def init_criterion(self):
        return LPRCTCLoss(
            max_time_steps=self.config_manager.loss["max_time_steps"],
            blank=self.config_manager.dataset["nc"] - 1
        )

    def loss(self, preds: torch.Tensor, batch_samples: LPRBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, level, module_info):
        if self.config_manager.model["name"] == "LPRNet":
            if module_info["module"] == "LPRBackbone":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
                module_info["args"]["dropout_rate"] = self.config_manager.loss["dropout_rate"]

            if module_info["module"] == "LPRHead":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
