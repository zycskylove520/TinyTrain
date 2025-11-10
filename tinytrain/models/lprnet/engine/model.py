import torch

from tinytrain.engine import TTConfigModel
from tinytrain.loss import LPRCTCLoss
from tinytrain.models.lprnet.data_format import LPRBatchDataInfo


class LPRModel(TTConfigModel):
    """LPRNet 车牌识别模型"""
    def init_criterion(self):
        return LPRCTCLoss(
            lpr_loss_gain=self.config_manager.loss['lpr_loss_gain'],
            blank=self.config_manager.dataset["nc"] - 1
        )

    def loss(self, preds: torch.Tensor, batch_samples: LPRBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model_level(self, level, module_info):
        if self.config_manager.model["name"] == "LPRNet":
            if module_info["module"] == "LPRBackbone":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
                module_info["args"]["dropout_rate"] = self.config_manager.loss["dropout_rate"]

            elif module_info["module"] == "LPRHead":
                module_info["args"]["in_channels"] = 448 + self.config_manager.dataset["nc"]
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]

        elif self.config_manager.model["name"] == "TTLPRNet":
            if module_info["module"] == "torch.nn.Dropout":
                module_info["args"]["p"] = self.config_manager.loss["dropout_rate"]

            elif module_info["module"] == "LPRHead":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
