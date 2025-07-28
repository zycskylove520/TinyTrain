from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.loss import ClassificationLoss
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLOClassificationModel(YOLOModel):
    def __init__(self, config_manager: ConfigManager):
        super(YOLOClassificationModel, self).__init__(config_manager)

    def init_criterion(self):
        return ClassificationLoss(self.config_manager.loss["cls_loss_gain"])

    def custom_parse_model(self, module_info):
        scale = self.config_manager.model["scale"]
        scale_info = self.config_manager.model["scales"][scale]
        depth = scale_info["depth"]
        if module_info["module"] == "C3k2":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有mlx型号开启c3k模块
            if scale in "mlx":
                module_info["args"]["c3k"] = True
