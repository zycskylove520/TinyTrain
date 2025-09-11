import torch

from tinytrain.cfg import ConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.engine import BaseModel
from tinytrain.models.face.face_loss import PartialFCLoss
from tinytrain.models.face.task.recognition.margin import CombinedMargin


class FaceRecognitionModel(BaseModel):
    """
    人脸识别模型。
    """

    def __init__(self, config_manager: ConfigManager, device, *args, **kwargs):
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
            embedding_size=self.config_manager.loss["embedding_size"],
            num_classes=self.config_manager.dataset["nc"],
            sample_rate=self.config_manager.loss["sample_rate"],
            cls_loss_gain=self.config_manager.loss["cls_loss_gain"])

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, layer, module_info):
        name = self.config_manager.model["name"]
        scale = self.config_manager.model["scale"]
        scales = self.config_manager.model["scales"]
        depth = self.DEPTH_GAIN

        if name == "MobileFaceNet":
            for i, _scale in enumerate(scales):
                i+=1
                expand = i
                if scale == _scale:
                    if module_info["type"] == "entry":
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "flow":
                        module_info["args"]["in_channels"] *= expand
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "head":
                        module_info["args"]["in_channels"] *= expand

            if module_info["module"] == "GDC":
                module_info["args"]["embedding_size"] = self.config_manager.loss["embedding_size"]
        elif name == "YOLOv11-face":
            for i, _scale in enumerate(scales):
                i += 1
                expand = i * 2
                if scale == _scale:
                    if module_info["type"] == "entry":
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "flow":
                        module_info["args"]["in_channels"] *= expand
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "head":
                        module_info["args"]["in_channels"] *= expand

                if module_info["module"] == "C3k2":
                    n = max(round(module_info["args"]["n"] * depth), 1)
                    module_info["args"]["n"] = n

                    # 只有m型号开启c3k模块
                    if scale == "m":
                        module_info["args"]["c3k"] = True

            if module_info["module"] == "GeneralFace":
                module_info["args"]["embedding_size"] = self.config_manager.loss["embedding_size"]
