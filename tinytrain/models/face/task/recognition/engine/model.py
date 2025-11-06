import torch

from tinytrain.cfg import TTConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.engine import TTConfigModel
from tinytrain.global_var import WORLD_SIZE
from tinytrain.models.face.face_loss import PartialFCLoss, FCLoss
from tinytrain.models.face.task.recognition.margin import CombinedMargin


class FaceRecognitionModel(TTConfigModel):
    """
    人脸识别专用模型，继承自 TTConfigModel。

    职责：
    1. 根据配置动态选用 MobileFaceNet / YOLOv11-FaceNet / ResFaceNet 等骨架。
    2. 组合 CombinedMargin + PartialFC 作为训练损失。
    3. 通过 custom_parse_model_level 钩子对不同尺度（n/s/m/l/x）自动缩放通道数、深度及模块开关。
    4. 统一对外提供 forward → loss / inference 两种调用模式，无需关心底层细节。

    设计要点：
    - 所有网络差异均由配置文件驱动，子类仅需关注“如何根据 scale 微调模块参数”。
    - init_criterion 返回的是 PartialFCLoss，已内置 margin-based 分类损失。
    - custom_parse_model_level 中仅原地修改 module_info，不返回任何值；解析完成后由基类统一构建网络。
    """
    def init_criterion(self):
        """
        构建人脸识别专用损失函数（PartialFC + CombinedMargin）。

        Returns:
            PartialFCLoss: 已部署到指定设备，可直接前向计算。
        """
        margin_loss = CombinedMargin(
            s=self.config_manager.loss["s"],
            m_arc=self.config_manager.loss["m_arc"],
            m_cos=self.config_manager.loss["m_cos"],
            interclass_filtering_threshold=self.config_manager.loss["interclass_filtering_threshold"],
        )

        if WORLD_SIZE > 1:
            return PartialFCLoss(
                margin_loss=margin_loss,
                device=self.device,
                embedding_size=self.config_manager.loss["embedding_size"],
                num_classes=self.config_manager.dataset["nc"],
                sample_rate=self.config_manager.loss["sample_rate"],
                cls_loss_gain=self.config_manager.loss["cls_loss_gain"]
            )
        else:
            return FCLoss(
                margin_loss=margin_loss,
                device=self.device,
                embedding_size=self.config_manager.loss["embedding_size"],
                num_classes=self.config_manager.dataset["nc"],
                cls_loss_gain=self.config_manager.loss["cls_loss_gain"]
            )

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        """
        训练模式：计算人脸识别损失。

        Args:
            preds (list[torch.Tensor]): 模型输出，仅使用 preds[0]（embedding）。
            batch_samples (BaseBatchDataInfo): 必须包含 .label 字段（全局唯一身份 ID）。

        Returns:
            tuple[float, dict]: (总损失, 详细分量字典)，
                                字典固定包含 key = "cls_loss"，value 为 float。
        """
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model_level(self, layer, module_info):
        """
        根据不同网络与尺度，原地修改模块配置（通道数、重复次数、开关等）。

        Args:
            layer (int): 当前正在解析的层序号。
            module_info (dict): 当前层原始配置，函数内直接原地修改。
        """
        name = self.config_manager.model["name"]
        scale = self.config_manager.model["scale"]
        scales = self.config_manager.model["scales"]
        depth = self.DEPTH_GAIN

        if name == "MobileFaceNet":
            # MobileFaceNet：按尺度索引线性扩增通道，GDCHead 层固定输出 embedding_size。
            for i, _scale in enumerate(scales):
                i += 1
                expand = i
                if scale == _scale:
                    if module_info["type"] == "entry":
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "flow":
                        module_info["args"]["in_channels"] *= expand
                        module_info["args"]["out_channels"] *= expand
                    elif module_info["type"] == "head":
                        module_info["args"]["in_channels"] *= expand

            if module_info["module"] == "GDCHead":
                module_info["args"]["embedding_size"] = self.config_manager.loss["embedding_size"]
        elif name == "YOLOv11_FaceNet":
            # YOLOv11_FaceNet：通道扩增系数 = 尺度索引*2；C3k2 模块深度随 depth gain 缩放，仅 scale=m 时开启 c3k 分支。
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

            if module_info["module"] == "GeneralFaceHead":
                module_info["args"]["embedding_size"] = self.config_manager.loss["embedding_size"]
        elif name == "ResFaceNet":
            # ResFaceNet：四个 ResNetLayer 重复次数完全由 scale 决定；GDCHead 同 MobileFaceNet。
            if scale == "n":
                if layer == 1 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 2
                elif layer == 2 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 2
                elif layer == 3 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 2
                elif layer == 4 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 2
            elif scale == "s":
                if layer == 1 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
                elif layer == 2 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 4
                elif layer == 3 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 6
                elif layer == 4 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
            elif scale == "m":
                if layer == 1 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
                elif layer == 2 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 4
                elif layer == 3 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 14
                elif layer == 4 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
            elif scale == "l":
                if layer == 1 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
                elif layer == 2 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 13
                elif layer == 3 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 30
                elif layer == 4 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 3
            elif scale == "x":
                if layer == 1 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 6
                elif layer == 2 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 26
                elif layer == 3 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 60
                elif layer == 4 and module_info["type"] == "flow" and module_info["module"] == "ResNetLayer":
                    module_info["args"]["n"] = 6
            if module_info["module"] == "GDCHead":
                module_info["args"]["embedding_size"] = self.config_manager.loss["embedding_size"]
