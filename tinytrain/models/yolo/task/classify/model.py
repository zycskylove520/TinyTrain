import torch

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.loss import ClassificationLoss
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLOClassificationModel(YOLOModel):
    """
    YOLO 图像分类模型。
    """

    def init_criterion(self):
        """
        返回分类损失实例。

        Returns
        -------
        ClassificationLoss
            损失权重由配置文件 `loss.cls_loss_gain` 控制。
        """
        return ClassificationLoss(self.config_manager.loss["cls_loss_gain"])

    def loss(self, preds: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, level, module_info):
        """
        分类任务专属解析逻辑，在通用解析完成后、模块实例化前调用。

        调整策略
        --------
        1. 若为 `C3k2` 模块：
           - 根据 depth 增益重算 `n`（Bottleneck 堆叠次数）。
           - 仅当 scale ∈ {"m", "l", "x"} 时开启 `c3k` 分支，提升大模型容量。
        """
        scale = self.config_manager.model["scale"]
        scale_info = self.config_manager.model["scales"][scale]
        depth = scale_info["depth"]
        if module_info["module"] == "C3k2":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有mlx型号开启c3k模块
            if scale in "mlx":
                module_info["args"]["c3k"] = True

        if module_info["module"] == "A2C2f":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有lx型号新增参数
            if scale in {"l", "x"}:
                module_info["args"]["residual"] = True
                module_info["args"]["mlp_ratio"] = 1.2
