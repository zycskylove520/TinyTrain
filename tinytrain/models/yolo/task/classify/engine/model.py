import torch

from tinytrain.data.data_format import ClassifyBatchDataInfo
from tinytrain.loss import ClassificationLoss
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLOClassificationModel(YOLOModel):
    """
    YOLOClassificationModel

    在 YOLO 检测骨架基础上，快速适配图像分类任务的专用子类。

    设计要点
    --------
    1. 复用 YOLOModel 的 backbone+neck（可选）结构，仅将 head 替换为 ClassificationHead。
    2. 损失函数采用 ClassificationLoss，支持类别权重与增益调节。
    3. 通过 custom_parse_model 钩子对 C3k2 / A2C2f 模块进行深度增益与分支开关的自动缩放，
       保证 n/s/m/l/x 五个尺度在分类任务上的一致性与最佳性价比。
    4. 对外接口与 TTBaseModel 保持一致：forward(data) 自动区分训练/推理模式。
    """

    def init_criterion(self):
        """
        构建分类损失实例。

        Returns
        -------
        ClassificationLoss
            损失权重由配置文件 `loss.cls_loss_gain` 控制。
        """
        return ClassificationLoss(self.config_manager.loss["cls_loss_gain"])

    def loss(self, preds: list[torch.Tensor], batch_samples: ClassifyBatchDataInfo) -> tuple[float, dict]:
        """
        训练模式：计算分类损失。

        Args
        ----
        preds : list[torch.Tensor]
            仅使用 preds[0]（logits，形状 [B, C]）。
        batch_samples : ClassifyBatchDataInfo
            必须包含 .label 字段（类别索引，形状 [B]）。

        Returns
        -------
        tuple[float, dict]
            (总损失, 详细分量字典)，
            字典固定包含 key = "cls_loss"，value 为 float。
        """
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, level, module_info):
        """
        分类任务专属解析逻辑，在通用解析完成后、模块实例化前调用。

        调整策略
        --------
        1. 若为 `C3k2` 模块：
           - 根据 depth 增益重算 `n`（Bottleneck 堆叠次数）。
           - 仅当 scale ∈ {"m", "l", "x"} 时开启 `c3k` 分支，提升大模型容量。
        2. 若为 `A2C2f` 模块：
           - 同样按 depth 缩放 `n`。
           - 在 l/x 尺度额外开启 residual 与 mlp_ratio=1.2，增强表达能力。

        Args
        ----
        level : int
            当前正在解析的层序号。
        module_info : dict
            当前层原始配置，函数内直接原地修改。
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
