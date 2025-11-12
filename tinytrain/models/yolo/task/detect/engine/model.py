"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

from tinytrain.cfg import TTConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.loss import YOLOV8DetectionLoss
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLODetectionModel(YOLOModel):
    """
    YOLODetectionModel

    专为目标检测任务设计的 YOLO 子类，继承自 YOLOModel。

    设计要点
    --------
    1. 在构造阶段即完成 stride 的自动推算，避免后续训练/推理阶段重复计算。
    2. 采用 YOLOV8DetectionLoss，支持 cls / box / dfl 三项损失加权。
    3. 通过 custom_parse_model_level 对 C3k2 / A2C2f / YOLODetectHead 进行宽度&深度增益缩放，
       并依据尺度 {n, s, m, l, x} 开启不同高级分支（c3k、residual、mlp_ratio）。
    4. 对外保持 TTBaseModel 统一接口：forward(data) 自动返回 loss 或推理结果。
    """

    def __init__(self, config_manager: TTConfigManager, device):
        """
        初始化检测模型。

        额外完成：
        - 推算并保存三个检测头的输出步长 strides；
        - 对 YOLODetectHead 头进行 bias 初始化；
        - 主动调用 initialize_weights()，确保 Conv / BN / Act 参数处于最佳初始分布。

        Args
        ----
        config_manager : TTConfigManager
            必须包含 dataset.nc / dataset.img_size / loss.* 等字段。
        device : torch.device
            模型所在设备。
        """
        self.reg_max = None
        super().__init__(config_manager, device=device)
        self.initialize_weights()

        input_channel = config_manager.model["network"][0]["args"]["in_channels"]
        m = self.module_list[-1]  # YOLODetectHead()

        # 初始化 strides
        self.strides = m.strides = self._initialize_stride(input_channel)
        m.bias_init(self.strides)

    def _initialize_stride(self, input_channel):
        """
        通过一次 dummy forward 推算三个检测头的输出步长，避免在后续训练阶段反复计算。

        Args
        ----
        input_channel : int
            输入图片的通道数（通常为 3）。

        Returns
        -------
        torch.Tensor
            长度 = 检测头数量，每项对应该头输出的步长（下采样倍数）。
        """
        stride = 256  # 2x min stride
        with torch.no_grad():  # 确保不会改变模型状态
            # 模拟一张 256x256 的图片
            dummy_input = torch.zeros(1, input_channel, stride, stride)
            # 前向传播，获取输出
            outputs = self.forward(dummy_input)[0]
            # 计算 stride
            stride_tensor = torch.tensor([stride / x.shape[-2] for x in outputs])
        return stride_tensor

    def init_criterion(self):
        """
        构建检测专用损失实例。

        Returns
        -------
        YOLOV8DetectionLoss
            已内置 strides / reg_max / imgsz / device 等信息，
            可直接前向计算 cls_loss / box_loss / dfl_loss。
        """
        return YOLOV8DetectionLoss(nc=self.config_manager.dataset["nc"],
                                   strides=self.strides,
                                   reg_max=self.reg_max,
                                   imgsz=self.config_manager.dataset["img_size"],
                                   device=self.device,
                                   cls_gain=self.config_manager.loss["cls_loss_gain"],
                                   box_gain=self.config_manager.loss["box_loss_gain"],
                                   dfl_gain=self.config_manager.loss["dfl_loss_gain"]
                                   )

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        """
        训练模式：计算检测损失。

        Args
        ----
        preds : list[torch.Tensor]
            仅使用 preds[0]（YOLODetectHead 输出列表）。
        batch_samples : BaseBatchDataInfo
            必须包含 .bboxes / .labels / .masks(可选) 等字段。

        Returns
        -------
        tuple[float, dict]
            (总损失, 详细分量字典)，
            字典固定包含 keys = {"cls_loss", "box_loss", "dfl_loss"}。
        """
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model_level(self, level, module_info):
        """
        检测任务专属解析逻辑，在通用解析完成后、模块实例化前调用。

        调整策略
        --------
        1. C3k2：
           - 按 depth 增益重算 n；
           - 仅当 scale ∈ {m, l, x} 时开启 c3k 分支。
        2. A2C2f：
           - 同样按 depth 缩放 n；
           - 在 l/x 尺度额外开启 residual 与 mlp_ratio=1.2。
        3. YOLODetectHead：
           - 按 width 增益缩放 from_channels；
           - 保存 reg_max 供损失函数使用。

        Args
        ----
        level : int
            当前正在解析的层序号。
        module_info : dict
            当前层原始配置，函数内直接原地修改。
        """
        scale = self.config_manager.model["scale"]

        width = self.WIDTH_GAIN
        depth = self.DEPTH_GAIN

        if module_info["module"] == "C3k2":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有mlx型号开启c3k模块
            if scale in {"m", "l", "x"}:
                module_info["args"]["c3k"] = True

        if module_info["module"] == "A2C2f":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有lx型号新增参数
            if scale in {"l", "x"}:
                module_info["args"]["residual"] = True
                module_info["args"]["mlp_ratio"] = 1.2

        if module_info["module"] == "YOLODetectHead":
            module_info["args"]["from_channels"] = [int(i * width) for i in module_info["args"]["from_channels"]]
            self.reg_max = module_info["args"].get("reg_max", 16)
