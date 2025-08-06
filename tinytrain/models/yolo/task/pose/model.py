import torch

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.loss.loss import YOLOV8DetectionLoss, YOLOV8PoseLoss
from tinytrain.models.yolo.task.detect import YOLODetectionModel
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLOPoseModel(YOLODetectionModel):
    def __init__(self, config_manager: ConfigManager, *args, **kwargs):
        super().__init__(config_manager, *args, **kwargs)
        self.initialize_weights()

        input_channel = config_manager.model["network"][0]["args"]["in_channels"]
        m = self.module_list[-1]  # Detect()

        # 初始化 stride
        self.stride = m.stride = self._initialize_stride(input_channel)

    def _initialize_stride(self, input_channel):
        """
        初始化 stride，避免在初始化阶段调用 forward 方法。
        """
        stride = 256  # 2x min stride
        with torch.no_grad():  # 确保不会改变模型状态
            # 模拟一张 256x256 的图片
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, input_channel, stride, stride, device=device)
            # 前向传播，获取输出
            outputs = self.forward(dummy_input)[0][0]
            # 计算 stride
            stride_tensor = torch.tensor([stride / x.shape[-2] for x in outputs])
        return stride_tensor

    def init_criterion(self):
        return YOLOV8PoseLoss(self,
                              self.config_manager.dataset["img_size"],
                              self.config_manager.loss["cls_loss_gain"],
                              self.config_manager.loss["box_loss_gain"],
                              self.config_manager.loss["dfl_loss_gain"],
                              self.config_manager.loss["pose_loss_gain"],
                              self.config_manager.loss["kobj_loss_gain"],
                              )

    def custom_parse_model(self, module_info):
        scale = self.config_manager.model["scale"]

        width = YOLOModel.WIDTH_GAIN
        depth = YOLOModel.DEPTH_GAIN

        if module_info["module"] == "C3k2":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有mlx型号开启c3k模块
            if scale in {"m", "l", "x"}:
                module_info["args"]["c3k"] = True

        if module_info["module"] == "YOLOPose":
            module_info["args"]["from_channels"] = [int(i * width) for i in module_info["args"]["from_channels"]]
            module_info["args"]["kpt_shape"] = self.config_manager.dataset["keypoint_shape"]
