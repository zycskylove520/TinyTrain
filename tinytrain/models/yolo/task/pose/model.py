import torch

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.loss.loss import YOLOV8PoseLoss
from tinytrain.models.yolo.yolo_model import YOLOModel


class YOLOPoseModel(YOLOModel):
    def __init__(self, config_manager: ConfigManager, device, *args, **kwargs):
        self.reg_max = None
        super().__init__(config_manager=config_manager, device=device, *args, **kwargs)
        self.initialize_weights()

        input_channel = config_manager.model["network"][0]["args"]["in_channels"]
        m = self.module_list[-1]  # YOLOPose()

        # 初始化 stride
        self.strides = m.strides = self._initialize_stride(input_channel)

    def _initialize_stride(self, input_channel):
        """
        初始化 stride，避免在初始化阶段调用 forward 方法。
        """
        stride = 256  # 2x min stride
        with torch.no_grad():  # 确保不会改变模型状态
            # 模拟一张 256x256 的图片
            dummy_input = torch.zeros(1, input_channel, stride, stride)
            # 前向传播，获取输出
            outputs = self.forward(dummy_input)[0][0]
            # 计算 stride
            stride_tensor = torch.tensor([stride / x.shape[-2] for x in outputs])
        return stride_tensor

    def init_criterion(self):
        return YOLOV8PoseLoss(nc=self.config_manager.dataset["nc"],
                              strides=self.strides,
                              reg_max=self.reg_max,
                              imgsz=self.config_manager.dataset["img_size"],
                              device=self.device,
                              kpt_shape=self.config_manager.dataset["keypoint_shape"],
                              cls_gain=self.config_manager.loss["cls_loss_gain"],
                              box_gain=self.config_manager.loss["box_loss_gain"],
                              dfl_gain=self.config_manager.loss["dfl_loss_gain"],
                              pose_gain=self.config_manager.loss["pose_loss_gain"],
                              kobj_gain=self.config_manager.loss["kobj_loss_gain"],
                              )

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model(self, level, module_info):
        scale = self.config_manager.model["scale"]

        width = self.WIDTH_GAIN
        depth = self.DEPTH_GAIN

        if module_info["module"] == "C3k2":
            n = max(round(module_info["args"]["n"] * depth), 1)
            module_info["args"]["n"] = n

            # 只有mlx型号开启c3k模块
            if scale in {"m", "l", "x"}:
                module_info["args"]["c3k"] = True

        if module_info["module"] == "YOLOPose":
            module_info["args"]["from_channels"] = [int(i * width) for i in module_info["args"]["from_channels"]]
            module_info["args"]["kpt_shape"] = self.config_manager.dataset["keypoint_shape"]
            self.reg_max = module_info["args"].get("reg_max", 16)
