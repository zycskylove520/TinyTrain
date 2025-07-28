from __future__ import annotations
from typing import TYPE_CHECKING

from TinyTrain.data import ImgDataInfo, DetectDataInfo
from TinyTrain.engine.predictor import BasePredictor
from TinyTrain.server.track_server.track_server_core import TrackServerCore
from TinyTrain.utils.box_utils import cxcywh_2_lxlyrxry

if TYPE_CHECKING:
    import torch


class YOLODetectionPredictor(BasePredictor):
    """
    检测预测器（兼容通用 BasePredictor）
    输入：图片/视频/目录等
    输出：DetectDataInfo（含 img + bboxes）
    """

    def __init__(self,
                 config_manager,
                 model,
                 callback,
                 backend=None,
                 **kwargs):
        super().__init__(config_manager, model, callback, backend, **kwargs)

        self.img_shape = kwargs.get("img_shape")

        # 绑定跟踪服务
        self.tracker_server = None
        if kwargs.get("track", False):
            assert isinstance(kwargs["track"], bool)
            track_backend = kwargs.get("track_backend", "bytetrack")
            self.tracker_server = TrackServerCore(config_manager=config_manager, callback=callback, backend=track_backend)

        # 注册解析器可以在 __init__ 里做，也可以放到首次调用时懒加载
        self.register_parsers()

    # ---------- 懒注册 ----------
    def register_parsers(self) -> None:
        from TinyTrain.utils.source_loader import ImageParser, VideoParser, SourceParserHub
        for ext in ("jpg", "jpeg", "png", "bmp"):
            SourceParserHub.register(ext, ImageParser)
        for ext in ("mp4", "avi", "mov"):
            SourceParserHub.register(ext, VideoParser)

    # ---------- 前处理 ----------
    def preprocess(self, data_info: ImgDataInfo) -> torch.Tensor:
        import torchvision.transforms as T
        from PIL import Image
        from TinyTrain.utils.checks import check_img_size
        import cv2

        if self.img_shape is None:
            data_info.target_shape = self.config_manager.dataset["img_size"] = check_img_size(data_info.origin_shape, 32)
        else:
            data_info.target_shape = self.img_shape

        transform = T.Compose([
            T.Resize(data_info.target_shape[::-1]),
            T.ToTensor(),
            T.Normalize(mean=0, std=1),
        ])
        img = Image.fromarray(cv2.cvtColor(data_info.img, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    # ---------- 后处理 ----------
    def postprocess(self, data_info: ImgDataInfo, inference_result: list[torch.Tensor]) -> DetectDataInfo:
        """
        inference_result: list[Tensor] 来自推理后端
        返回 DetectDataInfo（包含 img + bboxes）
        """
        from TinyTrain.utils.nms import detect_nms
        import numpy as np

        # detect_nms 输出: List[Tensor] 每张图的 [N,6] (x,y,w,h,conf,cls)
        dets = detect_nms(inference_result[0], conf_threshold=0.25, nms_threshold=0.5)[0]  # [N,6]

        # bboxes恢复原图尺寸，并转为lxlyrxry
        bboxes = dets[:, :4].cpu().numpy()
        target_w, target_h = data_info.target_shape
        origin_w, origin_h = data_info.origin_shape
        lxlyrxry_bboxes = cxcywh_2_lxlyrxry(bboxes)
        decode_bboxes = np.zeros_like(lxlyrxry_bboxes, dtype=np.int32)

        for i, box in enumerate(lxlyrxry_bboxes):
            decode_bboxes[i][0] = int(box[0] / target_w * origin_w)
            decode_bboxes[i][1] = int(box[1] / target_h * origin_h)
            decode_bboxes[i][2] = int(box[2] / target_w * origin_w)
            decode_bboxes[i][3] = int(box[3] / target_h * origin_h)

        # 构建 DetectDataInfo
        detect_info = DetectDataInfo(
            **data_info.__dict__,
            scores=dets[:, 4].cpu().numpy(),
            bboxes=decode_bboxes,
            bbox_format="lxlyrxry",
            normalized=False  # 已反归一化
        )
        return detect_info

    # ---------- 可视化 ----------
    def show(self, data_info: ImgDataInfo, result: DetectDataInfo):
        if self.tracker_server is not None:
            return self.tracker_server.get_server().track_results

        return self.show_predict_result(data_info, result)

    def show_predict_result(self, data_info: ImgDataInfo, result: DetectDataInfo):
        import cv2
        img = data_info.img
        for box in result.bboxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        out_path = self.output_dir / f"result_{data_info.frame_id}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"saved -> {out_path.resolve()}")
        return result.__dict__
