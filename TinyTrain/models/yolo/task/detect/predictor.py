from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from TinyTrain.data import ImgDataInfo, DetectDataInfo
from TinyTrain.engine.predictor import BasePredictor

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
                 backend_map=None,
                 **kwargs):
        super().__init__(config_manager, model, callback, backend_map, **kwargs)
        self.img_shape = kwargs.get("img_shape")

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

        if self.img_shape is None:
            data_info.target_shape = self.config_manager.dataset["img_size"] = check_img_size(data_info.origin_shape, 32)
        else:
            data_info.target_shape = self.img_shape

        transform = T.Compose([
            T.Resize(data_info.target_shape[::-1]),
            T.ToTensor(),
            T.Normalize(mean=0, std=1),
        ])
        img = Image.fromarray(data_info.img)
        tensor = transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    # ---------- 后处理 ----------
    def postprocess(self, data_info: ImgDataInfo, preds: list[torch.Tensor]) -> DetectDataInfo:
        """
        preds: list[Tensor] 来自推理后端
        返回 DetectDataInfo（包含 img + bboxes）
        """
        from TinyTrain.utils.nms import detect_nms

        # detect_nms 输出: List[Tensor] 每张图的 [N,6] (x,y,w,h,conf,cls)
        dets = detect_nms(preds[0], conf_threshold=0.25, nms_threshold=0.5)[0]  # [N,6]

        # 构建 DetectDataInfo
        detect_info = DetectDataInfo(
            img=data_info.img,
            origin_shape=data_info.origin_shape,
            current_shape=data_info.current_shape,
            target_shape=data_info.target_shape,
            img_file=data_info.img_file,
            bboxes=dets[:, :4].cpu().numpy(),  # [N,4]  (x,y,w,h)
            bbox_format="cxcywh",
            normalized=False  # 已反归一化
        )
        return detect_info

    # ---------- 可视化 ----------
    def show(self, data_info: ImgDataInfo, result: DetectDataInfo):
        import cv2

        img = data_info.img.copy()
        target_w, target_h = data_info.target_shape
        origin_w, origin_h = data_info.origin_shape
        for box in result.bboxes:
            cx = box[0]
            cy = box[1]
            w = box[2]
            h = box[3]
            decode_cx = cx / target_w * origin_w
            decode_cy = cy / target_h * origin_h
            decode_w = w / target_w * origin_w
            decode_h = h / target_h * origin_h

            decode_lx = int((decode_cx - decode_w / 2))
            decode_ly = int((decode_cy - decode_h / 2))
            decode_rx = int((decode_cx + decode_w / 2))
            decode_ry = int((decode_cy + decode_h / 2))
            cv2.rectangle(img, (decode_lx, decode_ly), (decode_rx, decode_ry), (0, 255, 0), 2)
        out_path = Path("result.jpg")
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"saved -> {out_path.resolve()}")

        return result.bboxes
