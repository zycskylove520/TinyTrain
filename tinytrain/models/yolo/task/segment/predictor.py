from __future__ import annotations

import numpy as np
import torch.nn.functional as F

from typing import TYPE_CHECKING

from tinytrain.data.data_format import ImgDataInfo, SegmentDataInfo
from tinytrain.engine.predictor import BasePredictor
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry
from tinytrain.cfg.TT_register import TTEngineRegistry
from tinytrain.utils.nms import detect_nms_with_mask
from tinytrain.utils.segment_utils import decode_pred_masks

if TYPE_CHECKING:
    import torch


class YOLOSegmentPredictor(BasePredictor):
    """
    YOLO实例分割预测器
    输入：图片/视频/目录等
    输出：SegmentDataInfo（含 img + bboxes + mask）
    """

    def __init__(self,
                 config_manager,
                 device: torch.device,
                 model,
                 callback,
                 backend=None,
                 **kwargs):
        super().__init__(config_manager=config_manager, device=device, model=model, callback=callback, backend=backend, **kwargs)

        self.img_shape = kwargs.get("img_shape")

        # 绑定跟踪服务
        self.tracker_server = None
        if kwargs.get("track", False):
            assert isinstance(kwargs["track"], bool)
            track_backend = kwargs.get("track_backend", "bytetrack")
            self.tracker_server = TTEngineRegistry.get(self.config_manager, "track_server", track_backend)(config_manager=self.config_manager, callback=self.callback)

    # ---------- 前处理 ----------
    def preprocess(self, data_info: ImgDataInfo) -> torch.Tensor:
        import torchvision.transforms as T
        from PIL import Image
        from tinytrain.utils.checks import check_img_size
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
    def postprocess(self, data_info: ImgDataInfo, inference_result: list[torch.Tensor]) -> SegmentDataInfo:
        """
        inference_result: list[Tensor] 来自推理后端
        返回 SegmentDataInfo
        """
        import numpy as np

        pred, proto = inference_result[0]
        # detect_nms_with_mask 输出: List[Tensor] 每张图的 [N,6+keypints_num] (x,y,w,h,conf,cls)
        outputs = detect_nms_with_mask(pred=pred,
                                       nc=self.config_manager.dataset["nc"],
                                       conf_threshold=self.config_manager.inference["predict_conf_threshold"],
                                       nms_threshold=self.config_manager.inference["predict_nms_threshold"])[0]

        target_w, target_h = data_info.target_shape
        origin_w, origin_h = data_info.origin_shape

        # bboxes转为lxlyrxry
        bboxes = cxcywh_2_lxlyrxry(outputs[:, :4])
        scores = outputs[:, 4].cpu().numpy()
        cls = outputs[:, 5].cpu().numpy()

        # 解码获得目标图像大小的mask图像
        pred_mask_vec = outputs[:, 6:]
        pred_masks = decode_pred_masks(proto[0], bboxes, pred_mask_vec, data_info.target_shape, bin=False, retina_masks=True)
        # mask图像插值回原图大小
        pred_masks = pred_masks.unsqueeze(1).float()  # [N,1,target_h,target_w]
        pred_masks = F.interpolate(pred_masks,
                                   size=(origin_h, origin_w),
                                   mode='bilinear',
                                   align_corners=False)
        # 二值化
        pred_masks = pred_masks.squeeze(1).gt_(0)  # [N,origin_h,origin_w]

        # 解码获得原图bboxes
        decode_bboxes = np.zeros_like(bboxes.cpu().numpy(), dtype=np.int32)
        for i, box in enumerate(bboxes):
            decode_bboxes[i][0] = int(box[0] / target_w * origin_w)
            decode_bboxes[i][1] = int(box[1] / target_h * origin_h)
            decode_bboxes[i][2] = int(box[2] / target_w * origin_w)
            decode_bboxes[i][3] = int(box[3] / target_h * origin_h)

        # 构建 SegmentDataInfo
        seg_info = SegmentDataInfo(
            **data_info.__dict__,
            scores=scores,
            label=cls,
            bboxes=decode_bboxes,
            bbox_format="lxlyrxry",  # 用于跟踪
            normalized=False,  # 已反归一化
            masks=pred_masks
        )
        return seg_info

    # ---------- 可视化 ----------
    def show(self, data_info: ImgDataInfo, result: SegmentDataInfo):
        if self.tracker_server is not None:
            return self.tracker_server.track_results

        return self.show_predict_result(data_info, result)

    def show_predict_result(self, data_info: ImgDataInfo, result: SegmentDataInfo):
        import cv2
        mask_alpha = 0.5

        # 1. 准备画布
        vis = data_info.img.copy()  # H,W,3  uint8

        # 2. 把 tensor 全部转成 numpy
        masks_np = result.masks.cpu().numpy().astype(bool)  # bool[N,H,W]
        bboxes = result.bboxes  # 已经是 numpy
        labels = result.label.astype(np.int32)  # 类别索引
        scores = result.scores  # 置信度

        # 3. 逐实例绘制
        for idx, mask in enumerate(masks_np):  # mask: bool[H,W]
            # 3.1 随机颜色
            color = np.random.randint(0, 256, 3, dtype=np.uint8).tolist()

            # 3.2 半透明 mask: 彩色 overlay
            overlay = vis.copy()
            overlay[mask] = color  # mask=True 的位置赋色
            vis = cv2.addWeighted(overlay, mask_alpha, vis, 1 - mask_alpha, 0)

            # 3.3 画框
            x1, y1, x2, y2 = map(int, bboxes[idx])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 3.4 写文字：类别索引 + 置信度
            txt = f"{labels[idx]}:{scores[idx]:.2f}"
            cv2.putText(vis, txt, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. 保存
        out_path = self.output_dir / f"result_{data_info.frame_id}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"saved -> {out_path.resolve()}")

        return result.__dict__
