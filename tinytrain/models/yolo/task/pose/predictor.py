from __future__ import annotations

from typing import TYPE_CHECKING

from tinytrain.data.data_format import ImgDataInfo, PoseDataInfo
from tinytrain.engine.predictor import BasePredictor
from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry
from tinytrain.cfg.TT_register import TTEngineRegistry
from tinytrain.utils.nms import detect_nms_with_keypoint

if TYPE_CHECKING:
    import torch


class YOLOPosePredictor(BasePredictor):
    """
    YOLO姿态估计预测器
    输入：图片/视频/目录等
    输出：PoseDataInfo（含 img + bboxes + keypoints）
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
        self.keypoint_shape = self.config_manager.dataset["keypoint_shape"]
        self.mask_threshold = self.config_manager.inference["predict_mask_threshold"]

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
    def postprocess(self, data_info: ImgDataInfo, inference_result: list[torch.Tensor]) -> PoseDataInfo:
        """
        inference_result: list[Tensor] 来自推理后端
        返回 PoseDataInfo
        """
        import numpy as np

        # detect_nms_with_keypoint 输出: List[Tensor] 每张图的 [N,6+keypints_num] (x,y,w,h,conf,cls)
        outputs = detect_nms_with_keypoint(inference_result[0],
                                           conf_threshold=self.config_manager.inference["predict_conf_threshold"],
                                           nms_threshold=self.config_manager.inference["predict_nms_threshold"],
                                           keypoint_shape=self.keypoint_shape)[0]

        # bboxes恢复原图尺寸，并转为lxlyrxry
        bboxes = outputs[:, :4].cpu().numpy()
        scores = outputs[:, 4].cpu().numpy()
        cls = outputs[:, 5].cpu().numpy()
        keypoints = outputs[:, 6:].reshape(-1, self.keypoint_shape[0], self.keypoint_shape[1]).cpu().numpy()

        target_w, target_h = data_info.target_shape
        origin_w, origin_h = data_info.origin_shape
        lxlyrxry_bboxes = cxcywh_2_lxlyrxry(bboxes)
        decode_bboxes = np.zeros_like(lxlyrxry_bboxes, dtype=np.int32)
        decode_keypoints = np.zeros_like(keypoints, dtype=np.float32)

        for i, box in enumerate(lxlyrxry_bboxes):
            decode_bboxes[i][0] = int(box[0] / target_w * origin_w)
            decode_bboxes[i][1] = int(box[1] / target_h * origin_h)
            decode_bboxes[i][2] = int(box[2] / target_w * origin_w)
            decode_bboxes[i][3] = int(box[3] / target_h * origin_h)
            decode_keypoints[i, :, 0] = keypoints[i, :, 0] / target_w * origin_w
            decode_keypoints[i, :, 1] = keypoints[i, :, 1] / target_h * origin_h
            decode_keypoints[i, :, 2] = keypoints[i, :, 2]

        # 构建 PoseDataInfo
        pose_info = PoseDataInfo(
            **data_info.__dict__,
            scores=scores,
            label=cls,
            bboxes=decode_bboxes,
            bbox_format="lxlyrxry",  # 用于跟踪
            normalized=False,  # 已反归一化
            keypoints=decode_keypoints,
            kpt_shape=self.keypoint_shape
        )
        return pose_info

    # ---------- 可视化 ----------
    def show(self, data_info: ImgDataInfo, result: PoseDataInfo):
        if self.tracker_server is not None:
            return self.tracker_server.track_results

        return self.show_predict_result(data_info, result)

    def show_predict_result(self, data_info: ImgDataInfo, result: PoseDataInfo):
        import cv2
        import random
        img = data_info.img
        for i, box in enumerate(result.bboxes):
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            object_kpt = result.keypoints[i]
            for kpt in object_kpt:
                if kpt[2] > self.mask_threshold:  # 可见
                    # 随机彩色
                    color = (random.randint(0, 255),
                             random.randint(0, 255),
                             random.randint(0, 255))
                    cv2.circle(img, (int(kpt[0]), int(kpt[1])), 5, color, -1)

        out_path = self.output_dir / f"result_{data_info.frame_id}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"saved -> {out_path.resolve()}")
        return result.__dict__
