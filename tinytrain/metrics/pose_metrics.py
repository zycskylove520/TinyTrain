import numpy as np
import torch

from pathlib import Path
from PIL import ImageDraw, Image

from .base import BaseImgResult


class PoseImgResult(BaseImgResult):
    """
    目标检测可视化实现：
    在图片上画框、写类别与置信度。
    """

    def __init__(self, keypoint_shape, save_dir: Path, plot_count: int = 4, mode: str = "val", max_sub_len: int = 3, rgb=True, draw_conf_threshold: float = 0.25):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)
        self.keypoint_shape = keypoint_shape
        self.draw_conf_threshold = draw_conf_threshold

    def _prepare_imgs(self, batch_samples):
        """默认：返回 (imgs_np, None)，预测由外部赋值。"""
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        return imgs_np, None

    def _draw_one_img(self, img: np.ndarray, pred: torch.Tensor) -> np.ndarray:
        """
        绘制检测框：框 + 类别 + 置信度。
        """
        from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry

        if pred.numel() == 0:
            return img

        mask = pred[:, 4] > self.draw_conf_threshold
        pred = pred[mask].cpu().numpy()
        bboxes, confs, labels = pred[:, :4], pred[:, 4], pred[:, 5]
        bboxes = cxcywh_2_lxlyrxry(bboxes).astype(int)
        keypoints = pred[:, 6:].reshape(-1, self.keypoint_shape[0], self.keypoint_shape[1])

        # Pillow 用 RGB；如果 img 是 BGR，先转 RGB
        pil_img = Image.fromarray(img[..., ::-1] if not self.rgb else img)
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2), conf, cls in zip(bboxes, confs, labels):
            if conf >= self.draw_conf_threshold:
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                cls_txt = str(int(cls))
                cls_w, cls_h = self._text_size(draw, cls_txt)
                draw.rectangle((x2 - cls_w - 2, y1, x2, y1 + cls_h + 2),
                               fill=(255, 255, 255))
                draw.text((x2 - cls_w, y1), cls_txt, fill=(0, 0, 0), font=self.font)

                conf_txt = f"{conf:.2f}"
                conf_w, conf_h = self._text_size(draw, conf_txt)
                draw.rectangle((x1, y2 - conf_h - 2, x1 + conf_w + 20, y2),
                               fill=(255, 255, 255))
                draw.text((x1, y2 - conf_h), conf_txt, fill=(0, 0, 0), font=self.font)

        # 画关键点（mask>0.5）
        radius = 5
        for kpts in keypoints:  # 遍历每条目标
            for idx, kpt in enumerate(kpts):
                x = kpt[0]
                y = kpt[1]
                m = kpt[2]  # 默认第三列为mask

                if m <= 0.5:  # 不可见跳过
                    continue
                # 简单调色：按关键点索引循环取色
                color = tuple(np.random.randint(0, 255, 3, dtype=np.uint8))
                left_up = (x - radius, y - radius)
                right_down = (x + radius, y + radius)
                draw.ellipse([left_up, right_down], fill=color, outline=color)

        drawn = np.array(pil_img)  # RGB
        return drawn
