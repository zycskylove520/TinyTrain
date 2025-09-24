import math
import numpy as np
import torch
import cv2

from pathlib import Path
from matplotlib import pyplot as plt

from tinytrain.utils.segment_utils import decode_pred_masks
from tinytrain.utils import LOGGER

from .base import BaseImgResult


class SegmentImgResult(BaseImgResult):
    """
    实例分割结果可视化类（OpenCV 实现）。

    职责：
    1. 收集一批图像及其对应的分割预测（检测框、置信度、类别、原型 mask、mask 系数）。
    2. 按置信度阈值过滤低分目标，调用 decode_pred_masks 生成二值 mask。
    3. 使用 OpenCV 在图像上绘制：
       - 半透明实例 mask（随机色，50 % 透明度）
       - 检测框（同色边框）
       - 类别标签与置信度（白底黑字，位于框角）
    4. 返回与输入通道顺序一致的 numpy 图像数组，供父类拼板保存。

    父类 BaseImgResult 负责：
    - 子图数量控制
    - 拼板布局
    - 文件写入
    本类仅关注单张图像的绘制逻辑。
    """

    def __init__(self, target_shape, save_dir: Path,
                 plot_count: int = 4, mode: str = "val",
                 max_sub_len: int = 3, rgb=True,
                 draw_conf_threshold: float = 0.25):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)
        self.draw_conf_threshold = draw_conf_threshold
        self.target_shape = target_shape

        self.imgs = []
        self.preds = []
        self.protos = []

    # ------------------------------------------------------------------
    # 新增两个小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _get_text_size(text, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=0.4, thickness=1):
        """计算文字宽高（像素）"""
        (tw, th), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        return tw, th + 4  # 上下各留 2 像素

    # ------------------------------------------------------------------
    # 核心：单张图绘制
    # ------------------------------------------------------------------
    def _draw_one_img(self, img: np.ndarray,
                      pred: torch.Tensor,
                      **kwargs) -> np.ndarray:
        """
        用 OpenCV 绘制检测框 + 类别 + 置信度 + 半透明 mask
        """
        from tinytrain.utils.box_utils import cxcywh_2_lxlyrxry

        if pred.numel() == 0:
            return img

        # 过滤低置信度
        mask = pred[:, 4] > self.draw_conf_threshold
        pred = pred[mask]
        bboxes = cxcywh_2_lxlyrxry(pred[:, :4])
        confs = pred[:, 4].cpu().numpy()
        labels = pred[:, 5].cpu().numpy()
        proto = kwargs['proto']
        pred_mask_vec = pred[:, 6:]
        pred_masks = decode_pred_masks(proto, bboxes, pred_mask_vec,
                                       self.target_shape,
                                       bin=True, retina_masks=True).cpu().numpy().astype(bool)

        # 准备彩色图（OpenCV 默认 BGR）
        drawn = img.copy()
        if self.rgb:
            drawn = cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)

        # 画框 + 文字
        bboxes = bboxes.cpu().numpy().astype(np.int32)
        for m, (x1, y1, x2, y2), conf, cls in zip(pred_masks, bboxes, confs, labels):
            # 随机颜色
            color = np.random.randint(0, 256, 3, dtype=np.uint8).tolist()

            # 半透明 mask: 彩色 overlay
            overlay = drawn.copy()
            overlay[m] = color  # mask=True 的位置赋色
            drawn = cv2.addWeighted(overlay, 0.5, drawn, 0.5, 0)

            # 画框
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)

            # 类别文字
            cls_txt = str(int(cls))
            tw, th = self._get_text_size(cls_txt)
            cv2.rectangle(drawn,
                          (x2 - tw - 2, y1),
                          (x2, y1 + th),
                          (255, 255, 255), -1)
            cv2.putText(drawn, cls_txt,
                        (x2 - tw, y1 + th - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            # 置信度文字
            conf_txt = f"{conf:.2f}"
            tw, th = self._get_text_size(conf_txt)
            cv2.rectangle(drawn,
                          (x1, y2 - th),
                          (x1 + tw + 4, y2),
                          (255, 255, 255), -1)
            cv2.putText(drawn, conf_txt,
                        (x1 + 2, y2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # 返回与输入一致的通道顺序
        if self.rgb:
            drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        return drawn

    # ------------------------------------------------------------------
    # 以下方法与原文件完全一致，无需改动
    # ------------------------------------------------------------------
    def add_sample(self, img, pred, proto):
        img = img.permute(1, 2, 0).contiguous()
        if img.max() <= 1.0:
            img = img * 255.
        img = img.byte().cpu().numpy()
        self.imgs.append(img)
        self.preds.append(pred)
        self.protos.append(proto)

    def do_plot(self):
        if not len(self.imgs):
            LOGGER.warning(f"No predict mask available, skipping segment image result plots.")
            return

        if self.is_plot:
            return
        if self.plot_tick >= self.plot_count:
            self.is_plot = True
            return
        self.plot_tick += 1

        imgs_np = np.stack(self.imgs, axis=0)
        B, H, W, _ = imgs_np.shape
        fill_len = min(self.max_sub_len, math.ceil(B ** 0.5))
        n_show = min(B, fill_len * fill_len)

        if self._fig is None:
            dpi = 100
            figsize = (fill_len * W / dpi, fill_len * H / dpi)
            self._fig, self._axs = plt.subplots(
                fill_len, fill_len,
                figsize=figsize, dpi=dpi,
                constrained_layout=True,
                gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            for ax in self._axs.flat:
                ax.axis('off')

        for idx in range(n_show):
            row, col = divmod(idx, fill_len)
            ax = self._axs[row, col]
            drawn = self._draw_one_img(imgs_np[idx],
                                       self.preds[idx],
                                       proto=self.protos[idx])
            ax.imshow(drawn if drawn.shape[-1] == 3 else drawn, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n_show, fill_len * fill_len):
            self._axs.flat[idx].clear()
            self._axs.flat[idx].axis('off')

        self._fig.savefig(
            self.save_dir / f'{self.mode}_img_result_{self.plot_tick}.png',
            dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(self._fig)
