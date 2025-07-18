import math
import numpy as np
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt


# -------------------- 基类 --------------------
class BaseImgResult(ABC):
    """
    只负责：
    1. 计数、决定是否绘制
    2. 创建/复用 Figure
    3. 循环调用子类实现的 _draw_one_img 并保存
    子类只需实现 _draw_one_img 即可。
    """

    def __init__(self,
                 save_dir: Path,
                 plot_count: int = 4,
                 mode: str = "val",
                 max_sub_len: int = 3,
                 rgb=True):
        self.save_dir = save_dir
        self.mode = mode
        self.max_sub_len = max_sub_len
        self.plot_count = plot_count
        self.plot_tick = 0
        self.is_plot = False
        self._fig = None
        self._axs = None
        self.font = ImageFont.load_default()
        self.rgb = rgb

    # ---------- 子类必须实现 ----------
    @abstractmethod
    def _draw_one_img(self, img: np.ndarray, pred: torch.Tensor) -> np.ndarray:
        """
        把单张图片画好并返回 uint8 RGB（或 BGR，后面统一转 RGB）。
        """
        pass

    # ---------- 子类可重写 ----------
    def _prepare_imgs(self, batch_samples) -> Tuple[np.ndarray, torch.Tensor]:
        """
        默认实现：把 Tensor 转成 uint8 np.ndarray，并把预测 Tensor 准备好。
        子类可以按任务重写（例如分类不需要 NMS）。
        """
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        return imgs_np, batch_samples.pred if hasattr(batch_samples, 'pred') else None

    # ---------- 公共绘制入口 ----------
    def plot(self, batch_samples, preds: list[torch.Tensor]):
        if self.is_plot:
            return
        if self.plot_tick >= self.plot_count:
            self.is_plot = True
            return
        self.plot_tick += 1

        imgs_np, _ = self._prepare_imgs(batch_samples)
        B, H, W, _ = imgs_np.shape
        fill_len = min(self.max_sub_len, math.ceil(B ** 0.5))
        n_show = min(B, fill_len * fill_len)

        # 复用 Figure
        if self._fig is None:
            dpi = 100
            figsize = (fill_len * W / dpi, fill_len * H / dpi)
            self._fig, self._axs = plt.subplots(
                fill_len, fill_len,
                figsize=figsize, dpi=dpi,
                constrained_layout=True,  # 取代 tight_layout
                gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            for ax in self._axs.flat:
                ax.axis('off')

        # 画子图
        for idx in range(n_show):
            row, col = divmod(idx, fill_len)
            ax = self._axs[row, col]
            drawn = self._draw_one_img(imgs_np[idx], preds[idx])  # RGB
            ax.imshow(drawn if drawn.shape[-1] == 3 else drawn, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

        # 清空多余子图
        for idx in range(n_show, fill_len * fill_len):
            self._axs.flat[idx].clear()
            self._axs.flat[idx].axis('off')

        self._fig.savefig(
            self.save_dir / f'{self.mode}_img_result_{self.plot_tick}.png',
            dpi=100, bbox_inches='tight', pad_inches=0)

    # ---------- Pillow 兼容 ----------
    def _text_size(self, draw: ImageDraw.Draw, text: str):
        try:
            return draw.textsize(text, font=self.font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]


class ClassifyImgResult(BaseImgResult):
    def __init__(self, save_dir: Path, plot_count: int = 4, mode: str = "val", max_sub_len: int = 3, rgb=True, **kwargs):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)
        # 额外存一份 class_names
        self.class_names = kwargs.get('class_names_dict', {})
        self.title = None

    def _prepare_imgs(self, batch_samples):
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()  # bchw->bhwc
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        true_labels = batch_samples.target.cpu().numpy()
        return imgs_np, true_labels  # 返回 (imgs, true_labels)

    def _draw_one_img(self, img: np.ndarray, info) -> np.ndarray:
        pred_tensor, true_label = info
        pred_label = int(torch.argmax(pred_tensor).item())

        pil_img = Image.fromarray(img.astype(np.uint8))
        # 若原通道顺序与需求相反，则翻转
        drawn = np.array(pil_img)
        if not self.rgb and drawn.shape[-1] == 3:  # 需要 BGR
            drawn = drawn[..., ::-1]
        elif self.rgb and drawn.shape[-1] == 3:  # 需要 RGB
            drawn = drawn[..., ::-1] if img.shape[-1] == 3 and img.dtype == np.uint8 else drawn
        # 这里 img 是 uint8 RGB，如果网络给的是 BGR 会经过 _prepare_imgs 的 ::-1 处理
        # 上面判断条件可再按实际 pipeline 微调
        self.title = (f"true:{self.class_names.get(true_label, true_label)}\n"
                      f"pred:{self.class_names.get(pred_label, pred_label)}")
        return drawn

    def plot(self, batch_samples, pred: torch.Tensor):
        if self.is_plot:
            return

        if self.plot_tick >= self.plot_count:
            self.is_plot = True
            return

        self.plot_tick += 1

        imgs_np, true_labels = self._prepare_imgs(batch_samples)
        pred_np = pred.cpu()
        B, *_ = imgs_np.shape
        fill_len = min(self.max_sub_len, math.ceil(B ** 0.5))
        n_show = min(B, fill_len * fill_len)

        if self._fig is None:
            dpi = 100
            self._fig, self._axs = plt.subplots(
                fill_len,
                fill_len,
                dpi=dpi,
                constrained_layout=True,  # 取代 tight_layout
                gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
            for ax in self._axs.flat:
                ax.axis('off')

        for idx in range(n_show):
            row, col = divmod(idx, fill_len)
            ax = self._axs[row, col]
            drawn = self._draw_one_img(imgs_np[idx], (pred_np[idx], true_labels[idx]))
            ax.imshow(drawn)
            ax.set_title(self.title)  # 设置标题
            ax.set_xticks([])
            ax.set_yticks([])

        self._fig.savefig(
            self.save_dir / f'{self.mode}_img_result_{self.plot_tick}.png',
            dpi=100, bbox_inches='tight', pad_inches=0)


class YOLODetectImgResult(BaseImgResult):
    def __init__(self, save_dir: Path, plot_count: int = 4, mode: str = "val", max_sub_len: int = 3, rgb=True):
        super().__init__(save_dir, plot_count, mode, max_sub_len, rgb)

    def _prepare_imgs(self, batch_samples):
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        return imgs_np, None

    def _draw_one_img(self, img: np.ndarray, pred: torch.Tensor) -> np.ndarray:
        from TinyTrain.utils.box_utils import cxcywh_2_lxlyrxry

        if pred.numel() == 0:
            return img

        pred = pred.cpu().numpy()
        bboxes, confs, labels = pred[:, :4], pred[:, 4], pred[:, 5]
        bboxes = cxcywh_2_lxlyrxry(bboxes).astype(int)

        # Pillow 用 RGB；如果 img 是 BGR，先转 RGB
        pil_img = Image.fromarray(img[..., ::-1] if not self.rgb else img)
        draw = ImageDraw.Draw(pil_img)

        for (x1, y1, x2, y2), conf, cls in zip(bboxes, confs, labels):
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

        drawn = np.array(pil_img)  # RGB
        return drawn
