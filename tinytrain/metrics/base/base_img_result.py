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

import math
import numpy as np
import torch

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from PIL import ImageFont, ImageDraw
from matplotlib import pyplot as plt


class BaseImgResult(ABC):
    """
    通用可视化基类：负责
    1) 计数 & 决定是否绘制；
    2) 复用 matplotlib Figure，避免重复创建；
    3) 循环调用子类实现的 `_draw_one_img` 并保存 PNG。

    子类只需实现 `_draw_one_img`，即可完成单张图的绘制逻辑。
    """

    def __init__(self,
                 save_dir: Path,
                 plot_count: int = 4,
                 mode: str = "val",
                 max_sub_len: int = 3,
                 rgb=True):
        """
        Args:
            save_dir (Path): 保存图片的目录。
            plot_count (int): 最多绘制多少张 batch。
            mode (str): 训练/验证/测试模式，用于文件名。
            max_sub_len (int): 子图网格行列上限（ceil(sqrt(B)) 的截断）。
            rgb (bool): 输入通道是否 RGB，若为 False 则会在 `_draw_one_img` 中翻转 BGR→RGB。
        """
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
    def _draw_one_img(self, img: np.ndarray, pred: torch.Tensor, **kwargs) -> np.ndarray:
        """
        把单张图片画好并返回 uint8 RGB（或 BGR，后面统一转 RGB）。

        Args:
            img (np.ndarray): HWC uint8 图像。
            pred (torch.Tensor): 任务相关预测张量，子类自行解释。

        Returns:
            np.ndarray: 绘制后的图像，HWC。
        """
        pass

    # ---------- 子类可重写 ----------
    def _prepare_imgs(self, batch_samples) -> Tuple[np.ndarray, torch.Tensor]:
        """
        默认实现：把 Tensor → uint8 np.ndarray，并把预测张量准备好。
        子类可按任务重写（如分类不需要 NMS，检测需要解码）。
        """
        imgs = batch_samples.data.permute(0, 2, 3, 1).contiguous()
        if imgs.max() <= 1.0:
            imgs = imgs * 255.
        imgs_np = imgs.byte().cpu().numpy()
        return imgs_np, batch_samples.pred if hasattr(batch_samples, 'pred') else None

    # ---------- 公共绘制入口 ----------
    def plot(self, batch_samples, preds: list[torch.Tensor], **kwargs):
        """
        主入口：根据批次绘制子图并保存。
        - 自动跳过超过 plot_count 的请求。
        - 自动复用 Figure。
        """
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

        plt.close(self._fig)

    # ---------- Pillow 兼容 ----------
    def _text_size(self, draw: ImageDraw.Draw, text: str):
        """
        Pillow 兼容接口：返回文字 (width, height)。
        """
        try:
            return draw.textsize(text, font=self.font)
        except AttributeError:
            bbox = draw.textbbox((0, 0), text, font=self.font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
