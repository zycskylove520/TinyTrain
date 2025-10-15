import torch

from typing import List, Sequence, Tuple, Union

from tinytrain.global_var.types import BoxFormat

StridesType = Sequence[Union[int, Tuple[int, int]]]
BaseSizesType = Sequence[Union[int, Tuple[int, int]]]


class AnchorGenerator:
    """
    标准 FPN 风格 AnchorGenerator
    支持任意 strides / ratios / scales / base_sizes / center_offset
    """

    def __init__(self,
                 strides: StridesType,
                 ratios: Sequence[float],
                 scales: Sequence[int],
                 base_sizes: BaseSizesType = None,
                 scale_major: bool = True,
                 center_offset: float = 0.5,
                 box_format: BoxFormat = 'lxlyrxry',
                 ):
        """
        Args:
            strides: 各层 stride，可单 int 或 (w_stride, h_stride)
            ratios: 高宽比列表
            scales: 缩放比例列表
            base_sizes: 各层基础边长，默认用 strides 代替
            scale_major: 先生 scale 还是 ratio
            center_offset: 锚点中心相对于网格点的偏移比例，0 表示锚点中心与网格左上角对齐，0.5 表示锚点中心落在网格中心（默认）。
        """
        assert all(r > 0 for r in ratios), 'ratios must be positive'
        assert all(s > 0 for s in scales), 'scales must be positive'

        self.strides = self._parse_strides(strides)
        self.num_levels = len(self.strides)

        self.ratios = torch.tensor(ratios, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.scale_major = scale_major
        self.box_format = box_format
        self.center_offset = center_offset

        if base_sizes is None:
            # 默认：base_size 取 stride 的短边
            self.base_sizes = self._parse_strides(tuple(min(s) for s in self.strides))
        else:
            assert len(base_sizes) == len(strides), f'base_sizes({len(base_sizes)}) must match strides({len(strides)})'
            self.base_sizes = self._parse_strides(base_sizes)

        # 缓存各层 base_anchors，避免重复计算
        self._base_anchors = self.gen_base_anchors()

    # ------------- 对外接口 -------------
    def grid_anchors(self,
                     featmap_sizes: Sequence[Tuple[int, int]],
                     img_shape: Tuple[int, int],
                     device: torch.device = torch.device('cpu'),
                     allowed_border: int = 0,
                     clip: bool = True):
        """
        生成多层级 anchor，并完成边界过滤/裁切
        Returns:
            multi_level_anchors: List[(K, 4)]
        """
        assert len(featmap_sizes) == self.num_levels
        if img_shape[0] == 0 or img_shape[1] == 0:
            return [ba.new_empty((0, 4)).to(device) for ba in self._base_anchors], \
                [torch.zeros(0, dtype=torch.bool, device=device) for _ in self._base_anchors]

        multi_level_anchors, valid_masks = [], []
        for lvl, (wh, base_anchor) in enumerate(zip(featmap_sizes, self._base_anchors)):
            anchors = self.single_level_grid_anchors(
                base_anchor.to(device), wh, self.strides[lvl], device)
            anchors, valid_mask = self.filter_anchors(
                anchors, img_shape, allowed_border, clip)
            multi_level_anchors.append(anchors)
            valid_masks.append(valid_mask)
        return multi_level_anchors, valid_masks

    # ------------- 内部逻辑 -------------
    def gen_base_anchors(self) -> List[torch.Tensor]:
        base_anchors = []
        for base_size in self.base_sizes:
            ba = self._gen_single_level_base_anchors(base_size, self.scales, self.ratios)
            base_anchors.append(ba)
        return base_anchors

    def _gen_single_level_base_anchors(self,
                                       base_size: Tuple[int, int],
                                       scales: torch.Tensor,
                                       ratios: torch.Tensor) -> torch.Tensor:
        w_base, h_base = base_size
        # 1. 得到 ws, hs —— 与之前完全一致
        if self.scale_major:
            ws = w_base * scales[:, None] * torch.sqrt(ratios[None, :])
            hs = h_base * scales[:, None] / torch.sqrt(ratios[None, :])
        else:
            ws = w_base * torch.sqrt(ratios[None, :]) * scales[:, None]
            hs = h_base / torch.sqrt(ratios[None, :]) * scales[:, None]
        ws = ws.reshape(-1)
        hs = hs.reshape(-1)

        # 2. 中心偏移只在这里生效：相对于网格点的偏移
        x_ctr = self.center_offset * w_base
        y_ctr = self.center_offset * h_base

        # 3. 根据格式生成 base_anchor
        if self.box_format == 'lxlyrxry':
            anchors = torch.stack([
                x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                x_ctr + 0.5 * ws, y_ctr + 0.5 * hs
            ], dim=1)
        elif self.box_format == 'lxlywh':
            anchors = torch.stack([
                x_ctr - 0.5 * ws, y_ctr - 0.5 * hs, ws, hs
            ], dim=1)
        else:  # cxcywh
            anchors = torch.stack([x_ctr, y_ctr, ws, hs], dim=1)
        return anchors.to(dtype=torch.float32)

    def single_level_grid_anchors(self,
                                  base_anchors: torch.Tensor,
                                  featmap_size: Tuple[int, int],
                                  stride: Tuple[int, int],
                                  device: torch.device) -> torch.Tensor:
        """内存优化：全程原地/半原地操作"""
        base_anchors = base_anchors.to(dtype=torch.float32)
        feat_w, feat_h = featmap_size
        if feat_w == 0 or feat_h == 0:
            return torch.empty((0, 4), dtype=torch.float32, device=device)

        w_stride, h_stride = stride

        # ---------- 生成偏移 ----------
        shift_x = torch.arange(0, feat_w, dtype=torch.float32, device=device) * w_stride
        shift_y = torch.arange(0, feat_h, dtype=torch.float32, device=device) * h_stride
        if torch.__version__ < '1.10':
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x)  # 老版本默认 ij
        else:
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shift_xx = shift_xx.reshape(-1)  # (K,)
        shift_yy = shift_yy.reshape(-1)  # (K,)

        # ---------- 组装 shifts ----------
        A, K = base_anchors.shape[0], shift_xx.shape[0]
        if self.box_format == 'lxlyrxry':
            shifts = torch.empty((K, 4), dtype=torch.float32, device=device)
            shifts[:, 0] = shift_xx
            shifts[:, 1] = shift_yy
            shifts[:, 2] = shift_xx
            shifts[:, 3] = shift_yy
        else:  # lxlywh / cxcywh
            shifts = torch.zeros((K, 4), dtype=torch.float32, device=device)
            shifts[:, 0] = shift_xx
            shifts[:, 1] = shift_yy

        # ---------- 原地广播加法 ----------
        # (A,4) -> (1,A,4)   (K,1,4)
        anchors = base_anchors.view(1, A, 4) + shifts.view(K, 1, 4)  # 返回 (K,A,4)
        return anchors.reshape(K * A, 4)

    # ------------- 过滤函数 -------------
    def filter_anchors(self,
                       anchors: torch.Tensor,
                       img_shape: Tuple[int, int],
                       allowed_border: int = 0,
                       clip: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        img_w, img_h = img_shape
        fmt = self.box_format

        # ---------- 1. 拿到 x1,y1,x2,y2 ----------
        if fmt == 'lxlyrxry':
            x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        elif fmt == 'lxlywh':
            x1, y1, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
            x2, y2 = x1 + w, y1 + h
        else:  # cxcywh
            cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
            x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h

        # ---------- 2. clip ----------
        if clip:
            x1.clamp_(0, img_w)
            y1.clamp_(0, img_h)
            x2.clamp_(0, img_w)
            y2.clamp_(0, img_h)

        # ---------- 3. 过滤 mask ----------
        if allowed_border >= 0:
            invalid = (x1 < -allowed_border) | (y1 < -allowed_border) | \
                      (x2 > img_w + allowed_border) | (y2 > img_h + allowed_border)
            valid_mask = ~invalid
        else:  # allowed_border < 0：必须全部在图内 + 面积>0
            inside = (x1 >= 0) & (y1 >= 0) & (x2 <= img_w) & (y2 <= img_h)
            positive_area = ((x2 - x1) > 0) & ((y2 - y1) > 0)
            valid_mask = inside & positive_area

        return anchors, valid_mask

    # ------------- 工具 -------------
    @staticmethod
    def _parse_strides(strides: StridesType) -> Tuple[Tuple[int, int], ...]:
        parsed = []
        for s in strides:
            if isinstance(s, int):
                parsed.append((s, s))
            else:
                parsed.append(tuple(s))
        return tuple(parsed)

    @property
    def num_base_anchors(self) -> Tuple[int, ...]:
        return tuple(ba.shape[0] for ba in self._base_anchors)


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    import torch
    import cv2
    import numpy as np

    # 参数：1 层 FPN，stride=32，3 组 scale，3 组 ratio
    strides = [32]
    scales = [1, 2, 4]
    ratios = [0.5, 1.0, 2.0]

    gen = AnchorGenerator(strides=strides, scales=scales, ratios=ratios)

    # 1×1 特征图 → 共 9 个框
    featmap_sizes = [(1, 1)]
    img_shape = (32, 32)  # 高，宽
    anchors, valid = gen.grid_anchors(featmap_sizes, img_shape, device=torch.device('cpu'))
    anchors = anchors[0].numpy()  # (9,4)  lxlyrxry 格式

    # 画图
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in anchors:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('anchors.jpg', canvas)
    print('saved -> anchors.jpg')
