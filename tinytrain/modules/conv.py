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
import torch.nn as nn
import torch

from tinytrain.cfg import TTModuleRegistry
from tinytrain.engine import TTBaseModel


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


@TTModuleRegistry.register
class CBA(nn.Module):
    """Standard convolution combined with {Conv2d、BatchNorm2d、Activation}."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act: bool | str | nn.Module = True, bn=True):
        """Initialize CBA layer with given arguments including activation."""
        super().__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        elif act is str:
            self.act = TTBaseModel.get_layer(act)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x


@TTModuleRegistry.register
class DWCBA(CBA):
    """Depth-wise convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, act=True, bn=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(in_channels, out_channels, kernel_size, stride, groups=math.gcd(in_channels, out_channels), dilation=dilation, act=act, bn=bn)


@TTModuleRegistry.register
class GhostCBA(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = out_channels // 2  # hidden channels
        self.cv1 = CBA(in_channels, c_, kernel_size, stride, None, groups, act=act)
        self.cv2 = CBA(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


@TTModuleRegistry.register
class AsymmetricConv2d(nn.Module):
    """
    训练时：3×1 + 1×3 两个卷积
    推理时：可 fuse 成单个 3×3 卷积
    """

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, groups=1, relu=True):
        super().__init__()
        # 只在第一条卷积里降采样；第二条 stride=1
        self.h_conv = nn.Conv2d(in_channels, out_channels,
                                kernel_size=(kernel, 1),
                                stride=stride,
                                padding=(padding, 0),
                                groups=groups, bias=True)
        self.v_conv = nn.Conv2d(out_channels, out_channels,
                                kernel_size=(1, kernel),
                                stride=1,
                                padding=(0, padding),
                                groups=groups, bias=True)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.h_conv(x)
        x = self.v_conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x