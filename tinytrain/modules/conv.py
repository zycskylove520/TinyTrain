import math
import torch.nn as nn
import torch

from tinytrain.cfg import TTModuleRegistry
from tinytrain.engine import BaseModel


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


@TTModuleRegistry.register
class CBA(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, act: bool | str | nn.Module = True):
        """Initialize CBA layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        elif act is str:
            self.act = BaseModel.get_layer(act)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


@TTModuleRegistry.register
class DWConv(CBA):
    """Depth-wise convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(in_channels, out_channels, kernel_size, stride, groups=math.gcd(in_channels, out_channels), dilation=dilation, act=act)


@TTModuleRegistry.register
class GhostConv(nn.Module):
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
