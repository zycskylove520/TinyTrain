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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinytrain.modules.conv import CBA
from tinytrain.cfg import TTModuleRegistry


@TTModuleRegistry.register
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, kernel_size=(3, 3), hidden_channels=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(out_channels * hidden_channels)  # hidden channels
        self.cv1 = CBA(in_channels, c_, kernel_size[0], 1)
        self.cv2 = CBA(c_, out_channels, kernel_size[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@TTModuleRegistry.register
class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, in_channels, out_channels, stride=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * out_channels
        self.cv1 = CBA(in_channels, out_channels, kernel_size=1, stride=1, act=True)
        self.cv2 = CBA(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, act=True)
        self.cv3 = CBA(out_channels, c3, kernel_size=1, act=False)
        self.shortcut = nn.Sequential(CBA(in_channels, c3, kernel_size=1, stride=stride, act=False)) if stride != 1 or in_channels != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


@TTModuleRegistry.register
class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, in_channels, out_channels, stride=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                CBA(in_channels, out_channels, kernel_size=7, stride=2, padding=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(in_channels, out_channels, stride, e=e)]
            blocks.extend([ResNetBlock(e * out_channels, out_channels, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


@TTModuleRegistry.register
class FPN(nn.Module):
    """
    Standard FPN (Lin et al. CVPR 2017)
    """

    def __init__(self,
                 in_channels_list: list[int],
                 out_channels: int = 256,
                 use_maxpool: bool = False):  # 是否额外加（stride=2 的 maxpool）
        super().__init__()
        self.out_channels = out_channels
        self.use_maxpool = use_maxpool
        self.num_ins = len(in_channels_list)

        # 1×1 lateral conv
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)
            )

        # 融合后 3×3 conv
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )

        # 可选 maxpool：对 最后一层 做 stride-2 maxpool
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(inputs) == self.num_ins, f"FPN expects {self.num_ins} feature maps, got {len(inputs)}"

        # 1. lateral 降维
        laterals = [lat(x) for lat, x in zip(self.lateral_convs, inputs)]

        # 2. 自顶向下融合
        for i in range(self.num_ins - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:],
                                      mode='nearest')
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3. 3×3 卷积输出
        outs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]

        # 4. 可选 maxpool
        if self.use_maxpool:
            p6 = self.maxpool(outs[-1])
            outs.append(p6)

        return outs


@TTModuleRegistry.register
class PAN(nn.Module):
    """
    标准 Path Aggregation Network
    输入:  backbone 给出的 C2, C3, C4, C5（可更多/更少）
    输出:  N2, N3, N4, N5（与原论文同名，通道数 = out_channels）
    """

    def __init__(self,
                 in_channels_list: list[int],
                 out_channels: int = 256):
        super().__init__()
        self.num_ins = len(in_channels_list)

        # 1. lateral 1×1 降维
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels_list:
            self.lateral_convs.append(CBA(ch, out_channels, 1))

        # 2. 自顶向下平滑 3×3
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            self.fpn_convs.append(CBA(out_channels, out_channels, 3))

        # 3. 自底向上下采样 3×3（stride=2）
        self.downsample_convs = nn.ModuleList()
        # 4. 自底向上平滑 3×3
        self.pafpn_convs = nn.ModuleList()
        for _ in range(self.num_ins - 1):  # 只需要 (n-1) 个
            self.downsample_convs.append(CBA(out_channels, out_channels, 3, s=2))
            self.pafpn_convs.append(CBA(out_channels, out_channels, 3))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(xs) == self.num_ins, f"PAN expects {self.num_ins} feats, got {len(xs)}"

        # ---- 1. lateral ----
        laterals = [lat(x) for lat, x in zip(self.lateral_convs, xs)]

        # ---- 2. top-down ----
        for i in range(self.num_ins - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
            laterals[i - 1] = laterals[i - 1] + up
        # 平滑
        inter_outs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]

        # ---- 3. bottom-up ----
        outs = [inter_outs[0]]  # N2
        for i in range(self.num_ins - 1):
            down = self.downsample_convs[i](inter_outs[i])  # 下采样
            inter_outs[i + 1] = inter_outs[i + 1] + down  # 融合
            outs.append(self.pafpn_convs[i](inter_outs[i + 1]))  # 平滑后得到 N3/N4/N5

        return outs


@TTModuleRegistry.register
class PAFPN(FPN):
    """Path-Aggregation FPN"""

    def __init__(self,
                 in_channels_list: list[int],
                 out_channels: int = 256,
                 use_maxpool: bool = False):
        super().__init__(in_channels_list, out_channels, use_maxpool)

        # ====== 下面是 PAFPN 额外增加的 bottom-up 支路 ======
        # 需要 (num_ins-1) 个 stride=2 的 3×3 做下采样
        self.downsample_convs = nn.ModuleList()
        # 对应融合后再做一次 3×3 平滑
        self.pafpn_convs = nn.ModuleList()
        for _ in range(self.num_ins - 1):
            self.downsample_convs.append(
                CBA(out_channels, out_channels, 3, stride=2, act=False)
            )
            self.pafpn_convs.append(
                CBA(out_channels, out_channels, 3, stride=1, act=False)
            )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        # 1. 先拿到 FPN 的 top-down 结果
        inter_outs = super().forward(xs)  # List[P2, P3, P4, P5] (+P6)
        # 如果之前开了 maxpool，先把 P6 拿出来，最后再拼回去
        p6 = None
        if self.use_maxpool:
            p6 = inter_outs.pop()  # 取出 P6，此时 inter_outs 长度 = num_ins

        # 2. bottom-up 路径
        for i in range(len(inter_outs) - 1):
            # 用 stride=2 卷积把低层特征下采样，加到下一层
            inter_outs[i + 1] = inter_outs[i + 1] + self.downsample_convs[i](inter_outs[i])

        # 3. 再做一次 3×3 平滑（除了最底层 P2）
        outs = [inter_outs[0]]  # P2 保持不动
        for i in range(1, len(inter_outs)):
            outs.append(self.pafpn_convs[i - 1](inter_outs[i]))

        # 4. 把 P6 放回去
        if p6 is not None:
            outs.append(p6)
        return outs


@TTModuleRegistry.register
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        self.conv3X3 = CBA(in_channels=in_channel, out_channels=out_channel // 2, kernel_size=3, act=False)

        self.conv5X5_1 = CBA(in_channels=in_channel, out_channels=out_channel // 4, kernel_size=3)
        self.conv5X5_2 = CBA(in_channels=out_channel // 4, out_channels=out_channel // 4, kernel_size=3, act=False)

        self.conv7X7_2 = CBA(in_channels=out_channel // 4, out_channels=out_channel // 4, kernel_size=3)
        self.conv7x7_3 = CBA(in_channels=out_channel // 4, out_channels=out_channel // 4, kernel_size=3, act=False)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


@TTModuleRegistry.register
class SSHPAN(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super(SSHPAN, self).__init__()
        self.ssh1 = SSH(in_channels, out_channels_list[0])
        self.ssh2 = SSH(in_channels, out_channels_list[1])
        self.ssh3 = SSH(in_channels, out_channels_list[2])

        # self.pan = PAN(in_channels_list=out_channels_list, out_channels=)

    def forward(self, x: list[torch.Tensor]):
        out1 = self.ssh1(x[0])
        out2 = self.ssh2(x[1])
        out3 = self.ssh3(x[2])

        return [out1, out2, out3]


@TTModuleRegistry.register
class BottleneckDW(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            CBA(in_channels=in_channels, out_channels=groups, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            CBA(in_channels=groups, out_channels=groups, groups=groups, kernel_size=kernel_size, stride=stride, padding=padding),
            CBA(in_channels=groups, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), act=False)
        )

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


@TTModuleRegistry.register
class Conv2Linear(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512, kernel_size=1, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.conv = CBA(in_channels, hidden_channels, kernel_size, stride, padding, groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        x = self.linear(x)
        x = self.bn(x)

        return x


@TTModuleRegistry.register
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, in_channels=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
