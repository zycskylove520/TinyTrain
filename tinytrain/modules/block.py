# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn

from .conv import Conv

from tinytrain.cfg.TT_register import TTModuleRegistry


@TTModuleRegistry.register
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, hidden_channels=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(out_channels * hidden_channels)  # hidden channels
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, groups, kernel_size=((3, 3), (3, 3)), hidden_channels=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


@TTModuleRegistry.register
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, hidden_channels=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(out_channels * hidden_channels)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1)  # optional act=FReLU(out_channels)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, groups, kernel_size=((1, 1), (3, 3)), hidden_channels=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


@TTModuleRegistry.register
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, kernel_size=(3, 3), hidden_channels=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(out_channels * hidden_channels)  # hidden channels
        self.cv1 = Conv(in_channels, c_, kernel_size[0], 1)
        self.cv2 = Conv(c_, out_channels, kernel_size[1], 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@TTModuleRegistry.register
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, c3k=False, hidden_channels=0.5, groups=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(in_channels, out_channels, n, shortcut, groups, hidden_channels)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, groups) if c3k else Bottleneck(self.c, self.c, shortcut, groups) for _ in range(n)
        )


@TTModuleRegistry.register
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, hidden_channels=0.5, kernel_size=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(in_channels, out_channels, n, shortcut, groups, hidden_channels)
        c_ = int(out_channels * hidden_channels)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, groups, kernel_size=(kernel_size, kernel_size), hidden_channels=1.0) for _ in range(n)))


@TTModuleRegistry.register
class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, in_channels, out_channels, n=1, hidden_channels=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert in_channels == out_channels
        self.c = int(in_channels * hidden_channels)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, in_channels, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


@TTModuleRegistry.register
class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


@TTModuleRegistry.register
class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, groups=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


@TTModuleRegistry.register
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


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
