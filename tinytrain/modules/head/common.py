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

from tinytrain.cfg import TTModuleRegistry
from tinytrain.modules.conv import CBA


@TTModuleRegistry.register
class ClassifyHead(nn.Module):
    """
    通用分类头。
    training模型下返回: [batch, nc]
    eval模式下返回: [batch, nc], nc已做softmax
    """

    def __init__(self, in_channels, nc, hidden_channels=1280, kernel_size=1, stride=1, padding=None, groups=1):
        """Initializes classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        self.conv = CBA(in_channels, hidden_channels, kernel_size, stride, padding, groups)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(batch, hidden_channels, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(hidden_channels, nc)  # to x(batch, out_channels)

    def forward(self, x):
        """Performs a forward pass of the YOLO model_config on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

        if not self.training:
            x = self.inference(x)
        return x

    def inference(self, x):
        x = F.softmax(x, dim=1)
        return x






