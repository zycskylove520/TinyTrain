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






