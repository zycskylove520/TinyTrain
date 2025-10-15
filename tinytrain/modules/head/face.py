import torch.nn as nn
import torch.nn.functional as F

from tinytrain.cfg import TTModuleRegistry
from tinytrain.modules.block.common import Conv2Linear
from tinytrain.modules.conv import CBA

@TTModuleRegistry.register
class GDCHead(nn.Module):
    """
    MobileFaceNet 人脸识别头。
    training模型下返回: [batch, embedding_size]
    eval模式下返回: [batch, embedding_size], embedding_size已做L2正则化
    """

    def __init__(self, in_channels, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            CBA(in_channels=in_channels, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            CBA(in_channels=512, out_channels=512, groups=512, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), act=False),
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        x = self.layers(x)

        if not self.training:
            x = F.normalize(x, p=2, dim=-1)
        return x


@TTModuleRegistry.register
class GeneralFaceHead(nn.Module):
    """
        通用人脸识别头。
        training模型下返回: [batch, embedding_size]
        eval模式下返回: [batch, embedding_size], embedding_size已做L2正则化
        """

    def __init__(self, in_channels, embedding_size):
        super().__init__()
        self.c2l = Conv2Linear(in_channels=in_channels, out_channels=embedding_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.c2l(x)

        if not self.training:
            x = F.normalize(x, p=2, dim=-1)
        return x