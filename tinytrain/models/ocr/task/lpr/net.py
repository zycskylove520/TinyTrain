import torch

from torch import nn

from tinytrain import TTModuleRegistry


class LPRBaseBlock(nn.Module):
    """LPRNet base block"""

    def __init__(self, in_channels, out_channels):
        super(LPRBaseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


@TTModuleRegistry.register
class LPRBackbone(nn.Module):
    """LPRNet backbone"""

    def __init__(self, nc, dropout_rate):
        super(LPRBackbone, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, bias=False),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1)),
            LPRBaseBlock(in_channels=64, out_channels=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2), dilation=(1, 1, 1)),
            LPRBaseBlock(in_channels=64, out_channels=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            LPRBaseBlock(in_channels=256, out_channels=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2), dilation=(1, 1, 1)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1, bias=False),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=nc, kernel_size=(13, 1), stride=1, bias=False),  # 20
            nn.BatchNorm2d(num_features=nc),
            nn.ReLU(),  # *** 22 ***
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 6, 13, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow) + 1e-8
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        return x


@TTModuleRegistry.register
class LPRHead(nn.Module):
    """LPRNet head"""

    def __init__(self, nc):
        super(LPRHead, self).__init__()
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + nc, out_channels=nc, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits
