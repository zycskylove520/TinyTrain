import torch

from torch import nn

from tinytrain import TTModuleRegistry
from tinytrain.modules.block.common import SSH
from tinytrain.modules.conv import CBA


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = nn.init.xavier_uniform(1)
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0.01


@TTModuleRegistry.register
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
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


@TTModuleRegistry.register
class LPRBackbone(nn.Module):
    """LPRNet backbone"""

    def __init__(self, nc, dropout_rate):
        super(LPRBackbone, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64), # 1
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1)),  # 3
            LPRBaseBlock(in_channels=64, out_channels=128),  # 4
            nn.BatchNorm2d(num_features=128),  # 5
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2), dilation=(1, 1, 1)),  # 7
            LPRBaseBlock(in_channels=64, out_channels=256),  # 8
            nn.BatchNorm2d(num_features=256),  # 9
            nn.ReLU(),  # 10
            LPRBaseBlock(in_channels=256, out_channels=256),  # 11
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),  # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2), dilation=(1, 1, 1)),  # 14
            nn.Dropout(dropout_rate),  # 15
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),  # 17
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),  # 19
            nn.Conv2d(in_channels=256, out_channels=nc, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=nc),  # 21
            nn.ReLU(),  # 22
        )

        self.apply(weights_init)

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        return x


@TTModuleRegistry.register
class LPRHead(nn.Module):
    """LPRNet head"""

    def __init__(self, in_channels, nc):
        super(LPRHead, self).__init__()
        self.container = nn.Conv2d(in_channels=in_channels, out_channels=nc, kernel_size=(1, 1), stride=(1, 1))

        self.apply(weights_init)

    def forward(self, x):
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


@TTModuleRegistry.register
class HalfPAFPN(nn.Module):
    def __init__(self, in_channels_list: list, out_channels):
        super(HalfPAFPN, self).__init__()
        self.num_in = len(in_channels_list)

        self.lateral_convs = nn.ModuleList()
        self.ssh_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(CBA(in_channels, out_channels, kernel_size=1, stride=1))
            self.ssh_convs.append(SSH(out_channels, out_channels))

        for _ in range(self.num_in - 1):
            self.fuse_convs.append(CBA(out_channels, out_channels, kernel_size=3, stride=(1, 2)))

    def forward(self, x: list):
        # 1. lateral 降维
        laterals = [lat(x) for lat, x in zip(self.lateral_convs, x)]

        # 2. FPN策略
        for i in range(self.num_in - 1, 0, -1):
            upsampled = torch.nn.functional.interpolate(laterals[i], size=laterals[i - 1].shape[-2:],
                                                        mode='nearest')
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3. SSH信息聚合
        sshs = [ssh(x) for ssh, x in zip(self.ssh_convs, laterals)]

        # 4. PAN策略
        fuse = sshs[-1]
        for i in range(self.num_in - 1, 0, -1):
            # up = torch.nn.functional.interpolate(sshs[i], size=sshs[i - 1].shape[-2:], mode='nearest')
            # sshs[i - 1] = sshs[i - 1] + up
            up = torch.nn.functional.interpolate(fuse, size=sshs[i - 1].shape[-2:], mode='nearest')
            fuse = up + sshs[i - 1]

        # 5. 信息融合
        # fuse = sshs[-1]
        # for i in range(self.num_in - 1,0 ,-1):
        #     fuse = self.fuse_convs[i](fuse) + sshs[i + 1]
        return fuse
