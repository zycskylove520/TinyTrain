# 构建基于Pytorch的原生模型

该章节将引导你学习如何使用tinytrain框架来构建一个基于Pytorch的简单的AI模型。

## TTEasyModel类

TTEasyModel类是极简手动构建模型的基类，专为无需配置、快速原型开发、教学演示或高度定制化模型结构设计的轻量级建模入口。

## 1. 搭建nn.Module原生模型

```python
from torch import nn


# 创建自定义算子
class My_CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class MyNet(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        self.repeat = repeat

        self.layer1 = My_CBA(in_channels, 16, 3, 2)
        self.layer2 = My_CBA(16, 16, 3, 1, 1)
        self.layer3 = My_CBA(16, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.layer1(x)
        for _ in range(self.repeat):
            x = self.layer2(x)
        x = self.layer3(x)
        return x
```

# 构建模型

创建一个继承自TTEasyModel的类，并通过setup_model来返回一个基于Pytorch的原生nn.Module模型。

```python
import torch

from torch import nn
from tinytrain.cfg import TTConfigManager
from tinytrain.engine import TTEasyModel


# 创建模型
class MyModel(TTEasyModel):
    def setup_model(self):
        return MyNet(in_channels=3, out_channels=32, repeat=2)  # 在这里构建模型


if __name__ == '__main__':
    config_manager = TTConfigManager("link.toml")
    model: nn.Module = MyModel(config_manager)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    res = model(x)
    print(res)
```

注意，此时的link文件不需要model字段，因为不是通过配置文件构建的。

学习TTConfigManager类，请参考：[什么是TTConfigManager](../core/01-TTConfigManager)

学习通过配置文件构建模型，请参考：[学习创建自定义AI模块](01-create-module.md)