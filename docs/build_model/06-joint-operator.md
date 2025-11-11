# 学习如何使用joint算子

joint算子是一系列用于将模块接受的多个输入融合成单个输出的算子，是一种特殊的算子。更多细节请参考源码中的：tinytrain/modules/joints.py文件。

请确保在学习该章节前已学习前置章节：[学习更高级的模型构建技巧](04-advanced-config-tips)

下面介绍几种常用的算子。

## Concat

与pytorch的concat功能一致，接受来自上一层的多个输出，并将所有输出按指定的维度拼接后作为下一层的输入。

```toml
# 假设模型的外界输入维度为:[1,3,64,64]

[[network]]
# 0  输出维度:[1,16,64,64]
type = "entry"
module = "My_CBA"
repeat = 2
from = [-1]
args.in_channels = 3
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 1  输出维度:[1,32,64,64]
type = "flow"
module = "My_CBA"
repeat = 1
from = [-1]
args.in_channels = 16
args.out_channels = 32
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 2  输出维度:[1,48,64,64]
type = "flow"
module = "Concat"
repeat = 1
from = [0, -1]  # 将第0层模块与上一层模块进行拼接 
args.dim = 1   # 沿维度1拼接
```

注意：from列表里元素的顺序会影响拼接的顺序，拼接按列表从左往右进行拼接。

Concat算子的`from=[0, -1]`，也可以写成`from=[0, 1]`，-1即表示它的上一层即第1层。但使用-1而不是直接指明层级的好处在于：
1. 直接指明的层级会作为临时变量保存，需要占用内存，并稍微增加计算时间。
2. 使用-1可以指明一条完整的数据的流向。

## Add

与pytorch的add功能一致，接受来自上一层的多个输出，并将所有输出按指定的维度做element-wise加法后作为下一层的输入。

```toml
# 假设模型的外界输入维度为:[1,3,64,64]

[[network]]
# 0  输出维度:[1,16,64,64]
type = "entry"
module = "My_CBA"
repeat = 2
from = [-1]
args.in_channels = 3
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 1  输出维度:[1,16,64,64]
type = "flow"
module = "My_CBA"
repeat = 1
from = [-1]
args.in_channels = 16
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 2  输出维度:[1,16,64,64]
type = "flow"
module = "Concat"
repeat = 1
from = [0, -1]  # 将第0层模块与上一层模块进行element-wise相加
```

注意：from列表里元素的顺序会影响相加的顺序，按列表从左往右进行相加。

## Combine

combine用于将自上一层的多个输出放进一个列表中，并将此列表返回作为下一层的输入。

首先定义一个能接受列表的AI算子：
```python
import torch

from torch import nn
from tinytrain import TTModuleRegistry


@TTModuleRegistry.register
class My_Fuse(nn.Module):
    """该算子做的事为接受多个特征图作为输入，并经过卷积后按通道拼接。"""
    def __init__(self, in_channels_list: list, out_channels):
        super().__init__()
        self.conv_list = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) for in_channels in in_channels_list]

    def forward(self, x: list):
        y = []
        for i, x_i in enumerate(x):
            y.append(self.conv_list[i](x_i))
            
        return torch.concat(y, dim=1)
```

以下展示Combine算子的配置使用方式：
```toml
# 假设模型的外界输入维度为:[1,3,64,64]

[[network]]
# 0  输出维度:[1,16,64,64]
type = "entry"
module = "My_CBA"
repeat = 2
from = [-1]
args.in_channels = 3
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 1  输出维度:[1,32,64,64]
type = "flow"
module = "My_CBA"
repeat = 1
from = [-1]
args.in_channels = 16
args.out_channels = 32
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
# 2  输出列表:[第0层的输出, 第1层的输出]
type = "flow"
module = "Combine"
repeat = 1
from = [0, -1]  # 获取第0层模块与上一层模块组合成列表
args.dim = 1   # 沿维度1拼接

[[network]]
# 3  获取上一层Combine的输入，输出维度:[1,128,64,64]
type = "flow"
module = "My_Fuse"
repeat = 1
from = [0, -1]
args.in_channels_list = [16, 32]  # 注意传递的通道数要和from里的层级的顺序一致
args.out_channels = 64
```