# 从配置文件构建AI模型

该章节将引导你学习如何使用**toml**或**yaml/yml**文件来构建一个简单的AI模型。

请确保在学习该章节前已学习前置章节：[学习创建自定义AI模块](01-create-module.md)

## TTConfigModel类

**TTConfigModel**类是面向配置文件驱动的结构化模型构建基类，支持通过配置描述文件（TOML/YAML）动态解析并构建深度学习模型，几乎适用于构建所有的Pytorch的AI模型。
使用配置文件的方式来建立AI模型，优势有：

1. 动态伸缩性：通过调整配置文件中的repeat参数来实现同样架构不同尺寸的模型。
2. 高效替换：修改配置文件即可替换模型中某个模块成其他模块，在测试前沿AI模块和寻找更好的模型架构时具有高效率。

## 1. 创建自定义AI模块

通常情况下，我们会将Conv2d+BatchNorm2d+ReLU组合成一个模块用于搭建AI模块。实现如下：

```python
from torch import nn
from tinytrain import TTModuleRegistry


@TTModuleRegistry.register
class My_CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x
```

## 2. 编写配置文件

使用toml文件来编写模型结构，创建my_model.toml文件，编写要求：

1. 必须存在name字段，该字段表示模型的名称。
2. 必须存在scale字段，该字段表示模型当前使用的默认尺寸。
3. 必须存在scales字段，该字段表示模型的不同尺寸，scale字段里设置的尺寸必须来自scales字段。
4. 必须存在network字段，该字段定义模型的结构。
5. 定义的模型，**至少需要一个entry和head模块，可以没有flow模块。**

以下展示一个简单的CNN模型：

```toml
name = "My_CNN_Net"
scale = "n"

# 下面定义了四种不同的模型尺寸，根据depth逐级增大
[scales.n]
depth = 0.50
summary = "this is a nano model."

[scales.s]
depth = 1
summary = "this is a small model."

[scales.m]
depth = 2
summary = "this is a medium model."

[scales.l]
depth = 3
summary = "this is a large model."

[[network]]
type = "entry"  # 使用entry来标记模型的输入
module = "My_CBA"  # 使用自定义的My_CBA算子
repeat = 1
from = [-1]
args.in_channels = 3
args.out_channels = 16
args.kernel_size = 3
args.stride = 2

[[network]]
type = "flow"  # 使用flow来标记模型的中间层
module = "My_CBA"
repeat = 2
from = [-1]
args.in_channels = 16
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1

[[network]]
type = "head"  # 使用head来标记模型的输出
module = "My_CBA"
repeat = 1
from = [-1]
args.in_channels = 16
args.out_channels = 32
args.kernel_size = 3
args.stride = 1
args.padding = 1
```

以上展示了一个标准的三层模块的AI模型，解释一下上面的字段：

1. scales字段中的depth字段决定了模型的尺寸，summary则是对该尺寸的说明，这两个字段都是必须的。
2. network字段：
    - type：模块的类型，有三种:entry、flow、head。其中entry表示模型的输入层，并且模型的第一层也必须是entry层。flow层为模型的中间层，可以有任意多层，也可以没有。head层为模型的输出层，模型最终的输出会由head层输出。
    - module：模块使用的AI算子名，即使用TTModuleRegistry类注册的AI模块，也可以是支持的第三方模块，比如直接使用`torch.nn.Conv_2d`。如果希望知道支持哪些第三方模块，请查看代码：`TTConfigModel.get_layer`
      函数中的candidate_pkgs变量，可自行增加更多第三方库。
    - repeat：定义该模块的重复次数，如果该模块的repeat次数大于1，则要求该模块的输入尺寸完全等价于该模块的输出尺寸，否则无法repeat。repeat大于1的情况下，repeat的次数等于`repeat * depth`
      ，如果计算结果为浮点数，则向上取整。通过不同的scales和repeat的组合就形成了同结构不同尺寸的模型。
    - from：模块接收哪一层的输入，必须为列表，至少一个元素。如果为-1表示接收上一层的输入，如果该模块为第一层模块，则-1表示接收外界的输入。可以接收多层的输入，只能使用-1和大于等于0的整数，如果为大于等于0的整数则表示接收对应层的模块的输入。
    - args：模块对应的AI算子的参数，要求必须和AI算子的参数名完全一致。

上面使用toml配置文件定义的模型，也可以使用一下yaml文件来定义：

```yaml
name: My_CNN_Net
scale: n

scales:
  n: { depth: 0.5, summary: "this is a nano model." }
  s: { depth: 1, summary: "this is a small model." }
  m: { depth: 2, summary: "this is a medium model." }
  l: { depth: 3, summary: "this is a large model." }

network:
  - { type: entry, from: [ -1 ], module: My_CBA, repeat: 1, args: { in_channels: 3, out_channels: 16, kernel_size: 3, stride: 2 } }
  - { type: flow, from: [ -1 ], module: My_CBA, repeat: 2, args: { in_channels: 16, out_channels: 16, kernel_size: 3, stride: 1, padding: 1 } }
  - { type: head, from: [ -1 ], module: My_CBA, repeat: 1, args: { in_channels: 16, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 } }
```

用户可根据自己喜好使用以上两种配置方式的任意一种。

## 3. 构建模型

在使用配置文件构建模型之前，需要先创建一个link文件，该文件可以是toml或yaml/yml文件，用于链接不同的配置文件。通过以下链接深入了解：[什么是TTConfigManager](../core/01-what-is-TTConfigManager.md)

以下创建一个link.toml文件，其内容如下：

```toml
model = "my_model.toml"  # 键名必须是model
```

同样给出link.yaml版本：

```yaml
model: my_model.toml  # 键名必须是model
```

通过上面定义的模型配置文件和link文件，使用以下代码来生成模型：

### 极简形式
直接使用TTConfigModel基类来构建模型，该方式构建的模型无法使用tinytrain框架进行训练、推理或导出。一半用于测试模型，或仅使用tinytrain框架的模型配置生成模型，而用其他框架或自己编写代码进行训练、推理或导出。
```python
import torch

from torch import nn
from tinytrain.engine import TTConfigModel
from tinytrain.cfg import TTConfigManager

if __name__ == '__main__':
    # 创建ConfigManager
    cfg = TTConfigManager(link_file="link.toml")  # 指定link文件
    model: nn.Module = TTConfigModel(config_manager=cfg)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    res = model(x)
    print(res)
```

### 标准形式
创建一个继承自TTConfigModel的类，该方式构建的模型支持使用tinytrain框架进行训练、推理或导出。
```python
import torch

from torch import nn
from tinytrain.engine import TTConfigModel
from tinytrain.cfg import TTConfigManager


# 创建模型
class MyModel(TTConfigModel):
    pass


if __name__ == '__main__':
    config_manager = TTConfigManager("link.toml")
    model: nn.Module = MyModel(config_manager)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    res = model(x)
    print(res)
```