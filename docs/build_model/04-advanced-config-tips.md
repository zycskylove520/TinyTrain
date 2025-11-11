# 学习更高级的模型构建技巧

该章节将介绍更高级的模型配置方法，您将学会搭建令人惊叹的先进模型。

请确保在学习该章节前已学习前置章节：[从配置文件构建AI模型](02-build-config-model.md)

## allow_repeat字段
allow_repeat字段允许repeat为1的符合要求的模块也能根据depth进行多次重复。默认为false，可以不在配置文件中使用。
```toml
[[network]]
type = "flow"  # 使用flow来标记模型的中间层
module = "My_CBA"
repeat = 1
allow_repeat = true  # 设置为true，该模块当允许进行repeat*depth计算
from = [-1]
args.in_channels = 16
args.out_channels = 16
args.kernel_size = 3
args.stride = 1
args.padding = 1
```

## 动态修改模型配置
绝大多数任务的配置参数不是绝对静态的，比如分类模型的最后一层需要输出n个类别，不同的数据集的类别数量不一致，因此需要能够动态修改模型参数。
以下使用一个简单的分类模型来展示这种动态修改的技巧。

### 创建需要的自定义模块
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

@TTModuleRegistry.register
class MyClassifyHead(nn.Module):
    def __init__(self, in_channels, nc, hidden_channels=1280, kernel_size=1, stride=1, padding=None):
        super().__init__()
        self.conv = My_CBA(in_channels, hidden_channels, kernel_size, stride, padding)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(batch, hidden_channels, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(hidden_channels, nc)  # to x(batch, out_channels)
```

### 使用toml文件搭建模型结构
该toml文件名称为：my_model.toml。
```toml
name = "My_Net"
scale = "n"

# 下面定义了四种不同的模型尺寸，根据depth逐级增大
[scales.n]
depth = 1
summary = "this is a nano model."

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
type = "head"  # 使用head来标记模型的输出
module = "MyClassifyHead"
repeat = 1
from = [-1]
args.in_channels = 16
args.nc = -1  # 随便写一个类别个数
args.kernel_size = 3
args.stride = 1
args.padding = 1
```

### 构建模型
假设我们通过一个数据集的配置文件来修改模型的类别个数，因此创建一个数据集配置文件：my_dataset.toml:
```toml
dataset_name = "mnist" 
nc = 10  # 用户可以通过外界修改nc的值
```
我们可以有很多份不同数据集的配置文件，创建一个link.toml文件，其内容如下：

```toml
model = "my_model.toml"  # 键名必须是model
dataset = "my_dataset.toml"
```

如果不知道为什么这么写，请参阅：[什么是TTConfigManager](../core/01-TTConfigManager)

为了实现动态修改MyClassifyHead算子的nc参数，我们需要重载TTConfigModel的custom_parse_model_level函数，在该函数中可以访问并修改任意network层的任意参数。
```python
from tinytrain.engine import TTConfigModel

# 创建模型
class MyModel(TTConfigModel):
    def custom_parse_model_level(self, level, module_info):
        # level是当前network模块的索引
        # module_info是当前network模块的参数，以字典的形式展示
        if module_info["type"] == "head":  # 也可以写成 if module_info["module"] == "MyClassifyHead":
            # 动态修改nc值为my_dataset.toml文件指向的nc值
            module_info["args"]["nc"] = self.config_manager.dataset["nc"]
```




