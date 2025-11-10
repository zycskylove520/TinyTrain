# **学习创建自定义AI模块**

如果希望通过**toml**或**yaml/yml**文件来构建模型，那么该章节是必须要了解的。下面将会引导你学习如何创建自定义的AI模块。

在学习完本章节后请继续学习：[从配置文件构建模型](02-build-config-model)

## 了解TTModuleRegistry

**TTModuleRegistry**是tinytrain框架AI模块的注册器。该类负责扫描所有的使用**TTModuleRegistry**装饰器装饰的AI模块加入到注册表中注册，并分发出去。
**TTModuleRegistry**仅支持**Pytorch**框架，进一步来说仅支持机继承自**nn.Module**的算子进行注册。

### 注册

#### 装饰器注册

使用装饰器的方式注册是最常用的也是推荐的注册方式。

```python
from torch import nn
from tinytrain import TTModuleRegistry


# 装饰器方式：无参
@TTModuleRegistry.register
class MyModule1(nn.Module):
    pass


# 装饰器方式：指定单个别名
@TTModuleRegistry.register("MM2")
class MyModule2(nn.Module):
    pass


# 装饰器方式：指定多个别名
@TTModuleRegistry.register("MMod", "MM3", ...)
class MyModule3(nn.Module):
    pass
```

<div>无参装饰器注册的情况下，类的名称会作为别名注册到**TTModuleRegistry**类的**MODULE_REGISTRY**类变量中保存。</div>
<div>有参装饰器注册支持为注册的模块赋予多个名称。请注意名称之间不要冲突，否则会抛出异常，提示别名冲突要求改正。</div>

**注意**：使用有参装饰器的情况下，不会将类的名称也进行注册，因此如果需要将模块的类名注册的情况下，再为它注册一个别名，请使用如下写法：

```python
from torch import nn
from tinytrain import TTModuleRegistry


@TTModuleRegistry.register("MyModule3", "MM3")  # 第一个别名为类名本身，后面可以继续添加其他别名
class MyModule3(nn.Module):
    pass
```

#### 插件注册

<div>当使用装饰器标记需要注册的模块后，如果注册的模块不处于tinytrain框架的tinytrain目录下，那么还需要通过插件注册的方式来激活。</div>
<div>假设您所编写的模块处于其他项目的目录如：/home/modules/blocks/cnn.py，使用以下代码来注册模块插件：</div>

```python
from tinytrain import TTModuleRegistry

TTModuleRegistry.register_plugin("/home/modules")  # 整包
TTModuleRegistry.register_plugin("/home/modules/cnn")  # 单模块
TTModuleRegistry.register_plugin("/home/modules/blocks/cnn.py")  # 单文件
```

**注意：目录使用路径需要为绝对路径。**
<div>注册插件时，如果注册的不是单个py文件，那么register_plugin函数支持第二个参数exclude来排除不需要注册的目录。</div>
<div>如果编写的模块处于tinytrain目录下，则不需要注册插件，tinytrain.__init__中已经为整个项目注册了插件。</div>

#### 函数式注册

使用该方式的优势在于灵活注册，不需要通过TTModuleRegistry.register_plugin函数进行插件注册，即使注册的模块不在tinytrain框架的tinytrain目录下。
方式如下：

```python
from torch import nn
from tinytrain import TTModuleRegistry


class MyModule4(nn.Module):
    pass


TTModuleRegistry.register_name(MyModule4)  # 无别名注册
TTModuleRegistry.register_name(MyModule4, "MM4", ...)  # 有别名注册
```

注册效果与装饰器注册一致。
