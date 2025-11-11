# TinyTrain 数据容器

## 概述

该模块为 `TinyTrain` 训练框架提供了统一的数据容器系统，容器设计遵循面向对象原则，提供了灵活的动态字段管理和智能深拷贝功能。

## 核心特性

* 统一基类设计：所有数据容器继承自 BaseDataInfo，提供一致的接口
* 动态字段管理：支持任意关键字参数，自动绑定为成员变量
* 智能深拷贝：可配置跳过特定字段的深拷贝，优化性能
* 类型安全：完整的类型注解支持
* 任务专用：为不同任务提供专门化的数据容器

### BaseDataInfo

所有数据容器的基类，提供基础功能：

```python
class BaseDataInfo:
    def __init__(self, **kwargs): pass  # 动态字段绑定

    def __deepcopy__(self, memodict): pass  # 智能深拷贝
```

从`BaseDataInfo`派生出支持各种AI任务的容器，比如`ImgDataInfo`、`ClassifyDataInfo`、`DetectDataInfo`等容器。

### BaseBatchDataInfo

模型训练或验证过程中对单个数据进行合批而需要的批数据容器。

```python
class BaseBatchDataInfo:
    def __init__(self, data: torch.Tensor | list[torch.Tensor] | Any | None = None):
        pass
```

`BaseBatchDataInfo`类中的`data`为模型的输入数据，可以是任何数据，在`TTConfigModel`类和`TTEasyModel`类的`forward`函数中作为输入数据传递给模型进行前向推理。

从`BaseBatchDataInfo`派生出支持各种AI任务的容器，比如`ImgBatchDataInfo`、`ClassifyBatchDataInfo`、`DetectBatchDataInfo`等批数据容器。

## 使用指南

`BaseDataInfo`一般使用在构建数据集时，比如在继承自`torch.utils.data.Dataset`的子类的__getitem__中进行图像增强时，使用tinytrain框架提供的图像增强算法，比如：`DynamicFilling`、`DynamicScaling`、
`DynamicRotating`和`DynamicShearing`等图像增强算法中作为参数传递而使用。

在构建数据集时，自定义数据集类实现的`collate_fn`函数必须返回`BaseBatchDataInfo`实例，该实例将：

- 通过`TTBaseTrainer`类的`execute_forward`函数传递给`TTBaseModel`类的`forward`函数进行前向推理获得损失。
- 通过`TTBaseValidator`类的`inference`函数传递给`TTBaseModel`类的`inference`函数进行前向推理获得模型输出结果。

