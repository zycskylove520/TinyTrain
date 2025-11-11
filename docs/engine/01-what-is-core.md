# TTBaseCore：引擎的核心

`TTBaseCore` 是 `TinyTrain` 框架的核心门面类，负责统一调度配置、模型、训练器、推理器、导出器等所有引擎组件，对外提供简洁易用的高级API接口。

## 主要特性
* 🎯 统一配置管理：支持链式配置文件（yaml/toml），支持运行时配置覆盖
* 🤖 自动引擎绑定：根据场景自动实例化训练器、推理器、导出器等组件
* 🔄 智能权重管理：自动搜索 last.pt / best.pt 等权重文件
* 🚀 分布式训练支持：内置 DDP 启动器，支持多GPU训练
* 📦 多格式导出：支持 ONNX、TensorRT、TorchScript 等后端导出
* 🧪 超参数调优：集成遗传算法进行自动超参数搜索
* 🎓 知识蒸馏：提供一站式知识蒸馏训练接口
* 🔧 设备自适应：自动检测并配置 CPU、CUDA、MPS 等设备

## 快速开始

### 引擎注册
`TTBaseCore`类无法直接使用，请继承该类，并重载`register_components`函数注册不同的`Engine`。

代码参考如下：
```python
class MyCore(TTBaseCore):
    @classmethod
    def register_components(cls):
        # 注册分类任务组件
        TTEngineRegistry.register(cls, task="classify", engine_type="model")(MyClassificationModel)
        TTEngineRegistry.register(cls, task="classify", engine_type="trainer")(MyClassificationTrainer)
        TTEngineRegistry.register(cls, task="classify", engine_type="validator")(MyClassificationValidator)
        TTEngineRegistry.register(cls, task="classify", engine_type="predictor")(MyClassificationPredictor)
        TTEngineRegistry.register(cls, task="classify", engine_type="exporter")(MyClassificationTrainer)
        TTEngineRegistry.register(cls, task="classify", engine_type="tuner")(MyClassificationValidator)
        TTEngineRegistry.register(cls, task="classify", engine_type="distiller")(MyClassificationPredictor)
```
一个`TTBaseCore`类可以实现并支持多个AI任务， 如上使用`TTEngineRegistry.register`为Core注册一个分类任务，其`engine_type`参数支持7种类型：`model`、`trainer`、`validator`、`predictor`、`exporter`、`tuner`和`distiller`，这七种引擎类型的名称不可修改。
task参数则可以根据自己喜好设定，但同一AI任务，其task参数的字符串必须相同。

当为`Core`类注册`model`后，继续添加：
- 注册`trainer`：支持模型训练。
- 注册`trainer`和`validator`：支持模型训练和验证。
- 注册`predictor`：支持模型推理。
- 注册`exporter`：支持模型导出。
- 注册`tuner`：支持超参数搜索。
- 注册`distiller`：支持模型蒸馏。

### 快捷配置覆盖
在构造`TTBaseCore`类的实例后，可以通过`set_config_overrides`函数来覆盖link文件里所指向的各种配置文件的值。

示例如下：
```python
core = TTBaseCore(link_file="link.toml")

# 覆盖核心配置
core.set_config_overrides(link_type='core', batch_size=32, workers=4)

# 覆盖模型配置
core.set_config_overrides(link_type='model', scale='l')

# 覆盖数据集配置  
core.set_config_overrides(link_type='dataset', cache=True)
```
`set_config_overrides`函数的`link_type`参数为link文件中的键名，指定`link_type`参数后，即可通过键值的方式为对应的配置文件重载参数。

