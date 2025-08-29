# <div align="center">TinyTrain:轻量级AI框架</div>
***
TinyTrain 是一个基于 PyTorch 的轻量级弹性可扩展 AI 框架，易于上手，支持高度自定义修改，能够实现绝大多数 AI 任务。该框架支持单机单卡、单机多卡以及多机多卡（尚未测试）训练模式。
目前，TinyTrain 已实现 YOLOv11 分类任务和目标检测任务，后续将不断增加更多 AI 任务适配。

### TinyTrain工作目录结构:
- assets:资产目录
- cfg:配置文件描述目录
- data:数据集目录
- engine:核心目录
- global_var:全局变量目录
- loss:损失目录
- metrics:评估指标目录
- modules:神经网络模块目录
- server:各种后端服务
- test:测试示例目录
- tools:工具目录
- tracker:不同的跟踪器目录
- utils:其他目录

# 安装tinytrain
项目提供了 whl 文件，可直接安装。建议安装后配合源码使用。
你也可以通过以下命令从源码安装（在 TinyTrain-main 目录下）：
```shell
pip install -e .
```

下面以YOLO目标检测模型来做示例。

# 模型训练
***
TinyTrain 提供了 project 目录，可用于直接训练模型。
你也可以创建一个新的脚本，使用以下代码开启模型训练：

```python
from tinytrain import YOLOCore

yolo = YOLOCore(r"link.toml")
# 配置core相关参数
yolo.set_config_overrides(link_type="core",
                          epochs=10,
                          # resume=False, 恢复训练
                          # device=[0,1]  支持多卡训练
                          )
# 配置dataset相关参数
yolo.set_config_overrides(
    link_type="dataset",
    img_size=640
)
# 直接训练
yolo.train(model_scale='n')
# 使用pt文件进行训练,如果设置了resume为True，则为恢复训练
yolo.train(model_scale="n", model="xxx.pt")
```
所有 AI 训练配置均通过 link.toml 文件指定，可从该文件访问所有可配置参数，并可通过 set_config_overrides 函数进行覆盖。若未指定训练结果目录，则默认生成在当前脚本执行的 runs 目录下。

# 模型推理
***
1. 训练完后直接推理

```python
from tinytrain import YOLOCore

yolo = YOLOCore(r"link.toml")
yolo.set_config_overrides(link_type="core",
                          epochs=10,
                          )
yolo.train(model_scale='n')

results = yolo.predict(source=r"xxx.jpg")
for result in results:
    print(result)
```

2. 使用pt文件进行推理

```python
from tinytrain import YOLOCore

yolo = YOLOCore(r"link.toml")

# 指定pt文件进行推理
results = yolo.predict(source=r"xxx.jpg", model=r"xxx.pt")
for result in results:
    print(result)
    
# 使用最新训练的pt文件进行推理
results = yolo.predict(source=r"xxx.jpg", use_best_pt=True)
for result in results:
    print(result)
```

3. 使用onnx文件进行推理

```python
from tinytrain import YOLOCore

yolo = YOLOCore(r"link.toml")

# 使用onnx进行推理
results = yolo.predict(
    model="xxx.onnx",
    engine="onnx",
    source=r"D:\project\python_code\TinyTrain-main\datasets\firework\images\val\2f730a7b97ef51dcbba8fe15aee67092.jpg",
    img_shape=(640, 640)
)
for result in results:
    print(result)
```

# 模型导出
***
目前只做了onnx导出适配。

```python
from tinytrain import YOLOCore

yolo = YOLOCore(r"link.toml")

# 导出onnx
yolo.export(
    use_best_pt=True,
    backend="onnx",
    input_shapes=[(1, 3, 640, 640)],
    # jit_export=True  # 支持jit导出
)
```