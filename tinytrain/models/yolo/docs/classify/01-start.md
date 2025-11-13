**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

# 简介
***
1. YOLO图像分类算法来自开源框架 *ultralytics*，链接：https://github.com/ultralytics/ultralytics
2. 该算法支持多个不同的模型结构，具体可查阅该文件父目录下的`task/classify/cfg/model`目录，来切换不同的模型。

# 数据集构建
***
1. 构建YOLO分类数据集时使用通用分类数据集格式，可以参考Pytorch的ImageFloder读取数据集的目录格式。
2. 支持多个训练集和验证集。

数据集目录格式如下：
<div>dataset/</div>
<div>├── train/</div>
<div>│   ├── cat/</div>
<div>│   │   ├── cat_001.jpg</div>
<div>│   │   ├── cat_002.jpg</div>
<div>│   │   └── ...</div>
<div>│   ├── dog/</div>
<div>│   │   ├── dog_001.jpg</div>
<div>│   │   ├── dog_002.jpg</div>
<div>│   │   └── ...</div>
<div>│   └── ...</div>
<div>│</div>
<div>├── val/</div>
<div>│   ├── cat/</div>
<div>│   │   ├── cat_003.jpg</div>
<div>│   │   ├── cat_004.jpg</div>
<div>│   │   └── ...</div>
<div>│   ├── dog/</div>
<div>│   │   ├── dog_003.jpg</div>
<div>│   │   ├── dog_004.jpg</div>
<div>│   │   └── ...</div>
<div>│   └── ...</div>

说明
- **`dataset/`**：数据集的根目录。
- **`train/`**：训练集目录，包含所有用于训练的图像。
  - **`class1/`**：类别1的图像目录。
  - **`class2/`**：类别2的图像目录。
  - **`...`**：其他类别的图像目录。
- **`val/`**：验证集目录，包含所有用于验证的图像。
  - **`class1/`**：类别1的图像目录。
  - **`class2/`**：类别2的图像目录。
  - **`...`**：其他类别的图像目录。

# 模型训练
***
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO分类模型的link.toml文件
yolo = YOLOCore(link_file="../../task/classify/link.toml")

# 可通过override覆盖配置文件参数
yolo.set_config_overrides(
  link_type="core",
  task="classify",  # 指定task为classify
  warmup_epochs=2,
  epochs=10,
  batch_size=16,
  lr0=1e-2,
  lr1=1e-4,
  scheduler="auto",
  workers=1,
  launch_tb=False,
  amp=False,
)

yolo.set_config_overrides(
  link_type="dataset",
  img_size=28,
  cache=False
)

# 启动训练
yolo.train(model_scale='n')
```

# 模型推理
***
推理时传入的source可以是图片、视频、摄像头索引等。
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO分类模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="classify",  # 指定task为classify
)

results = yolo.predict(
        use_best_pt=True,
        source="1.png",
        img_shape=(28, 28),
    )
for result in results:
    print(result)
```

# 模型导出
***
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO分类模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="classify",  # 指定task为classify
)

yolo.export(
        use_best_pt=True,
        backend="onnx",  # 导出到onnx平台
        input_shapes=[(1, 3, 28, 28)],
        # jit_export=True,
        opset_version=11
    )
```

# 超参数搜索
***
YOLO分类算法支持进行超参数搜索，以便在训练前寻找最优超参数。
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO分类模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="classify",  # 指定task为classify
)

# search_result里包括最优超参数结果
search_result = yolo.tune(
    model_scale='n',
    pop_size=40,
    generations=20
)
```