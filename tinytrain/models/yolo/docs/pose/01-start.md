**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

# 简介
***
1. YOLO姿态估计算法来自开源框架 *ultralytics*，链接：https://github.com/ultralytics/ultralytics
2. 该算法支持多个不同的模型结构，具体可查阅该文件父目录下的`task/detect/cfg/model`目录，来切换不同的模型。

# 数据集构建
***
1. 构建YOLO姿态估计数据集时使用YOLO姿态估计数据集格式。
2. 支持多个训练集和验证集。

数据集目录格式如下：
<div>dataset/</div>
<div>├── images/</div>
<div>│   ├── train/</div>
<div>│   │   ├── 1.jpg</div>
<div>│   │   ├── 2.jpg</div>
<div>│   │   └── ...</div>
<div>│   └── val/</div>
<div>│       ├── 3.jpg</div>
<div>│       ├── 4.jpg</div>
<div>│       └── ...</div>
<div>│</div>
<div>├── labels/</div>
<div>│   ├── train/</div>
<div>│   │   ├── 1.txt</div>
<div>│   │   ├── 2.txt</div>
<div>│   │   └── ...</div>
<div>│   └── val/</div>
<div>│       ├── 3.txt</div>
<div>│       ├── 4.txt</div>
<div>│       └── ...</div>

说明
- **`dataset/`**：数据集根目录。
- **`images/`**：包含所有图像文件。
  - **`train/`**：训练集图像，用于模型训练。
  - **`val/`**：验证集图像，用于模型验证。
- **`labels/`**：包含所有标签文件。
  - **`train/`**：训练集标签，与训练集图像对应。
  - **`val/`**：验证集标签，与验证集图像对应。

###### 文件命名规则
- 图像文件和标签文件的命名规则一致，例如：
  - 图像文件 `1.jpg` 对应的标签文件为 `1.txt`。
  - 图像文件 `2.jpg` 对应的标签文件为 `2.txt`。

###### txt文件内容规则
YOLO 姿态估计（YOLO-Pose）沿用 YOLOv5/v8 的扁平化文本格式，**每行**描述一个对象，结构为：
```text
<class_id> <x_center> <y_center> <width><height><x1><y1><mask_1><x2><y2><mask_2>...<xk><yk><mask_k>
```
- 所有坐标与宽高均已 **归一化到 0~1**（相对图像宽高）。
- 字段说明：
  1. `<class_id>` – 类别索引，从 `0` 开始。
  2. `<x_center> <y_center> <width> <height>` – 边界框中心、宽、高（归一化）。
  3. 关键点部分 – 共 `k` 个关键点，关键点的个数由dataset.toml中的keypoint_shape决定：
     - `<xi> <yi>` – 第 `i` 个关键点的 x、y 坐标（归一化）。
     - `<mask_i>` – 可见性标志  
       - `0`=未标注
       - `1`=被遮挡
       - `2`=可见

# 模型训练
***
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO姿态估计模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")

# 可通过override覆盖配置文件参数
yolo.set_config_overrides(
    link_type="core",
    task="pose",  # 指定task为pose
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
    img_size=640,
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

# 指定YOLO姿态估计模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="pose",  # 指定task为pose
)

results = yolo.predict(
        use_best_pt=True,
        source="1.png",
        img_shape=(640, 640),
    )
for result in results:
    print(result)
```

# 模型导出
***
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO姿态估计模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="pose",  # 指定task为pose
)

yolo.export(
        use_best_pt=True,
        backend="onnx",  # 导出到onnx平台
        input_shapes=[(1, 3, 640, 640)],
        # jit_export=True,
        opset_version=11
    )
```

# 目标跟踪
***
使用bytetrack跟踪算法对传入的视频或摄像头索引进行目标跟踪。
cfg目录下有tracker配置文件可进行跟踪相关参数配置。
***
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO姿态估计模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="pose",  # 指定task为pose
)

results = yolo.predict(
        use_best_pt=True,
        source="xxx.mp4",
        img_shape=(640, 640),
        track=True,  # 将该值设置为True开启跟踪
        track_backend="bytetrack",  # 指定跟踪算法为bytetrack
    )
for result in results:
    print(result)
```

# 超参数搜索
***
YOLO姿态估计算法支持进行超参数搜索，以便在训练前寻找最优超参数。
```python
from tinytrain.models.yolo import YOLOCore

# 指定YOLO姿态估计模型的link.toml文件
yolo = YOLOCore(link_file="link.toml")
yolo.set_config_overrides(
    link_type="core",
    task="pose",  # 指定task为pose
)

# search_result里包括最优超参数结果
search_result = yolo.tune(
    model_scale='n',
    pop_size=40,
    generations=20
)
```