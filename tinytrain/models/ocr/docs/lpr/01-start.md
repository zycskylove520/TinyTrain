**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

# 简介
***
1. 车牌识别算法来自开源模型 *LPRNet*，链接：https://github.com/sirius-ai/LPRNet_Pytorch
2. 因为LRPNet要求输入图片宽高必须为[94,24], 因此目前只支持一种模型结构，具体可查阅当前目录下的`cfg/model`目录。

# 数据集构建
***
1. 构建车牌识别数据集时,使用YOLO数据集格式。
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
每个 `.txt` 标签文件对应一张图像，文件应只有一行，写车牌号码，格式如下：
```txt
粤BBV8528
```

# 模型训练
***

```python
from tinytrain import OCRCore

# 指定车牌识别模型的link.toml文件
ocr = OCRCore(link_file="../../task/lpr/link.toml")

# 可通过override覆盖配置文件参数
ocr.set_config_overrides(
  link_type="core",
  task="lpr",  # 指定task为lpr
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

# 启动训练
ocr.train(model_scale='n')
```

# 模型推理
***
推理时传入的source可以是图片、视频、摄像头索引等。
```python
from tinytrain import OCRCore

# 指定车牌识别模型的link.toml文件
ocr = OCRCore(link_file="link.toml")
ocr.set_config_overrides(
    link_type="core",
    task="lpr",  # 指定task为lpr
)

results = ocr.predict(
        use_best_pt=True,
        source="1.png",
    )
for result in results:
    print(result)
```

# 模型导出
***
```python
from tinytrain import OCRCore

# 指定车牌识别模型的link.toml文件
ocr = OCRCore(link_file="link.toml")
ocr.set_config_overrides(
    link_type="core",
    task="lpr",  # 指定task为lpr
)

ocr.export(
        use_best_pt=True,
        backend="onnx",  # 导出到onnx平台
        input_shapes=[(1, 3, 24, 94)],
        # jit_export=True,
        opset_version=11
    )
```

# 超参数搜索
***
车牌识别算法支持进行超参数搜索，以便在训练前寻找最优超参数。
```python
from tinytrain import OCRCore

# 指定车牌识别模型的link.toml文件
ocr = OCRCore(link_file="link.toml")
ocr.set_config_overrides(
    link_type="core",
    task="lpr",  # 指定task为lpr
)

# search_result里包括最优超参数结果
search_result = ocr.tune(
    model_scale='n',
    pop_size=40,
    generations=20
)
```