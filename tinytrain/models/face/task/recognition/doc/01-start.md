**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

_该人脸识别算法可用于开集人脸识别或闭集人脸识别。_

YOLO分类算法支持多个不同的模型结构，具体可查阅当前目录下的`cfg/model`目录，来切换不同的模型。

# 数据集构建
***
构建人脸识别数据集时使用通用分类数据集格式，可以参考Pytorch的ImageFloder读取数据集的目录格式。

数据集目录格式如下：
root/
├── train/
│   ├── face1/
│   │   ├── face1_001.jpg
│   │   ├── face1_002.jpg
│   │   └── ...
│   ├── face2/
│   │   ├── face2_001.jpg
│   │   ├── face2_002.jpg
│   │   └── ...
│   ├── face3/
│   │   ├── face3_001.jpg
│   │   ├── face3_002.jpg
│   │   └── ...
│   └── ...
│
├── val/
│   ├── face1/
│   │   ├── face1_003.jpg
│   │   ├── face1_004.jpg
│   │   └── ...
│   ├── face2/
│   │   ├── face2_003.jpg
│   │   ├── face2_004.jpg
│   │   └── ...
│   ├── face3/
│   │   ├── face3_003.jpg
│   │   ├── face3_004.jpg
│   │   └── ...
│   └── ...

## 说明
- **`root/`**：数据集的根目录。
- **`train/`**：训练集目录，包含所有用于训练的图像。
  - **`face1/`**：1号人员的人脸的图像目录。
  - **`face2/`**：2号人员的人脸的图像目录。
  - **`...`**：其他人员的人脸的图像目录。
- **`val/`**：验证集目录，包含所有用于验证的图像。
  - **`face1/`**：1号人员的人脸的图像目录。
  - **`face2/`**：2号人员的人脸的图像目录。
  - **`...`**：其他人员的人脸的图像目录。

# 模型训练
***

```python
from tinytrain import FaceCore

# 指定人脸识别模型的link.toml文件
face = FaceCore(link_file="../link.toml")

# 可通过override覆盖配置文件参数
face.set_config_overrides(
    link_type="core",
    task="recognition",  # 指定task为recognition
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

face.set_config_overrides(
    link_type="dataset",
    img_size=28,
    cache=False
)

# 启动训练
face.train(model_scale='l')
```

# 模型推理
***
```python
from tinytrain import FaceCore

# 指定人脸识别模型的link.toml文件
face = FaceCore(link_file="link.toml")
face.set_config_overrides(
    link_type="core",
    task="recognition",  # 指定task为recognition
)

# source传入列表，使用两张图片进行推理，推理方式为计算两张图片的相似度
results = face.predict(
        use_best_pt=True,
        source=["1.png", "2.png"],
        img_shape=(28, 28),
    )
for result in results:
    print(result)
```

# 模型导出
***
人脸识别模型导出到第三方引擎的模型最终输出为人脸的特征向量。
```python
from tinytrain import FaceCore

# 指定人脸识别模型的link.toml文件
face = FaceCore(link_file="link.toml")
face.set_config_overrides(
    link_type="core",
    task="recognition",  # 指定task为recognition
)

face.export(
        use_best_pt=True,
        backend="onnx",  # 导出到onnx平台
        input_shapes=[(1, 3, 28, 28)],
        # jit_export=True,
        opset_version=11
    )
```