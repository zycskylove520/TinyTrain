**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

# 简介

***

1. 人脸识别算法来自开源框架 *insightface*，链接：https://github.com/deepinsight/insightface
2. 该算法可用于开集人脸识别或闭集人脸识别。
3. 该算法支持多个不同的模型结构，具体可查阅该文件父目录下的`task/recognition/cfg/model`目录，来切换不同的模型。

# 数据集构建

***

1. 训练集：构建人脸识别训练集时使用通用分类数据集格式，可以参考Pytorch的ImageFloder读取数据集的目录格式。
2. 验证集：构建人脸识别验证集时图片数据按配对形式，且需要额外再验证集图片目录下提供一个txt文件。
3. 仅支持传入一个训练集，但支持多个验证集。

数据集目录格式如下：
<div>├── train_dataset/</div>
<div>│   ├── face1/</div>
<div>│   │   ├── face1_001.jpg</div>
<div>│   │   ├── face1_002.jpg</div>
<div>│   │   └── ...</div>
<div>│   ├── face2/</div>
<div>│   │   ├── face2_001.jpg</div>
<div>│   │   ├── face2_002.jpg</div>
<div>│   │   └── ...</div>
<div>│   └── ...</div>
<div>│</div>
<div>├── val_dataset/</div>
<div>│   ├── class1/</div>
<div>│   │   ├── class1_001.jpg</div>
<div>│   │   └── class1_002.jpg</div>
<div>│   ├── class2/</div>
<div>│   │   ├── class2_001.jpg</div>
<div>│   │   └── class2_002.jpg</div>
<div>│   └── ...</div>
<div>│   └── val_pair.txt</div>

说明

- **`train_dataset/`**：训练集目录，包含所有用于训练的图像。
    - **`face1/`**：1号人员的人脸的图像目录。
    - **`face2/`**：2号人员的人脸的图像目录。
    - **`...`**：其他人员的人脸的图像目录。
- **`val_dataset/`**：验证集目录，包含所有用于验证的图像。
    - **`class1/`**：第1组人脸对的图像目录。
    - **`class2/`**：第2组人脸对的图像目录。
    - **`...`**：其他组人脸对的图像目录。
    - **`val_pair.txt`**: 验证集额外需要一个txt文件。

###### val_pair.txt文件格式如下:

```txt
class1/class1_001.jpg class1/class1_002.jpg 1
class2/class2_001.jpg class3/class3_001.jpg 0
```

- txt文件每一行需要来自当前验证集两张图片，第三个值为1表示两张图片为同一个人，而0则表示不是一个人。

# 模型训练

***

```python
from tinytrain import FaceCore

# 指定人脸识别模型的link.toml文件
face = FaceCore(link_file="../../task/recognition/link.toml")

# 可通过override覆盖配置文件参数
face.set_config_overrides(
    link_type="core",
    task="recognition",  # 指定task为recognition
    warmup_epochs=3,
    warmup_lr=1e-4,
    epochs=20,
    batch_size=16,
    lr0=1e-3,   # 推荐学习率
    lr1=1e-2,
    scheduler="LinearLR",
    workers=1
)

face.set_config_overrides(
    link_type="dataset",
    train_img_size=112,
    val_img_size=112,
)

# 启动训练
face.train(model_scale='n')
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
    img_shape=(112, 112),
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
    input_shapes=[(1, 3, 112, 112)],
    # jit_export=True,
    opset_version=11
)
```