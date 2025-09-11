**在阅读该md文件前，请确保您已阅读tinytrain目录下的README.md文件！**

_该人脸识别算法可用于开集人脸识别或闭集人脸识别。_

**注意：使用该算法训练模型，至少需要2张NVIDIA GPU！！！**

YOLO分类算法支持多个不同的模型结构，具体可查阅当前目录下的`cfg/model`目录，来切换不同的模型。

# 数据集构建
***
训练集：构建人脸识别训练集时使用通用分类数据集格式，可以参考Pytorch的ImageFloder读取数据集的目录格式。
验证集：构建人脸识别验证集时图片数据按配对形式，且需要额外再验证集图片目录下提供一个txt文件。

数据集目录格式如下：
├── train_dataset/
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
├── val_dataset/
│   ├── class1/
│   │   ├── class1_001.jpg
│   │   └── class1_002.jpg
│   ├── class2/
│   │   ├── class2_001.jpg
│   │   └── class2_002.jpg
│   ├── class3/
│   │   ├── class3_001.jpg
│   │   └── class3_002.jpg
│   └── ...
│   └── val_pair.txt

## 说明
- **`train_dataset/`**：训练集目录，包含所有用于训练的图像。
  - **`face1/`**：1号人员的人脸的图像目录。
  - **`face2/`**：2号人员的人脸的图像目录。
  - **`...`**：其他人员的人脸的图像目录。
- **`val_dataset/`**：验证集目录，包含所有用于验证的图像。
  - **`class1/`**：第1组人脸对的图像目录。
  - **`class2/`**：第2组人脸对的图像目录。
  - **`...`**：其他组人脸对的图像目录。\
  - **`val.txt`**: 验证集额外需要一个txt文件。

_val.txt文件格式如下:_
```txt
class1/class1_001.jpg class1/class1_002.jpg 1
class2/class2_001.jpg class3/class3_001.jpg 0
```
## 说明
- txt文件每一行需要来自当前验证集两张图片，第三个值为1表示两张图片为同一个人，而0则表示不是一个人。

### 训练数据集下载链接：
以下是转换成训练数据集格式的数据集，需要可自行下载：
- `Glint360K数据集`：通过网盘分享的文件：Glint360K
链接: https://pan.baidu.com/s/1HZ5q2wWekKppVTu9l-IRdQ?pwd=dans 提取码: dans 
--来自百度网盘超级会员v7的分享
- `Glint-Mini数据集`：通过网盘分享的文件：glintmini
链接: https://pan.baidu.com/s/1Al239PqFfIi7wtM_dsPAPA?pwd=dtji 提取码: dtji 
--来自百度网盘超级会员v7的分享
- `Webface数据集`：通过网盘分享的文件：webface
链接: https://pan.baidu.com/s/1U8l2ewVsgko1JOgpFO5I4Q?pwd=ny4b 提取码: ny4b 
--来自百度网盘超级会员v7的分享
- `UMDFaces数据集`：通过网盘分享的文件：faces_umd
链接: https://pan.baidu.com/s/1LuvG2gqDzr9wqJOm9zLMQg?pwd=re9d 提取码: re9d 
--来自百度网盘超级会员v7的分享

### 验证数据集下载链接：
以下是转换成验证数据集格式的数据集，需要可自行下载：
- `lfw数据集`：通过网盘分享的文件：lfw-align-112x112-图片版.zip
链接: https://pan.baidu.com/s/1Emng3y0AVyQyKs0Txx8-Sg?pwd=4795 提取码: 4795 
--来自百度网盘超级会员v7的分享
- `cfp_fp数据集`：通过网盘分享的文件：cfp_fp-align-112x112-图片版.zip
链接: https://pan.baidu.com/s/1K7UrPQfgOkS2VYEUHPlXMQ?pwd=w45w 提取码: w45w 
--来自百度网盘超级会员v7的分享
- `agedb_30数据集`：通过网盘分享的文件：agedb_30-align-112x112-图片版.zip
链接: https://pan.baidu.com/s/1w13ItuzyvNv_Lksbdi8OEg?pwd=syqk 提取码: syqk 
--来自百度网盘超级会员v7的分享


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