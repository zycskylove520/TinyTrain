该文档提供tinytrain框架训练好的人脸识别模型。

以下模型使用 7 张 NVIDIA TRX 4090训练：

| 模型                                    | 尺寸 | 准确率    | TPR@FAR=1e-3 | TPR@FAR=1e-4 | 欧式距离 | 训练集       | 验证集                 | 下载地址                                                             |
|---------------------------------------|----|--------|--------------|--------------|------|-----------|---------------------|------------------------------------------------------------------|
| mobilefacenet-n-1024-82.81%-1.58.onnx | n  | 96.31% | 82.81%       | 50.36%       | 1.58 | glint360K | lfw、cfp_fp、agedb_30 | [点击下载](https://pan.baidu.com/s/1iSwHFH3QHi1f-MCr7fieiw?pwd=b3n4) |
| mobilefacenet-s-1024-94.34%-1.57.onnx | s  | 98.36% | 94.34%       | 78.44%       | 1.57 | glint360K | lfw、cfp_fp、agedb_30 | [点击下载](https://pan.baidu.com/s/1iSwHFH3QHi1f-MCr7fieiw?pwd=b3n4) |
| mobilefacenet-m-1024-97.33%-1.56.onnx | m  | 98.78% | 97.33%       | 84.41%       | 1.56 | glint360K | lfw、cfp_fp、agedb_30 | [点击下载](https://pan.baidu.com/s/1iSwHFH3QHi1f-MCr7fieiw?pwd=b3n4) |
