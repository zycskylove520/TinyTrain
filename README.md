# <div style="text-align: center;">TinyTrain: 轻量级AI框架</div>

<div style="text-align: center;">轻量级 · 高效 · 易用的AI模型开发全流程框架</div>

***

<div style="text-align: center;">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=zycskylove520/TinyTrain)
[![GitHub Repo stars](https://img.shields.io/github/stars/zycskylove520/TinyTrain)](https://github.com/zycskylove520/TinyTrain/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/zycskylove520/TinyTrain)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/zycskylove520/TinyTrain)](https://github.com/zycskylove520/TinyTrain/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/zycskylove520/TinyTrain/pulls)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20macOS-brightgreen)

</div>

`TinyTrain` 是一个基于 `PyTorch` 构建的轻量级、弹性可扩展的深度学习框架，专为简化 AI 模型开发流程而设计。本框架具备高度模块化架构，支持快速上手和深度定制，可覆盖绝大多数AI任务。

## ✨ 核心特性
### 🚀 开发效率
- **开箱即用**: 预置经典模型和标准训练流程，快速启动项目
- **模块化设计**: 灵活组合训练组件，支持轻松定制和扩展
- **生产就绪**: 提供完整的训练、评估、可视化到模型导出全流程

### ⚡ 训练性能  
- **高效训练**: 支持分布式训练、混合精度、梯度累积等加速技术
- **轻量级架构**: 精简代码设计，低资源消耗，支持快速部署
- **多硬件支持**: 完整支持单机单卡、单机多卡及多机多卡训练(测试中)

### 🔧 实验管理
- **实验跟踪**: 完整的训练过程监控和超参数优化
- **弹性扩展**: 高度模块化架构，支持自定义模块和算法扩展

## 📦 功能支持

### 当前支持的算法系列

| 领域      | 支持任务                |
|---------|---------------------|
| YOLO系列  | 图像分类、目标检测、姿态估计、实例分割 |
| Face系列	 | 人脸识别	               |
| OCR系列	  | 车牌识别	               |

### 持续扩展中

我们正在积极适配更多 AI 任务和算法模型，欢迎社区贡献。

## 🏗️ 项目架构

```text
TinyTrain/
├── assets/          # 资源文件
├── cfg/            # 配置管理系统
├── data/           # 数据集与数据管道
├── engine/         # 核心引擎
├── global_var/     # 全局变量管理
├── labeling/       # 标注工具（开发中）
├── loss/           # 损失函数模块
├── metrics/        # 评估指标
├── models/         # AI模型
├── modules/        # 神经网络组件
├── server/         # 后端服务
├── test/           # 测试用例
├── tools/          # 实用工具集
├── tracker/        # 目标跟踪器
└── utils/          # 通用工具库
```

## 🔧 安装指南

### 快速安装

我们提供预编译的 whl 包，可直接安装使用：

```shell
pip install tinytrain-*.whl
```

### 源码安装

如需最新功能或进行二次开发，建议从源码安装：

```shell
git clone https://github.com/your-username/TinyTrain.git
cd TinyTrain

# 开发模式安装（推荐）
pip install -e .

# 或生产模式安装
pip install .
```

## 🎯 快速开始
`TinyTrain` 所有已实现的AI模型都在`tinytrain.models`目录下。

### 模型训练

`TinyTrain` 使用统一的配置管理系统，所有参数通过 `link.toml` 配置文件定义。以下以 YOLO 目标检测为例：

```python
from tinytrain.models.yolo import YOLOCore

# 初始化核心引擎
yolo = YOLOCore(r"link.toml")

# 覆盖训练参数
yolo.set_config_overrides(
    link_type="core",
    epochs=100,
    batch_size=32,
    save_dir="runs/"  # 训练结果输出目录
    # resume=True,    # 恢复训练
    # device=[0,1]   # 多GPU训练
)

# 覆盖数据参数
# 配置dataset相关参数
yolo.set_config_overrides(
    link_type="dataset",
    img_size=640
)

# 开始训练
yolo.train(model_scale='n')  # 从头训练 nano 模型

# 或从预训练权重开始
yolo.train(model_scale="n", model="pretrained.pt")
```

### 模型推理

#### 1. 训练后直接推理

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")
yolo.train(model_scale='n')

# 单张图像推理
results = yolo.predict(source=r"xxx.jpg")
for result in results:
    print(result)
```

#### 2. 加载训练好的模型

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# 使用最佳权重
results = yolo.predict(source=r"xxx.jpg", use_best_pt=True)
for result in results:
    print(result)

# 或指定权重文件
results = yolo.predict(source=r"xxx.jpg", model=r"xxx.pt")
for result in results:
    print(result)
```

#### 3. ONNX 运行时推理

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# 使用onnx进行推理
results = yolo.predict(
    model="xxx.onnx",
    backend="onnx",
    source=r"D:\project\python_code\TinyTrain-main\datasets\firework\images\val\2f730a7b97ef51dcbba8fe15aee67092.jpg",
    img_shape=(640, 640)
)
for result in results:
    print(result)
```

### 模型导出

当前支持 `ONNX` 格式导出，便于生产环境部署：

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# 导出为 ONNX 格式
yolo.export(
    use_best_pt=True,
    backend="onnx",
    input_shapes=[(1, 3, 640, 640)],
    # jit_export=True  # 支持 TorchScript 导出
)
```

## ⚙️ 配置系统

`TinyTrain` 采用 `TOML` 或 `yaml/yml` 格式的配置文件，所有训练参数集中管理，支持运行时动态覆盖.

## 🔮 未来路线图

* 测试多机多卡分布式训练
* 增加各种领域AI任务支持
* 开发可视化标注工具
* 优化模型压缩和量化功能
* 增加不同后端模型部署方案

## 🤝 贡献指南

我们欢迎社区贡献！请参阅 `CONTRIBUTING.md` 了解详细指南。

## 📄 许可证

本项目采用 `MIT` 许可证。详见 `LICENSE` 文件。

## 📞 联系我们

* GitHub: https://github.com/zycskylove520/TinyTrain
* Issues: `GitHub Issues` 或 QQ群：`271313723`
* Email: `zycskylove520@gmail.com` 或 `867245713@qq.com`

---
<div style="text-align: center;"> <strong>TinyTrain</strong> - 让 AI 开发更简单、更高效 </div>