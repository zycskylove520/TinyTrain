<p align="center">
  <img src="tinytrain/assets/images/logo.png" alt="TinyTrain Logo"/>
</p>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/zycskylove520/TinyTrain)](https://github.com/zycskylove520/TinyTrain/stargazers)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=zycskylove520/TinyTrain)
[![GitHub last commit](https://img.shields.io/github/last-commit/zycskylove520/TinyTrain)](https://github.com/zycskylove520/TinyTrain/commits/master)
[![GitHub Code License](https://img.shields.io/github/license/zycskylove520/TinyTrain)](LICENSE)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/zycskylove520/TinyTrain/pulls)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20macOS-brightgreen)

</div>

<h3 align="center">Configuration-Driven Full-Stack AI Framework · Unlocking Infinite Possibilities for Model Construction</h3>

## 🎯 Framework Overview

**TinyTrain** is a **production-grade full-stack deep learning framework** built on PyTorch, dedicated to providing researchers and engineers with efficient and flexible development experiences for small to medium-sized models. Through highly modular architecture design, this framework significantly reduces the engineering complexity of AI projects while maintaining the flexibility and extensibility required for research.

## ✨ Core Features

### 🚀 Development Efficiency Enhancement

- **Out-of-the-box**: Pre-built classic models and standard training workflows for rapid project startup
- **Modular Design**: Flexible combination of training components with easy customization and extension support
- **Production Ready**: Complete workflow covering training, evaluation, visualization, and model export

### ⚡ Training Performance Optimization

- **Efficient Training**: Native support for DDP distributed training, mixed precision, gradient accumulation and other acceleration technologies
- **Lightweight Architecture**: Concise code design with low resource consumption for fast deployment
- **Multi-Hardware Support**: Full support for single-GPU, multi-GPU on a single machine, and multi-machine multi-GPU training (testing in progress)

### 🔧 Experiment Management

- **Configuration Driven**: Unified TOML/YAML construction of models
- **Experiment Tracking**: Complete training process monitoring and hyperparameter recording
- **Elastic Expansion**: Highly modular architecture supporting custom modules and algorithm extension

## 📊 Feature Matrix

| Domain    | Supported Tasks              | Main Models                                                          |
|---------|-----------------------------|-------------------------------------------------------------|
| YOLO Series  | Image Classification, Object Detection, Pose Estimation, Instance Segmentation | YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv11, YOLOv12, Retinaface-YOLO |
| Face Series	| Face Recognition	       | MobilefaceNet, ResNetFace, YOLOFace                           |
| OCR Series	  | License Plate Recognition	       | LPRNet, TTLPRNet                                             |

> 🔄 **Continuously Expanding** - We are actively adapting to more AI tasks and algorithm models

## 🏗️ Project Architecture

```text
TinyTrain/
├── assets/         # Resource files
├── cfg/            # Unified configuration management system
├── data/           # Datasets and data pipelines
├── engine/         # Core engine
├── global_var/     # Global variable management
├── labeling/       # Annotation tools (under development)
├── loss/           # Loss function implementations
├── metrics/        # Evaluation metrics
├── models/         # AI model architecture implementations
├── modules/        # Neural network basic components
├── server/         # Backend services
├── test/           # Test cases
├── tools/          # Utility scripts
├── tracker/        # Object trackers
├── utils/          # General utility library
└── docs/           # Documentation resources
```

## 🛠️ Installation Guide

### System Requirements

* Python ≥ 3.8 (**recommended 3.10+**)
* PyTorch ≥ 1.9.0 (**recommended 2.0+**)
* CUDA ≥ 11.0 (**recommended 11.8+ for GPU training**)

### Quick Installation

### Method One: Source Installation (Recommended)

For the latest features or secondary development, we recommend installing from source:

```shell
git clone https://github.com/zycskylove520/TinyTrain.git
cd TinyTrain-main

# Development mode installation (recommended)
pip install -e .

# Or production mode installation
pip install .
```

#### Method Two: Whl Installation

We provide pre-compiled whl packages for direct installation:

```shell
pip install tinytrain-*.whl
```

We recommend using the projects in the examples directory when installing with this method.

## 🎯 Quick Start

All AI models implemented in `TinyTrain` are available in the `tinytrain.models` module, providing a unified interface design.

### Model Training

TinyTrain uses configuration-driven training, supporting flexible runtime parameter overrides:

```python
from tinytrain.models.yolo import YOLOCore

# Initialize core engine
yolo = YOLOCore(link_file=r"link.toml")

# Override training parameters
yolo.set_config_overrides(
    link_type="core",
    epochs=100,
    batch_size=32,
    save_dir="runs/"  # Training result output directory
    # resume=True,    # Resume training
    # device=[0,1]   # Multi-GPU training
)

# Override data parameters
# Configure dataset-related parameters
yolo.set_config_overrides(
    link_type="dataset",
    img_size=640
)

# Start training
yolo.train(model_scale='n')  # Train nano model, model scale: n, s, m, l, x

# Or start from pretrained weights
yolo.train(model_scale="n", model="pretrained.pt")
```

### Model Inference

#### 1. Inference Directly After Training

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")
yolo.train(model_scale='n')

# Single image inference
results = yolo.predict(source=r"xxx.jpg")
for result in results:
    print(result)
```

#### 2. Load Trained Model

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# Use best weights
results = yolo.predict(source=r"xxx.jpg", use_best_pt=True)
for result in results:
    print(result)

# Or specify weight file
results = yolo.predict(source=r"xxx.jpg", model=r"xxx.pt")
for result in results:
    print(result)
```

#### 3. ONNX Runtime Inference

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# Use ONNX for inference
results = yolo.predict(
    model="xxx.onnx",
    backend="onnx",
    source=r"D:\project\python_code\TinyTrain-main\datasets\firework\images\val\2f730a7b97ef51dcbba8fe15aee67092.jpg",
    img_shape=(640, 640)
)
for result in results:
    print(result)
```

### Model Export

Supports multiple format exports for production environment deployment

```python
from tinytrain.models.yolo import YOLOCore

yolo = YOLOCore(r"link.toml")

# Export as ONNX format
yolo.export(
    use_best_pt=True,
    backend="onnx",
    input_shapes=[(1, 3, 640, 640)],
    # jit_export=True  # Supports TorchScript export
)
```

## 🖥️ Command Line Interface

TinyTrain provides powerful command line support for script-based deployment and automated training:

### Create Training Script train.py

```python
from tinytrain.models.yolo import YOLOCore

# Initialize core engine
yolo = YOLOCore(r"link.toml")

# Start training
yolo.train()
```

> 💡 **Tip**: All functionalities under Core classes (training, inference, export) support unified configuration override interface, ensuring consistent user experience.

### Command Line Training Examples

```shell
# Basic training
python main.py -core --epochs 100 --batch-size 16

# Multi-GPU training
python main.py -core --epochs 200 --device 0,1,2,3

# Resume training
python main.py -core --batch-size 16 --resume 

# High-resolution training
python main.py -core --batch-size 16 --accumulate 4 -dataset --img_size 1024

# Switch to yolov8 model, and use small-sized model
python main.py -link --model=yolov8-det.toml -model --scale s
```

For more detailed usage and advanced features, please refer to the dedicated documentation and example code for each model.

## ⚙️ Configuration System

`TinyTrain` uses unified TOML/YAML configuration management, supporting dynamic override at runtime.

### YAML Format Configuration Example

```yaml
name: MobileFaceNet
scale: n

# model_config compound scaling constants
scales:
  n: { depth: 0.5, summary: "161 layers, 888576 parameters, 888576 gradients, 0.37 GFLOPs, for a 112×112 input." }
  s: { depth: 1, summary: "246 layers, 2066752 parameters, 2066752 gradients, 0.99 GFLOPs, for a 112×112 input." }
  m: { depth: 2, summary: "429 layers, 4768320 parameters, 4768320 gradients, 2.47 GFLOPs, for a 112×112 input." }
  l: { depth: 3, summary: "611 layers, 8671296 parameters, 8671296 gradients, 4.59 GFLOPs, for a 112×112 input." }
  x: { depth: 4, summary: "793 layers, 13775680 parameters, 13775680 gradients, 7.36 GFLOPs, for a 112×112 input." }
  xl: { depth: 5, summary: "975 layers, 20081472 parameters, 20081472 gradients, 10.77 GFLOPs, for a 112×112 input." }
  xxl: { depth: 6, summary: "1157 layers, 27588672 parameters, 27588672 gradients, 14.83 GFLOPs, for a 112×112 input." }

# model network struct
network:
  - { type: entry, from: [ -1 ], module: CBA, repeat: 1,  args: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 2, padding: 1 } } # 0-P1/2
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 1, allow_repeat: True, args: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, groups: 64 } } # 1
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 1, args: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 2, padding: 1, groups: 128 } } # 2-P2/4
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 4, args: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, groups: 128, residual: True } } # 3
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 1, args: { in_channels: 64, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, groups: 256 } } # 4-P3/8
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 6, args: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, groups: 256, residual: True } } # 5
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 1, args: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 2, padding: 1, groups: 512 } } # 6-P4/16
  - { type: flow, from: [ -1 ], module: BottleneckDW, repeat: 2, args: { in_channels: 128, out_channels: 128, kernel_size: 3, stride: 1, padding: 1, groups: 256, residual: True } } # 7
  - { type: head, from: [ -1 ], module: GDCHead, repeat: 1, args: { in_channels: 128, embedding_size: 512 } } # 8
```

### TOML Format Configuration Example

```toml
name = "YOLOv8-cls"
scale = "n"

# model_config compound scaling constants
[scales.n]
depth = 0.33
width = 0.25
summary = "117 layers, 1492826 parameters, 1492826 gradients, 0.44 GFLOPs, for a 224×224 input."

[scales.s]
depth = 0.33
width = 0.50
summary = "117 layers, 5258922 parameters, 5258922 gradients, 1.64 GFLOPs, for a 224×224 input."

[scales.m]
depth = 0.67
width = 0.75
summary = "209 layers, 18099514 parameters, 18099514 gradients, 6.06 GFLOPs, for a 224×224 input."

[scales.l]
depth = 1.00
width = 1.00
summary = "299 layers, 43773770 parameters, 43773770 gradients, 15.04 GFLOPs, for a 224×224 input."

[scales.x]
depth = 1.00
width = 1.25
summary = "299 layers, 67961370 parameters, 67961370 gradients, 23.42 GFLOPs, for a 224×224 input."

# ----------------------------backbone---------------------------------
[[network]]
# 0-P1/2
type = "entry"
module = "CBA"
repeat = 1
from = [-1]
args.in_channels = 3
args.out_channels = 64
args.kernel_size = 3
args.stride = 2

[[network]]
# 1-P2/4
type = "flow"
module = "CBA"
repeat = 1
from = [-1]
args.in_channels = 64
args.out_channels = 128
args.kernel_size = 3
args.stride = 2

[[network]]
# 2
type = "flow"
module = "C2f"
repeat = 3
from = [-1]
args.in_channels = 128
args.out_channels = 128
args.shortcut = true

# ...
```

### 📊 Two Formats Comparison

| Feature    | 	YAML Format        | 	TOML Format          |
|--------|-----------------|-------------------|
| Readability	   | ✅ Excellent, clear hierarchical structure     | ✅ Good, intuitive key-value pairs        |
| Syntax Simplicity	| ✅ Very concise, supports stream syntax   | ⚠️ Relatively verbose, requires explicit declaration    |
| Comment Support	  | ✅ Supports # comments	       | ✅ Supports # comments     |
| Array Representation	  | ✅ Concise - list syntax	       | ⚠️ Requires [[ ]] double parentheses syntax |
| Nested Structure	  | ✅ Indentation-based, natural and intuitive	   | ✅ Section-based, clearly structured       |
| Type Safety	  | ⚠️ Dynamic typing, possible implicit conversion	| ✅ Strong typing, reduces errors        |
| Tool Ecosystem	  | ✅ Widely supported, mature and stable	    | ✅ Growing popularity, Python-friendly   |
| Learning Curve	  | 🎯 Relatively gentle, intuitive to learn	    | 🎯 Moderate, syntax is explicit        |

### 🎯 Recommendation

#### Recommended to use YAML when:

* Configuration structure is complex with deep nesting
* Requires high readability and conciseness
* Team is familiar with YAML syntax
* Configuration contains large amounts of list data

#### Recommended to use TOML when:

* Configuration is relatively flat, mainly key-value pairs
* Requires strong type checking and validation
* Deeply integrated with Python projects
* Configuration requires strict syntax standards

### 💡 Best Practices

* **New projects recommended YAML** - Better readability and development experience
* **Existing TOML projects keep** - Avoid unnecessary migration costs
* **Team unified standard** - Ensure consistency of configuration format within projects
* **Documented configs** - Provide detailed comments for complex configurations

> 🔄 **Format Compatibility** - TinyTrain fully supports both YAML and TOML configuration formats, allowing flexible choice based on project needs

## 🔮 Roadmap

* Test multi-machine multi-GPU distributed training
* Increase support for various AI tasks and algorithm models
* Develop visual annotation tools
* Optimize model compression and quantization features
* Increase different backend model deployment solutions

## 🤝 Contributing

We warmly welcome community contributions! Please refer to the [Contributing Guide](CONTRIBUTING.md).

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Issue Feedback

If you encounter any issues, please contact us through the following ways:

* 📧 Email: zycskylove520@gmail.com or 867245713@qq.com
* 💬 QQ Group: 271313723
* 🐛 Issues: GitHub Issues
* 📝 Discussion Area: GitHub Discussions

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Thank you to all developers who have contributed to this project!

***
<div align="center">If this project has been helpful to you, please give us a ⭐️ !</div>
