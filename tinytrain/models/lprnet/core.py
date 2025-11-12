"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
LPRNetCore
========
LPRNetCore 是 TinyTrain 针对 LPRNet 模型的专用门面类。
它在 TTBaseCore 的基础上，通过 TTEngineRegistry 完成相关的一系列任务
所有核心组件（model / trainer / validator / predictor / exporter / server）
的自动注册与快速索引，实现「一行代码」切换任务与后端。

主要特性
--------
1. 统一注册
   - 每个任务在类定义阶段即通过装饰器完成注册，
     保证后续 TTBaseCore 内部 `TTEngineRegistry.get(task, component)` 能够
     零反射、零硬编码地实例化正确组件。
2. 多后端支持
   - 支持 onnx / tensorrt / torchscript 等多种导出与推理后端，
     通过 `"export_server"` / `"inference_server"` / `"track_server"` 等
     三级命名空间实现细粒度索引。
3. 链式配置
   - 继承 TTBaseCore 的 TTConfigManager，支持 link 配置文件链式继承，
     用户仅需改动少量字段即可切换模型规模、数据集路径、训练超参。


注册表结构（摘要）
------------------
lpr
  ├── model            -> LPRModel
  ├── trainer          -> LPRTrainer
  ├── validator        -> YOLOClassificationValidator
  ├── tuner            -> YOLOClassificationTuner
  ├── predictor        -> YOLOClassificationPredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  └── inference_server
        └── onnx       -> YOLOClassificationOnnxInferenceServer
"""

from tinytrain.engine import TTBaseCore, TTBaseExporter
from tinytrain.cfg import TTEngineRegistry
from tinytrain.models.lprnet.engine import (
    LPRModel,
    LPRTrainer,
    LPRValidator,
    LPRPredictor,
)
from tinytrain.server.export_server import TTBaseOnnxExportServer
from tinytrain.server.inference_server import TTBaseOnnxInferenceServer


class LPRNetCore(TTBaseCore):
    @classmethod
    def register_components(cls):
        # ---------- classify ----------
        TTEngineRegistry.register(cls, "lpr", "model")(LPRModel)
        TTEngineRegistry.register(cls, "lpr", "trainer")(LPRTrainer)
        TTEngineRegistry.register(cls, "lpr", "validator")(LPRValidator)
        TTEngineRegistry.register(cls, "lpr", "predictor")(LPRPredictor)
        TTEngineRegistry.register(cls, "lpr", "inference_server", "onnx")(TTBaseOnnxInferenceServer)
        TTEngineRegistry.register(cls, "lpr", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "lpr", "export_server", "onnx")(TTBaseOnnxExportServer)