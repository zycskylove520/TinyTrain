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
YOLOCore
========
YOLOCore 是 TinyTrain 针对 YOLO 系列任务（classify / detect / pose / segment）的专用门面类。
它在 TTBaseCore 的基础上，通过 TTEngineRegistry 完成 classify、detect、pose、segment 四大任务下
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
classify
  ├── model            -> YOLOClassificationModel
  ├── trainer          -> YOLOClassificationTrainer
  ├── validator        -> YOLOClassificationValidator
  ├── tuner            -> YOLOClassificationTuner
  ├── predictor        -> YOLOClassificationPredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  └── inference_server
        └── onnx       -> YOLOClassificationOnnxInferenceServer

detect
  ├── model            -> YOLODetectionModel
  ├── trainer          -> YOLODetectionTrainer
  ├── validator        -> YOLODetectionValidator
  ├── tuner            -> YOLODetectionTuner
  ├── predictor        -> YOLODetectionPredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  ├── inference_server
  │     └── onnx       -> TTBaseOnnxInferenceServer
  └── track_server
        └── bytetrack  -> ByteTrackServer

pose
  ├── model            -> YOLOPoseModel
  ├── trainer          -> YOLOPoseTrainer
  ├── validator        -> YOLOPoseValidator
  ├── tuner            -> YOLOPoseTuner
  ├── predictor        -> YOLOPosePredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  ├── inference_server
  │     └── onnx       -> TTBaseOnnxInferenceServer
  └── track_server
        └── bytetrack  -> ByteTrackServer

segment
  ├── model            -> YOLOSegmentModel
  ├── trainer          -> YOLOSegmentTrainer
  ├── validator        -> YOLOSegmentValidator
  ├── tuner            -> YOLOSegmentTuner
  ├── predictor        -> YOLOSegmentPredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  ├── inference_server
  │     └── onnx       -> TTBaseOnnxInferenceServer
  └── track_server
        └── bytetrack  -> ByteTrackServer
"""

from tinytrain.engine import TTBaseCore, TTBaseExporter
from tinytrain.cfg import TTEngineRegistry
from tinytrain.models.yolo.task.classify import (
    YOLOClassificationModel,
    YOLOClassificationTrainer,
    YOLOClassificationValidator,
    YOLOClassificationPredictor,
    YOLOClassificationOnnxInferenceServer,
    YOLOClassificationTuner
)
from tinytrain.models.yolo.task.detect import (
    YOLODetectionModel,
    YOLODetectionValidator,
    YOLODetectionPredictor,
    YOLODetectionTrainer,
    YOLODetectionTuner,
    YOLODetectionDistiller
)
from tinytrain.models.yolo.task.pose import (
    YOLOPoseModel,
    YOLOPoseTrainer,
    YOLOPoseValidator,
    YOLOPosePredictor,
    YOLOPoseTuner
)
from tinytrain.models.yolo.task.segment import (
    YOLOSegmentModel,
    YOLOSegmentTrainer,
    YOLOSegmentValidator,
    YOLOSegmentTuner,
    YOLOSegmentPredictor,
    YOLOSegmentOnnxInferenceServer
)
from tinytrain.server.export_server import TTBaseOnnxExportServer
from tinytrain.server.inference_server import TTBaseOnnxInferenceServer
from tinytrain.server.track_server import ByteTrackServer


class YOLOCore(TTBaseCore):
    @classmethod
    def register_components(cls):
        # ---------- classify ----------
        TTEngineRegistry.register(cls, "classify", "model")(YOLOClassificationModel)
        TTEngineRegistry.register(cls, "classify", "trainer")(YOLOClassificationTrainer)
        TTEngineRegistry.register(cls, "classify", "validator")(YOLOClassificationValidator)
        TTEngineRegistry.register(cls, "classify", "tuner")(YOLOClassificationTuner)
        TTEngineRegistry.register(cls, "classify", "predictor")(YOLOClassificationPredictor)
        TTEngineRegistry.register(cls, "classify", "inference_server", "onnx")(YOLOClassificationOnnxInferenceServer)
        TTEngineRegistry.register(cls, "classify", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "classify", "export_server", "onnx")(TTBaseOnnxExportServer)

        # ---------- detect ----------
        TTEngineRegistry.register(cls, "detect", "model")(YOLODetectionModel)
        TTEngineRegistry.register(cls, "detect", "trainer")(YOLODetectionTrainer)
        TTEngineRegistry.register(cls, "detect", "validator")(YOLODetectionValidator)
        TTEngineRegistry.register(cls, "detect", "tuner")(YOLODetectionTuner)
        TTEngineRegistry.register(cls, "detect", "predictor")(YOLODetectionPredictor)
        TTEngineRegistry.register(cls, "detect", "inference_server", "onnx")(TTBaseOnnxInferenceServer)
        TTEngineRegistry.register(cls, "detect", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "detect", "export_server", "onnx")(TTBaseOnnxExportServer)
        TTEngineRegistry.register(cls, "detect", "track_server", "bytetrack")(ByteTrackServer)
        TTEngineRegistry.register(cls, "detect", "distiller")(YOLODetectionDistiller)

        # ---------- pose ----------
        TTEngineRegistry.register(cls, "pose", "model")(YOLOPoseModel)
        TTEngineRegistry.register(cls, "pose", "trainer")(YOLOPoseTrainer)
        TTEngineRegistry.register(cls, "pose", "validator")(YOLOPoseValidator)
        TTEngineRegistry.register(cls, "pose", "tuner")(YOLOPoseTuner)
        TTEngineRegistry.register(cls, "pose", "predictor")(YOLOPosePredictor)
        TTEngineRegistry.register(cls, "pose", "inference_server", "onnx")(TTBaseOnnxInferenceServer)
        TTEngineRegistry.register(cls, "pose", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "pose", "export_server", "onnx")(TTBaseOnnxExportServer)
        TTEngineRegistry.register(cls, "pose", "track_server", "bytetrack")(ByteTrackServer)

        # ---------- segment ----------
        TTEngineRegistry.register(cls, "segment", "model")(YOLOSegmentModel)
        TTEngineRegistry.register(cls, "segment", "trainer")(YOLOSegmentTrainer)
        TTEngineRegistry.register(cls, "segment", "validator")(YOLOSegmentValidator)
        TTEngineRegistry.register(cls, "segment", "tuner")(YOLOSegmentTuner)
        TTEngineRegistry.register(cls, "segment", "predictor")(YOLOSegmentPredictor)
        TTEngineRegistry.register(cls, "segment", "inference_server", "onnx")(YOLOSegmentOnnxInferenceServer)
        TTEngineRegistry.register(cls, "segment", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "segment", "export_server", "onnx")(TTBaseOnnxExportServer)
        TTEngineRegistry.register(cls, "segment", "track_server", "bytetrack")(ByteTrackServer)
