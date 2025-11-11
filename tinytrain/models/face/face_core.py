"""
FaceCore
========
FaceCore 是 TinyTrain 针对人脸相关任务的专用门面类。
它在 TTBaseCore 的基础上，通过 TTEngineRegistry 完成 recognition 等任务下
所有核心组件（model / trainer / validator / predictor / exporter / server）
的自动注册与快速索引，实现「一行代码」切换后端与导出格式。

主要特性
--------
1. 统一注册
   - 每个组件在类定义阶段即通过装饰器完成注册，
     保证后续 TTBaseCore 内部 `TTEngineRegistry.get(task, component)` 能够
     零反射、零硬编码地实例化正确组件。
2. 多后端支持
   - 支持 onnx 导出与推理后端，
     通过 `"export_server"` / `"inference_server"` 两级命名空间实现细粒度索引。
3. 链式配置
   - 继承 TTBaseCore 的 TTConfigManager，支持 link 配置文件链式继承，
     用户仅需改动少量字段即可切换模型规模、数据集路径、训练超参。

注册表结构（摘要）
------------------
recognition
  ├── model            -> FaceRecognitionModel
  ├── trainer          -> FaceRecognitionTrainer
  ├── validator        -> FaceRecognitionValidator
  ├── predictor        -> FaceRecognitionPredictor
  ├── exporter         -> TTBaseExporter
  ├── export_server
  │     └── onnx       -> TTBaseOnnxExportServer
  └── inference_server
        └── onnx       -> TTBaseOnnxInferenceServer
"""

from tinytrain.cfg import TTEngineRegistry
from tinytrain.engine import TTBaseCore, TTBaseExporter
from tinytrain.models.face.task.recognition import (
    FaceRecognitionModel,
    FaceRecognitionTrainer,
    FaceRecognitionValidator,
    FaceRecognitionPredictor,
)
from tinytrain.server.export_server import TTBaseOnnxExportServer
from tinytrain.server.inference_server import TTBaseOnnxInferenceServer


class FaceCore(TTBaseCore):
    @classmethod
    def register_components(cls):
        # ---------- recognition ----------
        TTEngineRegistry.register(cls, "recognition", "model")(FaceRecognitionModel)
        TTEngineRegistry.register(cls, "recognition", "trainer")(FaceRecognitionTrainer)
        TTEngineRegistry.register(cls, "recognition", "validator")(FaceRecognitionValidator)
        TTEngineRegistry.register(cls, "recognition", "predictor")(FaceRecognitionPredictor)
        TTEngineRegistry.register(cls, "recognition", "inference_server", "onnx")(TTBaseOnnxInferenceServer)
        TTEngineRegistry.register(cls, "recognition", "exporter")(TTBaseExporter)
        TTEngineRegistry.register(cls, "recognition", "export_server", "onnx")(TTBaseOnnxExportServer)
