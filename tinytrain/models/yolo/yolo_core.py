"""
YOLOCore
========
YOLOCore 是 TinyTrain 针对 YOLO 系列任务（classify / detect）的专用门面类。
它在 Core 的基础上，通过 TTRegistry 完成 classify、detect 两大任务下
所有核心组件（model / trainer / validator / predictor / exporter / server）
的自动注册与快速索引，实现「一行代码」切换任务与后端。

主要特性
--------
1. 统一注册
   - 每个任务（classify / detect）在类定义阶段即通过装饰器完成注册，
     保证后续 Core 内部 `TTRegistry.get(task, component)` 能够
     零反射、零硬编码地实例化正确组件。
2. 多后端支持
   - 支持 onnx / tensorrt / torchscript 等多种导出与推理后端，
     通过 `"export_server"` / `"inference_server"` / `"track_server"` 等
     三级命名空间实现细粒度索引。
3. 链式配置
   - 继承 Core 的 ConfigManager，支持 link 配置文件链式继承，
     用户仅需改动少量字段即可切换模型规模、数据集路径、训练超参。

使用示例
--------
>>> from tinytrain import YOLOCore
>>> core = YOLOCore("cfg/yolo11-cls.yaml")
>>> core.train(model_scale="s")          # 分类任务，使用 YOLO-small 模型
>>> list(core.predict("assets/bus.jpg")) # 推理单张图片
>>> core.export("onnx")                  # 导出 onnx 并启动推理服务

注册表结构（摘要）
------------------
classify
  ├── model            -> YOLOClassificationModel
  ├── trainer          -> YOLOClassificationTrainer
  ├── validator        -> YOLOClassificationValidator
  ├── tuner            -> YOLOClassificationTuner
  ├── predictor        -> YOLOClassificationPredictor
  ├── exporter         -> BaseExporter
  ├── export_server
  │     └── onnx       -> BaseOnnxExportServer
  └── inference_server
        └── onnx       -> YOLOClassificationOnnxInferenceServer

detect
  ├── model            -> YOLODetectionModel
  ├── trainer          -> YOLODetectionTrainer
  ├── validator        -> YOLODetectionValidator
  ├── tuner            -> YOLODetectionTuner
  ├── predictor        -> YOLODetectionPredictor
  ├── exporter         -> BaseExporter
  ├── export_server
  │     └── onnx       -> BaseOnnxExportServer
  ├── inference_server
  │     └── onnx       -> BaseOnnxInferenceServer
  └── track_server
        └── bytetrack  -> ByteTrackServer
"""

from tinytrain.engine import Core, BaseExporter
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
    YOLODetectionTuner
)
from tinytrain.server.track_server.bytetrack_server import ByteTrackServer
from tinytrain.server.export_server.onnx_export_server import BaseOnnxExportServer
from tinytrain.server.inference_server import BaseOnnxInferenceServer
from tinytrain.utils.register import TTRegistry


class YOLOCore(Core):
    # ---------- classify ----------
    TTRegistry.register("classify", "model")(YOLOClassificationModel)
    TTRegistry.register("classify", "trainer")(YOLOClassificationTrainer)
    TTRegistry.register("classify", "validator")(YOLOClassificationValidator)
    TTRegistry.register("detect", "tuner")(YOLOClassificationTuner)
    TTRegistry.register("classify", "predictor")(YOLOClassificationPredictor)
    TTRegistry.register("classify", "exporter")(BaseExporter)
    TTRegistry.register("classify", "export_server", "onnx")(BaseOnnxExportServer)
    TTRegistry.register("classify", "inference_server", "onnx")(YOLOClassificationOnnxInferenceServer)

    # ---------- detect ----------
    TTRegistry.register("detect", "model")(YOLODetectionModel)
    TTRegistry.register("detect", "trainer")(YOLODetectionTrainer)
    TTRegistry.register("detect", "validator")(YOLODetectionValidator)
    TTRegistry.register("detect", "tuner")(YOLODetectionTuner)
    TTRegistry.register("detect", "predictor")(YOLODetectionPredictor)
    TTRegistry.register("detect", "inference_server", "onnx")(BaseOnnxInferenceServer)
    TTRegistry.register("detect", "exporter")(BaseExporter)
    TTRegistry.register("detect", "export_server", "onnx")(BaseOnnxExportServer)
    TTRegistry.register("detect", "track_server", "bytetrack")(ByteTrackServer)
