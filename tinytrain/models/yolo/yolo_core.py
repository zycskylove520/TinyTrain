from tinytrain.engine import Core, BaseExporter
from tinytrain.models.yolo.task.classify import YOLOClassificationModel, YOLOClassificationTrainer, YOLOClassificationValidator, YOLOClassificationPredictor, YOLOClassificationOnnxInferenceServer
from tinytrain.models.yolo.task.detect import YOLODetectionModel, YOLODetectionValidator,YOLODetectionPredictor,YOLODetectionTrainer
from tinytrain.server.track_server.bytetrack_server import ByteTrackServer
from tinytrain.server.export_server.onnx_export_server import BaseOnnxExportServer
from tinytrain.server.inference_server import BaseOnnxInferenceServer
from tinytrain.utils.register import TTRegistry


class YOLOCore(Core):
    # ---------- classify ----------
    TTRegistry.register("classify", "model")(YOLOClassificationModel)
    TTRegistry.register("classify", "trainer")(YOLOClassificationTrainer)
    TTRegistry.register("classify", "validator")(YOLOClassificationValidator)
    TTRegistry.register("classify", "predictor")(YOLOClassificationPredictor)
    TTRegistry.register("classify", "exporter")(BaseExporter)
    TTRegistry.register("classify", "export_server", "onnx")(BaseOnnxExportServer)
    TTRegistry.register("classify", "inference_server", "onnx")(YOLOClassificationOnnxInferenceServer)

    # ---------- detect ----------
    TTRegistry.register("detect", "model")(YOLODetectionModel)
    TTRegistry.register("detect", "trainer")(YOLODetectionTrainer)
    TTRegistry.register("detect", "validator")(YOLODetectionValidator)
    TTRegistry.register("detect", "predictor")(YOLODetectionPredictor)
    TTRegistry.register("detect", "inference_server", "onnx")(BaseOnnxInferenceServer)
    TTRegistry.register("detect", "exporter")(BaseExporter)
    TTRegistry.register("detect", "export_server", "onnx")(BaseOnnxExportServer)
    TTRegistry.register("detect", "track_server", "bytetrack")(ByteTrackServer)
