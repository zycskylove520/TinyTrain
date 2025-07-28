from TinyTrain.engine import Core, BaseExporter
from TinyTrain.models.yolo.task.classify import YOLOClassificationModel, YOLOClassificationTrainer, YOLOClassificationValidator, YOLOClassificationPredictor, YOLOClassificationOnnxInferenceServer
from TinyTrain.models.yolo.task.detect import YOLODetectionModel, YOLODetectionValidator
from TinyTrain.models.yolo.task.detect import YOLODetectionPredictor
from TinyTrain.models.yolo.task.detect import YOLODetectionTrainer
from TinyTrain.server.track_server.bytetrack_server import ByteTrackServer
from TinyTrain.server.export_server.onnx_export_server import BaseOnnxExportServer
from TinyTrain.server.inference_server import BaseOnnxInferenceServer
from TinyTrain.utils.register import TTRegistry


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

    # predict
    TTRegistry.register("detect", "predictor")(YOLODetectionPredictor)
    TTRegistry.register("detect", "inference_server", "onnx")(BaseOnnxInferenceServer)

    # export
    TTRegistry.register("detect", "exporter")(BaseExporter)
    TTRegistry.register("detect", "export_server", "onnx")(BaseOnnxExportServer)

    # track
    TTRegistry.register("detect", "track_server", "bytetrack")(ByteTrackServer)
