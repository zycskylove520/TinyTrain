from tinytrain.cfg import TTEngineRegistry
from tinytrain.engine import Core, BaseExporter
from tinytrain.models.face.task.recognition import (
    FaceRecognitionModel,
    FaceRecognitionTrainer,
    FaceRecognitionValidator,
    FaceRecognitionPredictor,
)
from tinytrain.server.export_server import BaseOnnxExportServer
from tinytrain.server.inference_server import BaseOnnxInferenceServer


class FaceCore(Core):
    @classmethod
    def register_components(cls):
        # ---------- recognition ----------
        TTEngineRegistry.register(cls, "recognition", "model")(FaceRecognitionModel)
        TTEngineRegistry.register(cls, "recognition", "trainer")(FaceRecognitionTrainer)
        TTEngineRegistry.register(cls, "recognition", "validator")(FaceRecognitionValidator)
        TTEngineRegistry.register(cls, "recognition", "predictor")(FaceRecognitionPredictor)
        TTEngineRegistry.register(cls, "recognition", "inference_server", "onnx")(BaseOnnxInferenceServer)
        TTEngineRegistry.register(cls, "recognition", "exporter")(BaseExporter)
        TTEngineRegistry.register(cls, "recognition", "export_server", "onnx")(BaseOnnxExportServer)
