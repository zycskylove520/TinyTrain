from tinytrain.cfg.TT_register import TTEngineRegistry
from tinytrain.engine import Core, BaseExporter
from tinytrain.models.face.task.recognition.export_server import FaceRecognitionOnnxExportServer
from tinytrain.models.face.task.recognition.model import FaceRecognitionModel
from tinytrain.models.face.task.recognition.predictor import FaceRecognitionPredictor
from tinytrain.models.face.task.recognition.trainer import FaceRecognitionTrainer
from tinytrain.models.face.task.recognition.validator import FaceRecognitionValidator
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
        TTEngineRegistry.register(cls, "recognition", "export_server", "onnx")(FaceRecognitionOnnxExportServer)