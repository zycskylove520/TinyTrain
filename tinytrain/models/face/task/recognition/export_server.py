import torch
from torch import nn

from tinytrain.engine import BaseModel
from tinytrain.server.export_server import BaseOnnxExportServer


class FaceRecognitionOnnxExportServer(BaseOnnxExportServer):
    def __init__(self, model: nn.Module,
                 device: torch.device,
                 **kwargs):
        super(FaceRecognitionOnnxExportServer, self).__init__(model, device, **kwargs)

        # 设置模型head层为export模式
        if isinstance(self.model, BaseModel):
            self.model.module_list[-1].export = True