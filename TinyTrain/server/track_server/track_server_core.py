from __future__ import annotations

from TinyTrain.cfg.config_manager import ConfigManager
from TinyTrain.utils.callback import Callback
from TinyTrain.utils.register import TTRegistry


class TrackServerCore:
    """
    解析来自各个推理引擎的模型格式，不包含pt或pth
    """

    def __init__(self,
                 config_manager: ConfigManager,
                 callback: Callback,
                 backend: str,
                 **kwargs
                 ):
        self.config_manager = config_manager
        self.callback = callback
        self.backend = backend

        self.track_server = self._server_select(**kwargs)

    def get_server(self):
        return self.track_server

    def _server_select(self, **kwargs):
        """
        构造对应的跟踪服务
        @return:
        """
        task = self.config_manager.core["task"]
        return TTRegistry.get(task, "track_server", self.backend)(config_manager=self.config_manager, callback=self.callback, **kwargs)
