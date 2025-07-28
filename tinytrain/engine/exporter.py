from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Union

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from torch import nn
    from tinytrain.server.export_server import ExportServerCore


class BaseExporter:
    def __init__(self,
                 config_manager: ConfigManager,
                 model: nn.Module,
                 callback: Callback,
                 backend: str | None = None,
                 **kwargs
                 ):
        self.config_manager = config_manager
        self.backend = backend

        # device
        from tinytrain.utils.checks import check_device_mini
        self.device = check_device_mini(self.config_manager.core["device"])

        # model
        self.export_server = self._setup_export_server(model=model, **kwargs)

        # callback
        self.callback = callback

    def export(self, export_dir: str | Path):
        import torch

        self.callback.run_callback(self, "on_export_start")
        with torch.inference_mode():
            self.export_server.export(export_dir)
        self.callback.run_callback(self, "on_export_end")

    def _setup_export_server(self, model: nn.Module, **kwargs) -> Union[nn.Module, ExportServerCore]:
        from tinytrain.server.export_server import ExportServerCore
        from torch import nn

        if isinstance(model, nn.Module):
            model = model.to(self.device)
            model.eval()
            return ExportServerCore(config_manager=self.config_manager, model=model, backend=self.backend, device=self.device, **kwargs)
        else:
            raise TypeError(f"only supported pytorch model!")
