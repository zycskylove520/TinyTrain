from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING

from TinyTrain.cfg.config_manager import ConfigManager
from TinyTrain.utils.any_utils import create_iter_directory
from TinyTrain.utils.callback import Callback

if TYPE_CHECKING:
    from TinyTrain.engine import BasePredictor


class BaseTrackServer:
    def __init__(self, config_manager: ConfigManager, callback: Callback, **kwargs):
        self.config_manager = config_manager
        self.tracker = None
        self.output_dir = None
        self.register_callback(callback)

        # results
        self.track_results: np.ndarray | None = None

    def register_callback(self, callback: Callback):
        callback.add_callback("on_predict_start", self.create_save_dir)

    def create_save_dir(self, predictor: BasePredictor):
        # save dir
        self.output_dir = predictor.output_dir / "track"
        self.output_dir.mkdir(parents=True, exist_ok=True)