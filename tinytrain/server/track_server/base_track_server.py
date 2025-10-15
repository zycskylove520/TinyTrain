from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import TTConfigManager
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from tinytrain.engine import TTBasePredictor


class TTBaseTrackServer:
    """
    目标跟踪服务器基类。

    职责
    ----
    1. 接收 predictor 的检测结果，运行跟踪算法（SORT / ByteTrack / DeepSORT 等）。
    2. 通过回调系统，在 predictor 启动时自动创建保存目录。
    3. 提供统一的接口 `update`，将检测框更新为跟踪轨迹。
    """

    def __init__(self, config_manager: TTConfigManager, callback: Callback, **kwargs):
        """
        初始化跟踪服务器。

        Args
        ----
        config_manager : TTConfigManager
            全局配置，用于读取跟踪器超参。
        callback : Callback
            回调注册器，用于在 predictor 生命周期钩子中挂载自定义逻辑。
        **kwargs
            透传给具体跟踪器的额外参数，如 track_thresh、match_thresh 等。
        """
        self.config_manager = config_manager
        self.tracker = None
        self.output_dir = None
        self.register_callback(callback)

        # results
        self.track_results: np.ndarray | None = None

    def register_callback(self, callback: Callback):
        """
        向回调注册器挂载钩子函数。

        目前仅注册 `on_predict_start` 钩子，用于创建保存目录。

        Args
        ----
        callback : Callback
            回调注册器实例。
        """
        callback.add_callback("on_predict_start", self.create_save_dir)

    def create_save_dir(self, predictor: TTBasePredictor):
        """
        predictor 启动时被回调，创建跟踪结果保存目录。

        目录结构
        --------
        <predictor.output_dir>/track/
            ├── video/
            ├── img/
            └── ...

        Args
        ----
        predictor : TTBasePredictor
            正在运行的预测器实例，提供 output_dir 等信息。
        """
        # save dir
        self.output_dir = predictor.output_dir / "track"
        self.output_dir.mkdir(parents=True, exist_ok=True)
