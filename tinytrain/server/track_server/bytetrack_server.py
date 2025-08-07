from __future__ import annotations

import cv2

from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data import DetectDataInfo
from tinytrain.server.track_server.base_track_server import BaseTrackServer
from tinytrain.tracker.bytetracker.byte_tracker import STrack, BYTETracker
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from tinytrain.engine import BasePredictor


class ByteTrackServer(BaseTrackServer):
    """
    ByteTrack 跟踪服务器实现，继承自 BaseTrackServer。
    在预测器运行期间，实时更新轨迹、绘制结果并支持保存图片 / 视频 / 实时显示。
    """

    def __init__(self, config_manager: ConfigManager, callback: Callback, **kwargs):
        """
        初始化 ByteTrack 跟踪服务器。

        Args
        ----
        config_manager : ConfigManager
            全局配置管理器，用于读取 tracker 超参（min_box_area、save_img 等）。
        callback : Callback
            回调注册器，用于挂载 on_predict_start / on_predict_batch_end / on_predict_end 钩子。
        **kwargs
            透传给 BYTETracker 的初始化参数。
        """
        super().__init__(config_manager, callback, **kwargs)
        self.img_shape = None
        self.tracker = BYTETracker(config_manager)
        self.stop_show_window = False

        # 初始化 VideoWriter
        self.video_writer = None
        self.video_fps = config_manager.tracker["save_video_fps"]
        self.video_initialized = False  # 标记 VideoWriter 是否已初始化

    def register_callback(self, callback: Callback):
        """
        注册 ByteTrack 所需的全部回调钩子。

        Args
        ----
        callback : Callback
            回调注册器实例。
        """
        super().register_callback(callback)
        callback.add_callback("on_predict_start", self.logger_record)
        callback.add_callback("on_predict_batch_end", self.update_track_results)
        callback.add_callback("on_predict_end", self.on_predict_end)

    def logger_record(self, predictor: BasePredictor):
        """预测开始时记录日志。"""
        LOGGER.info("start tracking...")

    def update_track_results(self, predictor: BasePredictor):
        """
        每帧预测结束后执行跟踪更新、结果绘制与保存。

        Args
        ----
        predictor : BasePredictor
            当前正在运行的预测器实例，包含最新检测结果。
        """
        detect_info: DetectDataInfo = predictor.postprocess_result
        bboxes = detect_info.bboxes
        scores = detect_info.scores

        track_results = {"track_ids": [], "bboxes": [], "scores": []}
        if bboxes.shape[0] == 0:
            self.track_results = track_results

        results: list[STrack] = self.tracker.update(bboxes, scores)

        for result in results:
            box = result.lxlywh
            vertical = box[2] / box[3] > 1.6
            if box[2] * box[3] > self.config_manager.tracker["min_box_area"] and not vertical:
                track_results["track_ids"].append(result.track_id)
                track_results["bboxes"].append(result.lxlyrxry)
                track_results["scores"].append(result.score)

        # 绘制跟踪结果
        self.draw_track_results(detect_info, track_results)

        # 实时显示
        self.runtime_show(detect_info)

        # 保存图片
        self.save_track_imgs(detect_info)

        # 写入视频
        self.save_track_video(detect_info)

        self.track_results = track_results

    def on_predict_end(self, predictor: BasePredictor):
        """
        预测结束后整理跟踪结果并生成视频。

        Args
        ----
        predictor : BasePredictor
            预测器实例，用于获取输出目录等信息。
        """
        if self.video_writer is not None:
            self.video_writer.release()
            LOGGER.info(f"wideo writer release...")
        LOGGER.info("end tracking...")

    def draw_track_results(self, detect_info: DetectDataInfo, track_results):
        """
        在图像上绘制跟踪结果。

        Args
        ----
        detect_info : DetectDataInfo
            当前帧的图像与元信息。
        track_results : dict
            当前帧的跟踪结果，键包含 track_ids、bboxes、scores。
        """
        img = detect_info.img

        # 图像左上角写 frame_id
        if detect_info.frame_id is not None:
            cv2.putText(img,
                        f'frame:{detect_info.frame_id}',
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 255),  # 黄色醒目
                        thickness=2,
                        lineType=cv2.LINE_AA)

        for i in range(len(track_results["track_ids"])):
            track_id = int(track_results["track_ids"][i])  # 取 track_id
            l, t, r, b = map(int, track_results["bboxes"][i])  # lx, ly, rx, ry

            # 画框
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)

            # 在框内部左上角写 track_id
            id_txt = str(track_id)
            cv2.putText(img, id_txt, (l + 2, t + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    def runtime_show(self, detect_info: DetectDataInfo):
        """
        实时显示跟踪图像。

        Args
        ----
        detect_info : DetectDataInfo
            当前帧的图像与元信息。
        """
        img = detect_info.img
        if self.config_manager.tracker["show_realtime_result"] and not self.stop_show_window:
            cv2.imshow("realtime track result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC 键退出
                self.stop_show_window = True

    def save_track_imgs(self, detect_info: DetectDataInfo):
        """
        在图像上保存跟踪结果。

        Args
        ----
        detect_info : DetectDataInfo
            当前帧的图像与元信息。
        """
        img = detect_info.img
        if self.config_manager.tracker["save_img"]:
            out_path = self.output_dir / f"img/frame_{detect_info.frame_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, img)

    def _initialize_video_writer(self, w, h):
        """
        初始化 VideoWriter。
        """
        video_out_path = self.output_dir / "video/result.mp4"
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(video_out_path), fourcc, self.video_fps, (w, h))

    def save_track_video(self, detect_info: DetectDataInfo):
        """
        实时保存跟踪结果到视频。

        Args
        ----
        detect_info : DetectDataInfo
            当前帧的图像与元信息。
        """
        if not self.video_initialized and self.config_manager.tracker["save_video"]:
            w, h = detect_info.origin_shape
            self._initialize_video_writer(w, h)
            self.video_initialized = True

        if self.video_writer is not None:
            self.video_writer.write(detect_info.img)