from __future__ import annotations

import cv2

from typing import TYPE_CHECKING

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data import DetectDataInfo
from tinytrain.server.track_server.base_track_server import BaseTrackServer
from tinytrain.tools import images_to_video
from tinytrain.tracker.bytetracker.byte_tracker import STrack, BYTETracker
from tinytrain.utils import LOGGER
from tinytrain.utils.callback import Callback

if TYPE_CHECKING:
    from tinytrain.engine import BasePredictor


class ByteTrackServer(BaseTrackServer):
    """
    ByteTrack 跟踪服务器实现，继承自 BaseTrackServer。
    在预测器运行期间实时更新轨迹、绘制结果并支持保存图片 / 视频 / 实时显示。
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
        self.tracker = BYTETracker(config_manager)
        self.stop_show_window = False

    def register_callback(self, callback: Callback):
        """
        注册 ByteTrack 所需的全部回调钩子。

        Args
        ----
        callback : Callback
            回调注册器实例。
        """
        super().register_callback(callback)
        callback.add_callback("on_predict_start", self.on_predict_start)
        callback.add_callback("on_predict_batch_end", self.on_predict_batch_end)
        callback.add_callback("on_predict_end", self.on_predict_end)

    def on_predict_start(self, predictor: BasePredictor):
        """预测开始时记录日志。"""
        LOGGER.info("start tracking...")

    def on_predict_batch_end(self, predictor: BasePredictor):
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

        # 保存图片
        self.save_track_imgs(detect_info, track_results)

        self.track_results = track_results

    def on_predict_end(self, predictor: BasePredictor):
        """
        预测结束后整理跟踪结果并生成视频。

        Args
        ----
        predictor : BasePredictor
            预测器实例，用于获取输出目录等信息。
        """
        self.save_video()

        LOGGER.info("end tracking...")

    def save_track_imgs(self, detect_info: DetectDataInfo, track_results):
        """
        在图像上绘制跟踪结果并保存 / 显示。

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

        if self.config_manager.tracker["save_img"]:
            out_path = self.output_dir / f"img/frame_{detect_info.frame_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, img)

        # ----------- 新增：边推理边显示 -----------
        if self.config_manager.tracker["show_realtime_result"] and not self.stop_show_window:
            cv2.imshow("realtime track result", detect_info.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC 键退出
                self.stop_show_window = True

    def save_video(self):
        """
        将保存的图片序列合成为视频。

        仅在 save_img 与 save_video 同时开启时执行。
        """
        if not (self.config_manager.tracker["save_img"] and self.config_manager.tracker["save_video"]):
            return

        video_out = self.output_dir / "video/result.mp4"
        video_out.parent.mkdir(parents=True, exist_ok=True)
        fps = self.config_manager.tracker["save_video_fps"]
        images_to_video(self.output_dir/"img", video_out, fps)
