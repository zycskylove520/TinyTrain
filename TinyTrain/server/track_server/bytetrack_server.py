from __future__ import annotations

import cv2
import numpy as np

from typing import TYPE_CHECKING

from TinyTrain.cfg.config_manager import ConfigManager
from TinyTrain.data import DetectDataInfo
from TinyTrain.server.track_server.base_track_server import BaseTrackServer
from TinyTrain.tools import images_to_video
from TinyTrain.tracker.bytetracker.byte_tracker import STrack, BYTETracker
from TinyTrain.utils import LOGGER
from TinyTrain.utils.callback import Callback

if TYPE_CHECKING:
    from TinyTrain.engine import BasePredictor


class ByteTrackServer(BaseTrackServer):
    def __init__(self, config_manager: ConfigManager, callback: Callback, **kwargs):
        super().__init__(config_manager, callback, **kwargs)
        self.tracker = BYTETracker(config_manager)
        self.stop_show_window = False

    def register_callback(self, callback: Callback):
        super().register_callback(callback)
        callback.add_callback("on_predict_start", self.on_predict_start)
        callback.add_callback("on_predict_batch_end", self.on_predict_batch_end)
        callback.add_callback("on_predict_end", self.on_predict_end)

    def on_predict_start(self, predictor: BasePredictor):
        LOGGER.info("start tracking...")

    def on_predict_batch_end(self, predictor: BasePredictor):
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
        # 读取跟踪结果保存路径下所有的图片合并为视频
        self.save_video()

        LOGGER.info("end tracking...")

    def save_track_imgs(self, detect_info: DetectDataInfo, track_results):
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
            cv2.imwrite(out_path, img)

        # ----------- 新增：边推理边显示 -----------
        if self.config_manager.tracker["show_realtime_result"] and not self.stop_show_window:
            cv2.imshow("realtime track result", detect_info.img)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # ESC 键退出
                self.stop_show_window = True

    def save_video(self):
        if not (self.config_manager.tracker["save_img"] and self.config_manager.tracker["save_video"]):
            return

        video_out = self.output_dir / "video/result.mp4"
        video_out.parent.mkdir(parents=True, exist_ok=True)
        fps = self.config_manager.tracker["save_video_fps"]
        images_to_video(self.output_dir, video_out, fps)
