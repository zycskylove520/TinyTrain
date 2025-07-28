import numpy as np
import torch

from tinytrain.tracker.bytetracker import matching
from .base_tracker import TrackState, BaseTrack
from .kalman_filter import KalmanFilter


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, box, score):
        """
        @param box: 单个检测框，要求为lxlywh格式。
        @param score: 置信度分数。
        """

        # wait activate
        self._lxlywh = np.asarray(box, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """
        对多个目标进行卡尔曼滤波器预测。

        Args:
            stracks (list): 包含多个 STrack 对象的列表。
        """
        if len(stracks) > 0:
            # 提取每个 STrack 的卡尔曼滤波器均值和协方差矩阵
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 对于非跟踪状态的目标，将其速度置为 0
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0  # 假设速度信息在均值向量的第8个位置

            # 使用共享的卡尔曼滤波器进行批量预测
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            # 更新每个 STrack 的卡尔曼滤波器状态
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.lxlywh_2_cxcyah(self._lxlywh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.lxlywh_2_cxcyah(new_track.lxlywh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_lxlywh = new_track.lxlywh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.lxlywh_2_cxcyah(new_lxlywh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def lxlywh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._lxlywh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def lxlyrxry(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.lxlywh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def lxlywh_2_cxcyah(lxlywh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(lxlywh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_lxlyah(self):
        return self.lxlywh_2_cxcyah(self.lxlywh)

    @staticmethod
    def lxlyrxry_2_lxlywh(lxlyrxry):
        ret = np.asarray(lxlyrxry).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def lxlywh_2_lxlyrxry(lxlywh):
        ret = np.asarray(lxlywh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, config_manager):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.config_manager = config_manager
        self.track_threshold = config_manager.tracker["track_threshold"]
        self.det_threshold = self.track_threshold + 0.1
        self.max_time_lost = config_manager.tracker["track_buffer"]
        self.kalman_filter = KalmanFilter()

    def update(self, bboxes: np.ndarray, scores:np.ndarray) -> list[STrack]:
        """
        update只允许进行单张图片结果进行更新

        @param bboxes: 单张图片检测的bboxes结果，其shape为：[num_anchors, lxlyrxry]，要求box已反归一化
        @param scores: 单张图片检测的置信度结果，其shape为：[num_anchors, conf]
        """
        self.frame_id += 1
        activated_starcks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []

        # 置信度高于跟踪阈值的检测框。
        remain_inds = scores > self.track_threshold
        # 高置信度检测框与分数
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        # 置信度在较低阈值（0.1）和跟踪阈值之间的检测框。
        inds_low = scores > 0.1
        inds_high = scores < self.track_threshold
        inds_second = np.logical_and(inds_low, inds_high)
        # 低置信度检测框与分数
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.lxlyrxry_2_lxlywh(box), score) for
                          (box, score) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed_stracks: list[STrack] = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        # 合并正在跟踪的目标和丢失目标
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # 将新的目标和跟踪池中已有的目标计算距离
        dists = matching.iou_distance(strack_pool, detections)
        # matches: 匹配的轨迹和检测框的索引对。
        # 这是一个二维数组，每一行表示一个匹配对，其中第一列是轨迹的索引，第二列是检测框的索引。
        # 例如，[[0, 1], [2, 3]] 表示轨迹索引为 0 的轨迹与检测框索引为 1 的检测框匹配，轨迹索引为 2 的轨迹与检测框索引为 3 的检测框匹配。
        # u_track: 未匹配的轨迹索引。
        # 这是一个一维数组，包含那些没有与任何检测框匹配的轨迹的索引。
        # 例如，[1, 4] 表示轨迹索引为 1 和 4 的轨迹没有找到匹配的检测框。
        # u_detection: 未匹配的检测框索引。
        # 这是一个一维数组，包含那些没有与任何轨迹匹配的检测框的索引。
        # 例如，[0, 2] 表示检测框索引为 0 和 2 的检测框没有找到匹配的轨迹。
        matches, unmatch_track, unmatch_detection = matching.linear_assignment(dists, thresh=self.config_manager.tracker["match_threshold"])

        for itracked, idet in matches:
            track = strack_pool[itracked]  # 拿出匹配的轨迹
            det = detections[idet]  # 拿出匹配的检测框
            if track.state == TrackState.Tracked:
                # 当前轨迹处于跟踪状态，则将当前轨迹加入到已激活轨迹列表中
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # 当前轨迹未处于跟踪状态，则重新激活该轨迹，并加入到重激活轨迹列表中
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.lxlyrxry_2_lxlywh(box), score) for
                                 (box, score) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # 在高分框中未匹配的轨迹如果属于激活状态，则拿出来和低分框进行匹配
        r_tracked_stracks = [strack_pool[i] for i in unmatch_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, unmatch_track, unmatch_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track: STrack = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 经过低分框匹配后依旧未能匹配的轨迹，标记为丢失
        for i in unmatch_track:
            track = r_tracked_stracks[i]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 拿高分框中未能和激活轨迹列表中的轨迹相匹配的框和不确定的轨迹进行匹配
        detections = [detections[i] for i in unmatch_detection]
        dists = matching.iou_distance(unconfirmed_stracks, detections)
        matches, unmatch_unconfirmed, unmatch_detection = matching.linear_assignment(dists, thresh=0.7)
        # 匹配上的轨迹重新加入到激活轨迹列表中
        for itracked, idet in matches:
            unconfirmed_stracks[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed_stracks[itracked])
        # 不确定的轨迹列表中依旧未匹配上的轨迹，则标记未移除，加入移除轨迹列表
        for it in unmatch_unconfirmed:
            track = unconfirmed_stracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 和不确定轨迹匹配完后依旧没有匹配上的高分框，则当作新的目标，创建一条新的轨迹记录
        for inew in unmatch_detection:
            track = detections[inew]
            if track.score < self.det_threshold:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # 丢失轨迹列表中的轨迹如果多帧依旧存在，则标记为移除并加入到移除轨迹中
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        # 从lost_stracks中剔除那些处于tracked_stracks中的轨迹
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        # 从lost_stracks中剔除那些处于removed_stracks中的轨迹
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
        return resa, resb
