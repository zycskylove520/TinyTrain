"""
BaseTrack: 多目标跟踪（MOT）基类
为所有跟踪器提供统一的 ID 管理、状态机与特征缓存接口，支持跨相机扩展。

设计要点
- 全局唯一 ID：类级计数器 `_count` 保证不同实例不会重复
- 状态机：New → Tracked → Lost → Removed 四态流转，便于生命周期管理
- 特征缓存：`features` / `curr_feature` 支持 ReID 及长期更新
- 跨相机：`location` 记录最近一次出现的世界坐标或相机编号
- 历史轨迹：`OrderedDict` 按帧号插入，O(1) 查询最新位置
"""

import numpy as np
from collections import OrderedDict


class TrackState(object):
    """
    跟踪目标状态枚举
    通过整数常量减少内存占用，方便状态判断与日志打印。
    """
    New = 0  # 刚被检测器发现，尚未确认
    Tracked = 1  # 已激活并持续跟踪
    Lost = 2  # 连续若干帧未匹配，可能暂时遮挡
    Removed = 3  # 确认消失，移出跟踪列表


class BaseTrack:
    """
    跟踪目标抽象基类
    子类需实现 `activate / predict / update` 以接入具体滤波器或 ReID 策略。
    """

    # ---------- 类级计数器 ----------
    _count: int = 0  # 已分配的最大 track_id

    # ---------- 实例属性 ----------
    track_id: int = 0
    is_activated: bool = False
    state: int = TrackState.New

    history: OrderedDict = OrderedDict()  # {frame_id: xywh}
    features: list = []                   # 历史 ReID 特征列表
    curr_feature: np.ndarray | None = None
    score: float = 0.0                    # 检测置信度
    start_frame: int = 0                  # 首次出现帧号
    frame_id: int = 0                     # 最近更新帧号
    time_since_update: int = 0            # 距上次匹配的帧数

    # 多相机扩展：可存 (camera_id, x, y) 或世界坐标
    location: tuple[float, float] = (np.inf, np.inf)

    # ---------- 只读属性 ----------
    @property
    def end_frame(self) -> int:
        """当前目标在逻辑上存在的最后一帧（与 frame_id 同步）"""
        return self.frame_id

    # ---------- ID 生成 ----------
    @staticmethod
    def next_id() -> int:
        """分配下一个全局唯一 track_id"""
        BaseTrack._count += 1
        return BaseTrack._count

    # ---------- 子类必须实现 ----------
    def activate(self, *args, **kwargs) -> None:
        """首次关联检测框 → 正式跟踪"""
        raise NotImplementedError

    def predict(self) -> None:
        """基于运动/外观模型预测下一帧位置"""
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        """匹配成功后更新状态、特征与轨迹"""
        raise NotImplementedError

    # ---------- 状态切换 ----------
    def mark_lost(self) -> None:
        """标记为 Lost，等待重新匹配或最终移除"""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """确认目标已永久消失，从跟踪器中移除"""
        self.state = TrackState.Removed
