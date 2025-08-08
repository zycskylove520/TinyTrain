from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Dict, Any

if TYPE_CHECKING:
    import torch
    import numpy as np


# region
# -----------------------------------------------------------------------------
# 基础数据容器
# -----------------------------------------------------------------------------
class BaseDataInfo:
    """
    所有数据信息类的 **根容器**。

    功能
    ----
    - **动态字段管理**：构造函数支持任意关键字参数，自动绑定为成员变量。
    - **智能深拷贝**：通过 __deepcopy__ 跳过指定字段，避免多进程/多线程拷贝大对象或共享资源。
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: 任意键值对，将动态绑定为成员变量。
        """
        super().__init__()
        self._exclude_from_deepcopy = set()  # 定义不参与深拷贝的字段集合
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __deepcopy__(self, memodict: Dict[int, Any]) -> 'BaseDataInfo':
        """
        自定义深拷贝，跳过 _exclude_from_deepcopy 中的字段。

        Args:
            memodict: deepcopy 内部缓存字典。

        Returns:
            BaseDataInfo: 深拷贝后的新实例。
        """
        new_instance = self.__class__.__new__(self.__class__)  # 创建一个新的实例
        for key, value in self.__dict__.items():
            if key in self._exclude_from_deepcopy:
                # 如果字段在排除列表中，直接赋值
                setattr(new_instance, key, value)
            else:
                # 否则，递归深拷贝
                setattr(new_instance, key, deepcopy(value, memodict))
        return new_instance


class AnyDataInfo(BaseDataInfo):
    """透传任意 BaseDataInfo 对象"""

    def __init__(self, data: BaseDataInfo, **kwargs):
        """
        Args:
            data (BaseDataInfo): 被包装的原始数据对象。
            **kwargs: 透传给父类的额外字段。
        """
        super().__init__(**kwargs)
        self.data = data


class TextDataInfo(BaseDataInfo):
    """
    纯文本任务的数据容器。
    """

    def __init__(self,
                 text: str,
                 **kwargs):
        """
        Args:
            text (str): 文本内容。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.text = text


class ImgDataInfo(BaseDataInfo):
    """
    单张图像的 **通用元数据** 容器，支持帧号、仿射矩阵、前后帧指针等。

    典型用途
    --------
    - 视频帧序列：frame_id + next_ImgDataInfo。
    - Mosaic/Copy-Paste：next_ImgDataInfo 指向下一张待拼接图像。
    - 仿射变换：affine_matrix 记录当前图像所有几何变换，推理阶段用于坐标回推。
    """

    def __init__(self,
                 frame_id: int = 0,
                 img: np.ndarray | None = None,
                 origin_shape: tuple[int, int] | None = None,
                 target_shape: tuple[int, int] | None = None,
                 img_file: Path | None = None,
                 next_ImgDataInfo=None,
                 **kwargs
                 ) -> None:
        """
        Args:
            frame_id: 视频帧序号（文件/摄像头场景下可设为 0）。
            img: HWC 格式的 numpy 数组。
            origin_shape: 原始宽高 (W, H)。
            target_shape: 最终模型输入宽高 (W, H)。
            img_file: 图像文件路径。
            next_ImgDataInfo: 下一张图（用于 Mosaic 等融合增强）。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.frame_id = frame_id
        self.img = img
        self.origin_shape = origin_shape
        self.target_shape = target_shape
        self.img_file = img_file
        self.next_ImgDataInfo = next_ImgDataInfo

        # self._exclude_from_deepcopy.add('next_ImgDataInfo')  # dataloader多进程拷贝存在问题，暂时不开放


class ClassifyDataInfo(ImgDataInfo):
    """
    分类任务专用单张图像数据容器。
    """

    def __init__(self,
                 label: np.ndarray | None = None,
                 **kwargs
                 ) -> None:
        """
        Args:
            label (np.ndarray | None): 类别索引数组（支持多标签）。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.label = label


class DetectDataInfo(ClassifyDataInfo):
    """
    检测任务专用单张图像数据容器，支持 bbox 坐标变换与格式转换。
    """

    def __init__(self,
                 scores: np.ndarray | None = None,
                 bboxes: np.ndarray | None = None,
                 bbox_format: Literal["lxlyrxry", "lxlywh", "cxcywh"] = "cxcywh",
                 normalized: bool = True,
                 **kwargs
                 ) -> None:
        """
        Args:
            scores: 每个 bbox 的置信度。
            bboxes: 边界框坐标 [N, 4]。
            bbox_format: bbox 字符串格式。
            normalized: 坐标是否已归一化到 [0, 1]。
            **kwargs: 透传给父类。
        """

        super().__init__(**kwargs)
        self.scores = scores
        self.bboxes = bboxes
        self.bbox_format = bbox_format
        self.normalized = normalized

    def move(self, move_x: int, move_y: int):
        """在像素坐标下平移 bbox（仅当 normalized=False 时有效）。"""
        assert not self.normalized and self.bboxes is not None

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            return

        if self.bbox_format == "cxcywh" or self.bbox_format == "lxlywh":
            self.bboxes[..., 0] += move_x
            self.bboxes[..., 1] += move_y
        elif self.bbox_format == "lxlyrxry":
            self.bboxes[..., 0] += move_x
            self.bboxes[..., 1] += move_y
            self.bboxes[..., 2] += move_x
            self.bboxes[..., 3] += move_y
        else:
            raise ValueError(f"bbox_format {self.bbox_format} is not supported.")

    def scale(self, scale_x: float, scale_y: float):
        """在像素坐标下缩放 bbox（仅当 normalized=False 时有效）。"""
        assert not self.normalized and self.bboxes is not None

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            return

        self.bboxes[..., 0::2] *= scale_x
        self.bboxes[..., 1::2] *= scale_y

    def normalize(self, w, h):
        """将像素坐标归一化到 [0, 1]。"""
        assert self.bboxes is not None

        if self.normalized:
            return

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            self.normalized = True
            return

        self.bboxes[..., 0::2] /= w
        self.bboxes[..., 1::2] /= h
        self.normalized = True

    def denormalize(self, w, h):
        """将归一化坐标恢复到像素坐标。"""
        if not self.normalized:
            return

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            self.normalized = False
            return

        assert self.bboxes is not None
        self.bboxes[..., 0::2] *= w
        self.bboxes[..., 1::2] *= h
        self.normalized = False

    def convert_format(self, box_format="cxcywh"):
        """
        在 3 种 bbox 格式间任意转换，同时支持归一化/反归一化保持。

        Args:
            box_format: 目标格式。
        """
        from tinytrain.utils.box_utils import (lxlyrxry_2_cxcywh,
                                               lxlywh_2_cxcywh,
                                               cxcywh_2_lxlyrxry,
                                               lxlywh_2_lxlyrxry,
                                               cxcywh_2_lxlywh,
                                               lxlyrxry_2_lxlywh)

        assert self.bboxes is not None

        if self.bbox_format == box_format:
            return

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            self.bbox_format = box_format
            return

        if box_format == "cxcywh":
            if self.bbox_format == "lxlyrxry":
                func = lxlyrxry_2_cxcywh
            elif self.bbox_format == "lxlywh":
                func = lxlywh_2_cxcywh
            else:
                raise ValueError(f"bbox_format {self.bbox_format} is not supported.")
        elif box_format == "lxlyrxry":
            if self.bbox_format == "cxcywh":
                func = cxcywh_2_lxlyrxry
            elif self.bbox_format == "lxlywh":
                func = lxlywh_2_lxlyrxry
            else:
                raise ValueError(f"bbox_format {self.bbox_format} is not supported.")
        elif box_format == "lxlywh":
            if self.bbox_format == "cxcywh":
                func = cxcywh_2_lxlywh
            elif self.bbox_format == "lxlyrxry":
                func = lxlyrxry_2_lxlywh
            else:
                raise ValueError(f"bbox_format {self.bbox_format} is not supported.")
        else:
            raise ValueError(f"bbox_format {self.bbox_format} is not supported.")

        self.bboxes = func(self.bboxes)
        self.bbox_format = box_format

    def __len__(self) -> int:
        """返回 bbox 数量。"""
        assert self.bboxes is not None
        return len(self.bboxes)

    def __getitem__(self, index: int) -> np.ndarray:
        """按索引返回单个 bbox。"""
        assert self.bboxes is not None
        return self.bboxes[index]


class SegmentDataInfo(DetectDataInfo):
    """
    分割任务数据容器，额外携带多边形/掩码。
    """

    def __init__(self,
                 segments: list | None = None,
                 **kwargs
                 ) -> None:
        """
        Args:
            segments (list | None):
                每个实例的多边形或 RLE 掩码列表，长度等于 bbox 数。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.segments = segments


class PoseDataInfo(DetectDataInfo):
    """
    姿态估计任务数据容器，额外携带关键点。
    """

    def __init__(self,
                 key_points: torch.Tensor | None = None,
                 kpt_shape: list | None = None,
                 **kwargs
                 ) -> None:
        """
        Args:
            key_points (Tensor | None):
                每个实例的关键点坐标，形状例如: [N, K, 3]（x, y, visible）。
            kpt_shape (list | None):
                单张图关键点维度信息，如 [17, 3]。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.key_points = key_points
        self.kpt_shape = kpt_shape


# endregion

# region
# -----------------------------------------------------------------------------
# 批数据容器
# -----------------------------------------------------------------------------
class BaseBatchDataInfo:
    """
    单个 batch 的 **通用容器**，data 字段可存放张量 / 列表 / 任意对象。
    """

    def __init__(self,
                 data: torch.Tensor | list[torch.Tensor] | Any | None = None
                 ):
        """
        Args:
            data: 一批原始输入或中间特征。
        """
        self.data = data


class ImgBatchDataInfo(BaseBatchDataInfo):
    """
    图像 batch 元数据，记录每张图的原始/目标尺寸，便于后处理还原。
    """

    def __init__(self,
                 origin_shapes: torch.Tensor | None = None,
                 target_shapes: torch.Tensor | None = None,
                 **kwargs  # 其他关键字参数（传递给父类）
                 ) -> None:
        """
        Args:
            origin_shapes: [B, 2] 原始宽高 (W, H)。
            target_shapes: [B, 2] 增强后宽高 (W, H)。
            **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.origin_shapes = origin_shapes
        self.target_shapes = target_shapes


class ClassifyBatchDataInfo(ImgBatchDataInfo):
    """分类 batch 容器，额外携带标签张量。"""

    def __init__(self,
                 target: torch.Tensor | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化分类批量数据信息。

        Args:
           target: 分类目标（PyTorch张量）
           **kwargs: 透传给父类。
        """
        super().__init__(**kwargs)
        self.target = target


class DetectBatchDataInfo(ClassifyBatchDataInfo):
    """
    检测 batch 容器，额外携带 bbox 及索引。

    说明
    ----
    - bboxes: [N, 4] 全部 bbox，按 cxcywh 或 lxlyrxry 格式。
    - bboxes_idx: [N] long Tensor，标示每个 bbox 属于 batch 中第几张图。
    """

    def __init__(self,
                 bboxes: torch.Tensor | None = None,
                 bboxes_idx: torch.Tensor | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化检测批量数据信息。

        :param bboxes: 边界框坐标（PyTorch张量）
        :param bboxes_idx: 边界框索引（PyTorch张量）,指明每个边界框来自该批次的第几张图片，idx必须是long, int, byte or bool
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.bboxes = bboxes
        self.bboxes_idx = bboxes_idx  # 记录每个bbox来自哪张图片


class SegmentBatchDataInfo(DetectBatchDataInfo):
    """分割 batch 容器，目前与检测保持一致，未来可拓展 segments 字段。"""

    def __init__(self,
                 **kwargs
                 ) -> None:
        """
        初始化分割批量数据信息。

        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)


class PoseBatchDataInfo(DetectBatchDataInfo):
    """姿态估计 batch 容器，未来可拓展 key_points 字段。"""

    def __init__(self,
                 batch_key_points: torch.Tensor | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化姿态估计批量数据信息。

        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.batch_key_points = batch_key_points
# endregion
