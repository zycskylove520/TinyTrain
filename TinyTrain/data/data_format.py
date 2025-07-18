from __future__ import annotations

import numpy as np

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Dict, Any

if TYPE_CHECKING:
    import torch


class BaseDataInfo:
    """
    基础数据信息类，用于存储基本的数据信息，提供通用的深拷贝方法。。
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._exclude_from_deepcopy = set()  # 定义不参与深拷贝的字段集合
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __deepcopy__(self, memodict: Dict[int, Any]) -> 'BaseDataInfo':
        """
        自定义深拷贝方法，动态处理所有字段。
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
        super().__init__(**kwargs)
        self.data = data


class TextDataInfo(BaseDataInfo):
    def __init__(self,
                 text: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.text = text


class ImgDataInfo(BaseDataInfo):
    """
    基础图像信息类，用于存储图像相关的基本信息。
    """

    def __init__(self,
                 img: np.ndarray | None = None,
                 origin_shape: tuple[int, int] | None = None,
                 current_shape: tuple[int, int] | None = None,
                 target_shape: tuple[int, int] | None = None,
                 img_file: Path | None = None,
                 next_ImgDataInfo=None,
                 affine_matrix=np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]], dtype=np.float32),
                 **kwargs
                 ) -> None:
        """
        初始化基础图像信息。

        :param img: 图像数据（NumPy数组）
        :param origin_shape: 图像最初的尺寸 (宽度, 高度)
        :param current_shape: 图像经过一次图像增强算子变换结束后的尺寸 (宽度, 高度)
        :param target_shape: 图像增强结束后的尺寸 (宽度, 高度)
        :param img_file: 图像文件路径
        :param next_ImgDataInfo: 下一个ImgDataInfo，如果要做多张图片融合增强等操作则需要
        :param affine_matrix: 图像所经历的仿射变换矩阵,可用于验证集和推理
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.img = img
        self.origin_shape = origin_shape
        self.current_shape = current_shape
        self.target_shape = target_shape
        self.img_file = img_file
        self.next_ImgDataInfo = next_ImgDataInfo
        self.affine_matrix = affine_matrix

        # self._exclude_from_deepcopy.add('next_ImgDataInfo')  # dataloader多进程拷贝存在问题，暂时不开放


class ClassifyDataInfo(ImgDataInfo):
    """
    分类数据信息类，继承自基础图像信息类，用于存储分类任务相关的数据。
    """

    def __init__(self,
                 label: np.ndarray | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化分类数据信息。

        :param label: 分类标签
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.label = label


class DetectDataInfo(ClassifyDataInfo):
    """
    检测数据信息类，继承自分类数据信息类，用于存储目标检测任务相关的数据。
    """

    def __init__(self,
                 bboxes: np.ndarray | None = None,
                 bbox_format: Literal["lxlyrxry", "lxlywh", "cxcywh"] = "cxcywh",
                 normalized: bool = True,
                 **kwargs
                 ) -> None:
        """
        初始化检测数据信息。

        :param bboxes: 边界框坐标
        :param bbox_format: 边界框格式（"lxlyrxry"、"lxlywh" 或 "cxcywh"）
        :param normalized: 边界框坐标是否归一化
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.bboxes = bboxes
        self.bbox_format = bbox_format
        self.normalized = normalized

    def move(self, move_x: int, move_y: int):
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
        assert not self.normalized and self.bboxes is not None

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            return

        self.bboxes[..., 0::2] *= scale_x
        self.bboxes[..., 1::2] *= scale_y

    def affine_transform(self):
        assert not self.normalized and self.bboxes is not None

        # in yolo dataset, background image bboxes shape is: [0, 4]
        if self.bboxes.shape[0] == 0:
            return

        old_box_format = self.bbox_format
        self.convert_format(box_format="lxlyrxry")
        new_bboxes = []

        for bbox in self.bboxes:
            lx, ly, rx, ry = bbox

            # Transform 4 corners
            corners = np.array([
                [lx, ly],
                [rx, ly],
                [lx, ry],
                [rx, ry]
            ], dtype=np.float32)

            ones = np.ones((4, 1), dtype=np.float32)
            corners_homo = np.hstack([corners, ones])  # [x, y, 1]
            transformed = (self.affine_matrix @ corners_homo.T).T

            transformed_lx = transformed[:, 0].min()
            transformed_ly = transformed[:, 1].min()
            transformed_rx = transformed[:, 0].max()
            transformed_ry = transformed[:, 1].max()

            new_bboxes.append([transformed_lx, transformed_ly, transformed_rx, transformed_ry])
        self.bboxes = np.array(new_bboxes)
        self.convert_format(box_format=old_box_format)

    def normalize(self, w, h):
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
        from TinyTrain.utils.box_utils import (lxlyrxry_2_cxcywh,
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
        assert self.bboxes is not None
        return len(self.bboxes)

    def __getitem__(self, index: int) -> np.ndarray:
        assert self.bboxes is not None
        return self.bboxes[index]


class SegmentDataInfo(DetectDataInfo):
    """
    分割数据信息类，继承自检测数据信息类，用于存储图像分割任务相关的数据。
    """

    def __init__(self,
                 segments: list | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化分割数据信息。

        :param segments: 分割区域列表
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.segments = segments


class PoseDataInfo(DetectDataInfo):
    """
    姿态估计数据信息类，继承自检测数据信息类，用于存储姿态估计任务相关的数据。
    """

    def __init__(self,
                 key_points: list | None = None,
                 kpt_shape: list | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化姿态估计数据信息。

        :param key_points: 关键点列表
        :param kpt_shape: 关键点形状
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.key_points = key_points
        self.kpt_shape = kpt_shape


class BaseBatchDataInfo:
    """
    基础批量数据信息类，用于存储批量数据的基本信息。
    """

    def __init__(self,
                 data: torch.Tensor | list[torch.Tensor] | None = None
                 ):
        """
        初始化基础批量数据信息。

        :param data: 批量数据（PyTorch张量或张量列表）
        """
        self.data = data


class ImgBatchDataInfo(BaseBatchDataInfo):
    """
    图像批量数据信息类，用于存储批量图像数据及其相关属性。

    该类继承自 BaseBatchDataInfo，用于存储批量图像数据的形状信息，包括原始形状和调整后的形状。
    这些信息通常在图像预处理和后处理阶段非常有用，例如在批量图像大小调整或恢复原始图像大小时。
    """

    def __init__(self,
                 origin_shapes: torch.Tensor | None = None,
                 target_shapes: torch.Tensor | None = None,
                 **kwargs  # 其他关键字参数（传递给父类）
                 ) -> None:
        """
        初始化图像批量数据信息。

        :param origin_shapes: 批量图像的原始尺寸组成的tensor，每一行是一个图像(宽度, 高度)。
        :param target_shapes: 批量图像经过图像增强变换完成的尺寸组成的tensor，每一行是一个图像(宽度, 高度)。
        :param kwargs: 其他关键字参数，将传递给父类 BaseBatchDataInfo。
        """
        super().__init__(**kwargs)
        self.origin_shapes = origin_shapes
        self.target_shapes = target_shapes


class ClassifyBatchDataInfo(ImgBatchDataInfo):
    """
    分类批量数据信息类，继承自基础批量数据信息类，用于存储分类任务相关的批量数据。
    """

    def __init__(self,
                 target: torch.Tensor | None = None,
                 **kwargs
                 ) -> None:
        """
        初始化分类批量数据信息。

        :param target: 分类目标（PyTorch张量）
        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
        self.target = target


class DetectBatchDataInfo(ClassifyBatchDataInfo):
    """
    检测批量数据信息类，继承自分类批量数据信息类，用于存储目标检测任务相关的批量数据。
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
    """
    分割批量数据信息类，继承自检测批量数据信息类，用于存储图像分割任务相关的批量数据。
    """

    def __init__(self,
                 **kwargs
                 ) -> None:
        """
        初始化分割批量数据信息。

        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)


class PoseBatchDataInfo(DetectBatchDataInfo):
    """
    姿态估计批量数据信息类，继承自检测批量数据信息类，用于存储姿态估计任务相关的批量数据。
    """

    def __init__(self,
                 **kwargs
                 ) -> None:
        """
        初始化姿态估计批量数据信息。

        :param kwargs: 其他关键字参数（传递给父类）
        """
        super().__init__(**kwargs)
