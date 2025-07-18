from __future__ import annotations

import random
import numpy as np

from typing import TYPE_CHECKING

from .data_format import BaseDataInfo, ImgDataInfo, ClassifyDataInfo, DetectDataInfo

# 仅在类型检查阶段导入
if TYPE_CHECKING:
    import albumentations as A
    import cv2


def normalize(img, mean, std, max_pixel_value=255.) -> np.ndarray:
    import albucore
    denominator = np.reciprocal(np.array(std, dtype=np.float32) * max_pixel_value)
    return albucore.normalize(img, mean, denominator)


class DynamicFilling:
    def __init__(
            self,
            target_size: tuple[int, int],
            p: float,
            task: str = "classify",
            fill_value=114,
    ):
        assert 0.0 <= p <= 1.0, "p should be in [0, 1]"
        assert task in {"classify", "detect"}, "only classify / detect supported now"
        self.target_size = target_size  # (w, h)
        self.p = p
        self.task = task
        self.fill_value = fill_value

    def __call__(self, sample: BaseDataInfo) -> BaseDataInfo:
        if random.random() < self.p:
            transform = self._pad_branch()
        else:
            transform = self._resize_branch()

        if self.task == "classify":
            assert isinstance(sample, ClassifyDataInfo)
            out = transform(image=sample.img)
            sample.img = out["image"]

        elif self.task == "detect":
            assert isinstance(sample, DetectDataInfo)
            out = transform(
                image=sample.img,
                bboxes=sample.bboxes,
                class_labels=sample.label,
            )
            sample.img = out["image"]
            sample.bboxes = out["bboxes"]
            sample.label = out["class_labels"]

        else:
            raise NotImplementedError(f"task {self.task} not implemented")

        sample.current_shape = sample.img.shape[:2][::-1]
        return sample

    def _pad_branch(self):
        """保持宽高比 + 填充"""
        import albumentations as A
        import cv2

        tf = [
            A.LongestMaxSize(max_size_hw=(self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.target_size[1],
                min_width=self.target_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                fill=self.fill_value,
            ),
        ]
        if self.task == "detect":
            return A.Compose(
                tf,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_area=100,
                    min_visibility=0.1,
                    filter_invalid_bboxes=True,
                ),
                p=1.0,
            )
        else:  # classify
            return A.Compose(tf, p=1.0)

    def _resize_branch(self):
        """直接拉伸"""
        import albumentations as A
        import cv2

        tf = [A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LINEAR)]
        if self.task == "detect":
            return A.Compose(
                tf,
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_area=100,
                    min_visibility=0.1,
                    filter_invalid_bboxes=True,
                ),
                p=1.0,
            )
        else:  # classify
            return A.Compose(tf, p=1.0)


class Mosaic:
    def __init__(self,
                 task: str = "detect",
                 layout: str = "3x3"
                 ):
        assert task in ["classify", "detect", "segment", "pose"]
        assert layout in ["2x2", "3x3", ]
        self.task = task
        self.layout = layout

    def __call__(self, sample: BaseDataInfo):
        pass

    def mosaic_2x2(self, sample: ImgDataInfo):
        pass

    def mosaic_3x3(self, sample: ImgDataInfo):
        pass


if __name__ == '__main__':

    from TinyTrain.data import DetectDataInfo, DynamicFilling
    from TinyTrain.utils.box_utils import cxcywh_2_lxlyrxry
    from TinyTrain.utils.data_utils import cv_imread

    img = cv_imread(r"9.jpg")
    img_data = DetectDataInfo(
        img=img,
        origin_shape=img.shape[:2][::-1],
        current_shape=img.shape[:2][::-1],
        target_shape=(320, 608),
        bboxes=np.array([[0.398133, 0.339827, 0.060877, 0.080808],
                         [0.474432, 0.329726, 0.060877, 0.101010]]),
        label=np.array([0, 1]),
    )
    w, h = img_data.target_shape

    df = DynamicFilling(target_size=img_data.target_shape, p=1, task="detect")
    res = df(img_data)
    bboxes = cxcywh_2_lxlyrxry(res.bboxes)
    for bbox in bboxes:
        decode_lx = int(bbox[0] * w)
        decode_ly = int(bbox[1] * h)
        decode_rx = int(bbox[2] * w)
        decode_ry = int(bbox[3] * h)
        print(decode_lx, decode_ly, decode_rx, decode_ry)

        cv2.rectangle(res.img, (decode_lx, decode_ly), (decode_rx, decode_ry), (255, 0, 0), 2)
    cv2.imwrite(f"test2.jpg", res.img)
