from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class SourceParser(ABC):
    """解析任意来源 -> Python 对象流"""

    @abstractmethod
    def stream(self, source: Any) -> Iterable[Any]:
        """yield 单个样本，直到结束再 yield None"""
        pass


class ImageParser(SourceParser):
    def stream(self, source):
        from tinytrain.data import ImgDataInfo
        from tinytrain.utils.data_utils import cv_imread

        img = cv_imread(str(source))

        yield ImgDataInfo(
            img=img,
            origin_shape=img.shape[:2][::-1],
            current_shape=img.shape[:2][::-1],
            target_shape=None,
            img_file=Path(source)
        )
        yield None


class VideoParser(SourceParser):
    def stream(self, source):
        from tinytrain.data import ImgDataInfo
        import cv2

        cap = cv2.VideoCapture(str(source))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 从0开始计数
            yield ImgDataInfo(
                frame_id=frame_id,
                img=frame,
                origin_shape=frame.shape[:2][::-1],
                current_shape=frame.shape[:2][::-1],
                target_shape=None,
                img_file=None
            )
        cap.release()
        yield None


class TextFileParser(SourceParser):
    def stream(self, source):
        from tinytrain.data import TextDataInfo

        with open(source, encoding="utf-8") as f:
            for line in f:
                yield TextDataInfo(text=line.rstrip("\n"))
        yield None


class NullParser(SourceParser):
    """直接把用户给的对象包装为 AnyDataInfo"""

    def stream(self, source):
        from tinytrain.data import AnyDataInfo
        yield AnyDataInfo(data=source)
        yield None


class SourceParserHub:
    _parsers = {}

    @classmethod
    def register(cls, suffix: str, parser: type[SourceParser]):
        cls._parsers[suffix] = parser

    @classmethod
    def auto(cls, source) -> SourceParser:
        from tinytrain.data import BaseDataInfo
        if isinstance(source, BaseDataInfo):
            return NullParser()  # 透传
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower().lstrip(".")
            return cls._parsers.get(suffix, NullParser)()
        return NullParser()  # 其它类型透传


# 内置注册
SourceParserHub.register("jpg", ImageParser)
SourceParserHub.register("jpeg", ImageParser)
SourceParserHub.register("png", ImageParser)
SourceParserHub.register("bmp", ImageParser)
SourceParserHub.register("mp4", VideoParser)
SourceParserHub.register("avi", VideoParser)
SourceParserHub.register("mov", VideoParser)
SourceParserHub.register("txt", TextFileParser)
