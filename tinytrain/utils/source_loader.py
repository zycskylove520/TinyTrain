from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class SourceParser(ABC):
    """
    统一「任意来源 → Python 对象流」的解析器抽象基类。

    主要功能
    --------
    1. 屏蔽文件类型差异：图片、视频、文本、甚至内存对象都可迭代输出。
    2. 约定流式接口 `stream()`：每次 yield 一条样本，结束后 yield None 标识 EOF。
    3. 与 SourceParserHub 配合实现「自动路由」，用户无需关心具体解析器。

    用法示例
    --------
    >>> parser = SourceParserHub.auto("demo.jpg")   # 自动返回 ImageParser
    >>> for data_info in parser.stream("demo.jpg"):
    ...     if data_info is None: break
    ...     print(data_info.img.shape)
    """

    @abstractmethod
    def stream(self, source: Any) -> Iterable[Any]:
        """
        将任意来源解析为数据对象流。

        Args:
            source (Any): 输入源，可以是路径、URL、摄像头索引或内存对象。

        Yields:
            Any: 单条数据封装对象（如 ImgDataInfo / TextDataInfo / AnyDataInfo），
                 流结束后 yield None 表示终止。
        """
        pass


class ImageParser(SourceParser):
    """
    图片文件解析器，支持常见静态图格式。
    """
    def stream(self, source):
        """
        逐张图片生成 ImgDataInfo。

        Args:
            source (str | Path): 图片文件路径。

        Yields:
            ImgDataInfo: 包含 img, origin_shape, current_shape, img_file 等信息。
            None: 迭代结束标志。
        """
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
    """
    视频文件/流解析器，按帧输出。
    """
    def stream(self, source):
        """
        逐帧生成 ImgDataInfo。

        Args:
            source (str | Path): 视频文件路径或设备索引。

        Yields:
            ImgDataInfo: 包含 frame_id, img, origin_shape 等信息。
            None: 迭代结束标志。
        """
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
    """
    纯文本文件解析器，按行输出。
    """
    def stream(self, source):
        """
        逐行读取文本并生成 TextDataInfo。

        Args:
            source (str | Path): 文本文件路径。

        Yields:
            TextDataInfo: 封装单行文本内容。
            None: 迭代结束标志。
        """
        from tinytrain.data import TextDataInfo

        with open(source, encoding="utf-8") as f:
            for line in f:
                yield TextDataInfo(text=line.rstrip("\n"))
        yield None


class NullParser(SourceParser):
    """
    透传解析器：将用户传入的任何对象原样包装为 AnyDataInfo。
    """

    def stream(self, source):
        """
        直接包装输入对象。

        Args:
            source (Any): 任意对象。

        Yields:
            AnyDataInfo: 封装后的统一数据对象。
            None: 迭代结束标志。
        """
        from tinytrain.data import AnyDataInfo
        yield AnyDataInfo(data=source)
        yield None


class SourceParserHub:
    """
    解析器自动路由中心。

    主要功能
    --------
    1. 根据文件后缀或对象类型自动匹配对应的 SourceParser。
    2. 支持运行时注册新的解析器，无需改动框架源码。
    3. 未命中策略：返回 NullParser，保证下游流程始终可迭代。

    用法示例
    --------
    >>> SourceParserHub.register("gif", ImageParser)   # 动态注册
    >>> parser = SourceParserHub.auto("sample.gif")    # 返回 ImageParser 实例
    """

    _parsers: dict[str, type[SourceParser]] = {}

    @classmethod
    def register(cls, suffix: str, parser: type[SourceParser]):
        """
        注册新的后缀解析器。

        Args:
            suffix (str): 文件后缀（不含 '.'，不区分大小写）。
            parser (type[SourceParser]): 解析器类，须继承 SourceParser。
        """
        cls._parsers[suffix] = parser

    @classmethod
    def auto(cls, source: Any) -> SourceParser:
        """
        根据输入自动选择最合适的解析器。

        Args:
            source (Any): 输入源，可以是路径、URL、已封装的数据对象等。

        Returns:
            SourceParser: 对应的解析器实例；未匹配时返回 NullParser。
        """
        from tinytrain.data import BaseDataInfo

        # 已封装的数据对象直接透传
        if isinstance(source, BaseDataInfo):
            return NullParser()

        # 路径/字符串：按后缀匹配
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower().lstrip(".")
            parser_cls = cls._parsers.get(suffix, NullParser)
            return parser_cls()

        # 其它类型统一透传
        return NullParser()


# 内置解析器注册
SourceParserHub.register("jpg", ImageParser)
SourceParserHub.register("jpeg", ImageParser)
SourceParserHub.register("png", ImageParser)
SourceParserHub.register("bmp", ImageParser)
SourceParserHub.register("mp4", VideoParser)
SourceParserHub.register("avi", VideoParser)
SourceParserHub.register("mov", VideoParser)
SourceParserHub.register("txt", TextFileParser)
