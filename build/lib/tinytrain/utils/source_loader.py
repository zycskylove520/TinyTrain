from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Type

from tinytrain.data.data_format import BaseDataInfo


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
        yield None


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
        from tinytrain.data.data_format import ImgDataInfo
        from tinytrain.utils.data_utils import cv_imread

        img = cv_imread(str(source))

        yield ImgDataInfo(
            img=img,
            origin_shape=img.shape[:2][::-1],
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
            source (str | Path | int): 视频文件路径或设备索引。

        Yields:
            ImgDataInfo: 包含 frame_id, img, origin_shape 等信息。
            None: 迭代结束标志。
        """
        from tinytrain.data.data_format import ImgDataInfo
        import cv2

        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if isinstance(source, int):
                frame_id += 1
            else:
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            yield ImgDataInfo(
                frame_id=frame_id,
                img=frame,
                origin_shape=frame.shape[:2][::-1],  # type: ignore[arg-type]
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
        from tinytrain.data.data_format import TextDataInfo

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
        from tinytrain.data.data_format import AnyDataInfo
        yield AnyDataInfo(data=source)
        yield None


class SourceParserHub:
    """
    解析器自动路由中心，用于根据输入源的类型或格式自动选择合适的解析器。

    主要功能：
    1. 根据文件后缀、对象类型或输入源的前缀自动匹配对应的 SourceParser。
    2. 支持运行时注册新的解析器，无需改动框架源码。
    3. 未命中策略：如果没有找到匹配的解析器，抛出 ValueError 异常。

    设计理念：
    - 提供一个灵活的解析器选择机制，支持多种类型的输入源。
    - 通过注册机制，方便扩展新的解析器类型。
    - 保证代码的可维护性和可扩展性。

    Usage Example:
    >>> SourceParserHub.register("gif", ImageParser)   # 动态注册解析器
    >>> parser = SourceParserHub.auto("sample.gif")    # 自动选择解析器
    >>> for data_info in parser.stream("sample.gif"):  # 使用解析器
    ...     if data_info is None: break
    ...     print(data_info.img.shape)

    Attributes：
        _parsers (dict): 以文件后缀为键，解析器类为值的字典。
        _type_parsers (dict): 以类型为键，解析器类为值的字典。
        _prefix_parsers (dict): 以输入源前缀为键，解析器类为值的字典。

    Methods：
        register(suffix, parser): 注册新的文件后缀解析器。
        register_type(source_type, parser): 注册新的类型解析器。
        register_prefix(prefix, parser): 注册新的前缀解析器。
        auto(source): 自动选择解析器。
    """

    _parsers: dict[str, Type[SourceParser]] = {}
    _type_parsers: dict[type, Type[SourceParser]] = {}
    _prefix_parsers: dict[str, Type[SourceParser]] = {}

    @classmethod
    def register(cls, suffix: str, parser: Type[SourceParser]):
        """
        注册新的文件后缀解析器。

        Args:
            suffix (str): 文件后缀（不含 '.'，不区分大小写）。
            parser (Type[SourceParser]): 解析器类，须继承自 SourceParser。

        示例：
        >>> SourceParserHub.register("jpg", ImageParser)
        """
        cls._parsers[suffix.lower()] = parser

    @classmethod
    def register_type(cls, source_type: type, parser: Type[SourceParser]):
        """
        注册新的类型解析器。

        Args:
            source_type (type): 输入源的类型。
            parser (Type[SourceParser]): 解析器类，须继承自 SourceParser。

        示例：
        >>> SourceParserHub.register_type(int, VideoParser)
        """
        cls._type_parsers[source_type] = parser

    @classmethod
    def register_prefix(cls, prefix: str, parser: Type[SourceParser]):
        """
        注册新的前缀解析器。

        Args:
            prefix (str): 输入源的前缀（不区分大小写）。
            parser (Type[SourceParser]): 解析器类，须继承自 SourceParser。

        示例：
        >>> SourceParserHub.register_prefix("rtsp://", VideoParser)
        """
        cls._prefix_parsers[prefix.lower()] = parser

    @classmethod
    def _select_parser(cls, source: Any) -> SourceParser:
        """
        动态选择解析器。

        根据输入源的类型或格式选择合适的解析器。选择顺序如下：
        1. 如果输入源是 BaseDataInfo 实例，返回 NullParser。
        2. 如果输入源的类型在 _type_parsers 中注册，返回对应的解析器。
        3. 如果输入源是字符串且以某个前缀开头，返回对应的解析器。
        4. 如果输入源是字符串或 Path 对象，根据文件后缀选择解析器。
        5. 如果没有找到匹配的解析器，抛出 ValueError 异常。

        Args:
            source (Any): 输入源，可以是路径、URL、已封装的数据对象等。

        Returns:
            SourceParser: 选择的解析器实例。

        Raises:
            ValueError: 如果没有找到匹配的解析器。
        """
        # 已封装的数据对象直接透传
        if isinstance(source, BaseDataInfo):
            return NullParser()

        # 按类型选择解析器
        if type(source) in cls._type_parsers:
            return cls._type_parsers[type(source)]()

        # 按前缀选择解析器
        if isinstance(source, str):
            for prefix, parser in cls._prefix_parsers.items():
                if source.lower().startswith(prefix):
                    return parser()

        # 按后缀选择解析器
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower().lstrip(".")
            if suffix in cls._parsers:
                return cls._parsers[suffix]()

        # 其它类型统一透传
        return NullParser()

    @classmethod
    def auto(cls, source: Any) -> SourceParser:
        """
        自动选择解析器。

        调用 _select_parser 方法选择合适的解析器。

        Args:
            source (Any): 输入源，可以是路径、URL、已封装的数据对象等。

        Returns:
            SourceParser: 选择的解析器实例。

        Raises:
            ValueError: 如果没有找到匹配的解析器。
        """
        return cls._select_parser(source)


# 内置解析器注册
"""
注册常见的文件格式及其对应的解析器。
"""
SourceParserHub.register("jpg", ImageParser)  # 注册 JPEG 图片解析器
SourceParserHub.register("jpeg", ImageParser)  # 注册 JPEG 图片解析器
SourceParserHub.register("png", ImageParser)  # 注册 PNG 图片解析器
SourceParserHub.register("bmp", ImageParser)  # 注册 BMP 图片解析器
SourceParserHub.register("gif", ImageParser)  # 注册 GIF 图片解析器
SourceParserHub.register("webm", VideoParser)  # 注册 WebM 视频解析器
SourceParserHub.register("mkv", VideoParser)  # 注册 MKV 视频解析器
SourceParserHub.register("flv", VideoParser)  # 注册 FLV 视频解析器
SourceParserHub.register("mp4", VideoParser)  # 注册 MP4 视频解析器
SourceParserHub.register("avi", VideoParser)  # 注册 AVI 视频解析器
SourceParserHub.register("mov", VideoParser)  # 注册 MOV 视频解析器
SourceParserHub.register("txt", TextFileParser)  # 注册文本文件解析器

# 注册类型解析器
"""
注册基于类型的解析器。
"""
SourceParserHub.register_type(int, VideoParser)  # 注册整数类型的解析器，用于处理摄像头索引

# 注册前缀解析器
"""
注册基于前缀的解析器。
"""
SourceParserHub.register_prefix("rtsp://", VideoParser)  # 注册 RTSP 流的解析器
SourceParserHub.register_prefix("http://", VideoParser)  # 注册 HTTP 流的解析器
SourceParserHub.register_prefix("https://", VideoParser)  # 注册 HTTPS 流的解析器
