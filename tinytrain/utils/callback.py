"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import inspect

from dataclasses import dataclass
from typing import Callable, List, Dict

from tinytrain.utils import LOGGER


@dataclass
class Event:
    """
    单个事件类
    """
    name: str
    default_callback: Callable = lambda *args, **kwargs: None  # 支持任意参数


class Events:
    """
    事件定义类

    该类定义了训练、验证、推理、导出等各个阶段的生命周期事件。
    每个事件都是一个元组，包含两个元素：
    - 事件名称 (str): 事件的唯一标识符
    - 回调函数形式 (Callable): 调用者应传入的回调函数形式，参数类型根据事件阶段而定
    """

    """训练阶段"""
    ON_PREPARE_TRAIN_START = Event("on_prepare_train_start", lambda trainer: None)
    ON_PREPARE_TRAIN_END = Event("on_prepare_train_end", lambda trainer: None)
    ON_TRAIN_START = Event("on_train_start", lambda trainer: None)
    ON_TRAIN_EPOCH_START = Event("on_train_epoch_start", lambda trainer: None)
    ON_TRAIN_BATCH_START = Event("on_train_batch_start", lambda trainer: None)
    ON_BEFORE_ZERO_GRAD = Event("on_before_zero_grad", lambda trainer: None)
    ON_TRAIN_BATCH_END = Event("on_train_batch_end", lambda trainer: None)
    ON_TRAIN_EPOCH_END = Event("on_train_epoch_end", lambda trainer: None)
    ON_MODEL_SAVE = Event("on_model_save", lambda trainer: None)
    ON_TRAIN_END = Event("on_train_end", lambda trainer: None)

    """验证阶段"""
    ON_VAL_START = Event("on_val_start", lambda validator: None)
    ON_VAL_BATCH_START = Event("on_val_batch_start", lambda validator: None)
    ON_VAL_BATCH_END = Event("on_val_batch_end", lambda validator: None)
    ON_VAL_END = Event("on_val_end", lambda validator: None)

    """推理阶段"""
    ON_PREDICT_START = Event("on_predict_start", lambda predictor: None)
    ON_PREDICT_BATCH_START = Event("on_predict_batch_start", lambda predictor: None)
    ON_PREDICT_PREPROCESS_END = Event("on_predict_preprocess_end", lambda predictor: None)
    ON_PREDICT_INFERENCE_END = Event("on_predict_inference_end", lambda predictor: None)
    ON_PREDICT_BATCH_END = Event("on_predict_batch_end", lambda predictor: None)
    ON_PREDICT_END = Event("on_predict_end", lambda predictor: None)

    """导出阶段"""
    ON_EXPORT_START = Event("on_export_start", lambda exporter: None)
    ON_EXPORT_END = Event("on_export_end", lambda exporter: None)

    """自定义多参数示例"""
    ON_CUSTOM_EVENT = Event("on_custom_event", lambda a, b, c, d=4, e=5: None)  # 支持默认参数
    ON_COMPLEX_EVENT = Event("on_complex_event", lambda data, config, *args, **kwargs: None)  # 支持可变参数


class CallbackWrapper:
    """
    回调函数包装器

    对回调函数进行包装，提供优先级管理、单次执行、异常处理等功能，
    并包含严格的参数验证机制。

    属性:
        fn: Callable - 被包装的回调函数
        priority: int - 优先级，数值越大优先级越高
        once: bool - 是否只执行一次
        swallow_exceptions: bool - 是否吞掉异常
        _called: bool - 记录是否已被调用过
        sig: inspect.Signature - 回调函数的参数签名
        param_names: List[str] - 回调函数的参数名列表
        prototype: str - 格式化的函数原型字符串

    方法:
        __call__: 执行回调函数，包含参数验证和异常处理
        __lt__: 定义优先级比较规则，用于排序
        _generate_prototype: 生成易读的函数原型字符串
        _validate_arguments: 验证参数并绑定，参数不匹配时抛出详细异常
        _generate_argument_error: 生成详细的参数错误信息
    """

    def __init__(self, fn: Callable, priority: int = 0, once: bool = False, swallow_exceptions: bool = True, ):
        """
        初始化回调包装器

        Args:
            fn: 要包装的回调函数
            priority: 优先级，默认为0，数值越大越优先执行
            once: 是否只执行一次，默认为False
            swallow_exceptions: 是否吞掉异常，默认为True
        """
        self.fn = fn
        self.priority = priority
        self.once = once
        self.swallow_exceptions = swallow_exceptions
        self._called = False

        # 分析函数参数签名
        self.sig = inspect.signature(fn)
        self.param_names = list(self.sig.parameters.keys())

        # 生成函数原型字符串
        self.prototype = self._generate_prototype()

    def _generate_prototype(self) -> str:
        """生成易读的函数原型字符串"""
        params = []
        for name, param in self.sig.parameters.items():
            if param.default == param.empty:
                if param.kind == param.VAR_POSITIONAL:
                    params.append(f"*{name}")
                elif param.kind == param.VAR_KEYWORD:
                    params.append(f"**{name}")
                else:
                    params.append(name)
            else:
                params.append(f"{name}={param.default!r}")

        return f"{self.fn.__name__}({', '.join(params)})"

    def _validate_arguments(self, args, kwargs) -> tuple:
        """验证参数并绑定，如果参数不匹配则抛出详细异常"""
        try:
            # 尝试绑定参数
            bound_args = self.sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args.args, bound_args.kwargs
        except TypeError as e:
            # 生成详细的错误信息
            error_msg = self._generate_argument_error(args, kwargs, str(e))
            raise TypeError(error_msg) from e

    def _generate_argument_error(self, provided_args, provided_kwargs, original_error: str) -> str:
        """生成详细的参数错误信息"""
        # 统计提供的参数
        provided_positional = len(provided_args)
        provided_keyword = len(provided_kwargs)

        # 分析函数参数需求
        required_args = 0
        optional_args = 0
        has_var_args = False
        has_var_kwargs = False
        param_details = []

        for name, param in self.sig.parameters.items():
            if param.kind == param.VAR_POSITIONAL:
                has_var_args = True
                param_details.append(f"    *{name}: 可变位置参数")
            elif param.kind == param.VAR_KEYWORD:
                has_var_kwargs = True
                param_details.append(f"    **{name}: 可变关键字参数")
            elif param.default == param.empty:
                required_args += 1
                param_details.append(f"    {name}: 必需参数")
            else:
                optional_args += 1
                param_details.append(f"    {name}: 可选参数 (默认值: {param.default!r})")

        # 构建错误信息
        error_lines = [
            f"回调函数参数不匹配!",
            f"函数原型: {self.prototype}",
            f"原始错误: {original_error}",
            "",
            "函数参数要求:",
            f"  - 必需参数: {required_args} 个",
            f"  - 可选参数: {optional_args} 个",
            f"  - 支持可变位置参数: {'是' if has_var_args else '否'}",
            f"  - 支持可变关键字参数: {'是' if has_var_kwargs else '否'}",
            "",
            "详细参数信息:"
        ]
        error_lines.extend(param_details)

        error_lines.extend([
            "",
            "您提供的参数:",
            f"  - 位置参数: {provided_positional} 个 {list(provided_args)}",
            f"  - 关键字参数: {provided_keyword} 个 {list(provided_kwargs.keys())}",
            "",
            "建议的调用方式:",
            f"  {self.prototype}"
        ])

        # 提供具体示例
        if required_args > 0:
            example_args = [f"arg{i}" for i in range(required_args)]
            example_call = f"{self.fn.__name__}({', '.join(example_args)})"
            error_lines.append(f"  示例: {example_call}")

        return "\n".join(error_lines)

    def __call__(self, *args, **kwargs):
        """
        执行回调函数

        如果设置了once=True且已被调用过，则直接返回。
        执行前会严格验证参数，参数不匹配时抛出详细异常。

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Raises:
            TypeError: 当参数不匹配且swallow_exceptions=False时
        """
        if self.once and self._called:
            return

        try:
            # 严格验证参数
            call_args, call_kwargs = self._validate_arguments(args, kwargs)

            # 调用函数
            self.fn(*call_args, **call_kwargs)

        except Exception as e:
            if not self.swallow_exceptions:
                raise e
            LOGGER.error(f"[Callback Error] {str(e)}")

    def __lt__(self, other):
        """
        定义小于比较，用于优先级排序

        Args:
            other: 另一个CallbackWrapper实例

        Returns:
            bool: 当前实例的优先级是否高于另一个实例
        """
        return self.priority > other.priority


class Callback:
    """
    全局回调管理器

    管理所有事件的回调函数，支持优先级排序、单次执行、异常处理等功能。
    提供事件的注册、移除、执行等操作，包含严格的参数验证机制。

    特性:
        - 支持优先级管理 (priority)
        - 支持单次执行 (once)
        - 支持异常处理配置 (swallow_exceptions)
        - 严格的参数验证和详细的错误信息
        - 自动注册默认回调函数
        - 支持事件和字符串两种方式指定事件

    属性:
        _events: Dict[str, Event] - 所有事件对象的字典
        _callbacks: Dict[str, List[CallbackWrapper]] - 事件到回调包装器的映射

    方法:
        add_callback: 添加回调函数
        set_callback: 设置回调函数（覆盖模式）
        remove_callback: 移除回调函数
        run_callback: 执行指定事件的所有回调函数
        get_callback_prototype: 获取回调函数的原型
        list_callbacks_with_prototypes: 列出所有回调函数及其原型
        get_event_by_name: 通过事件名获取事件对象
        summary: 获取回调统计信息
    """

    def __init__(self):
        """初始化回调管理器，自动注册所有事件和默认回调"""

        # 自动收集所有Event实例
        self._events: Dict[str, Event] = {}
        for attr_name in dir(Events):
            if attr_name.startswith('__'):
                continue
            attr = getattr(Events, attr_name)
            if isinstance(attr, Event):
                self._events[attr_name] = attr

        # 初始化回调字典
        self._callbacks: Dict[str, List[CallbackWrapper]] = {}
        for event_obj in self._events.values():
            self._callbacks[event_obj.name] = []
            # 注册默认回调
            if event_obj.default_callback:
                self.add_callback(event_obj, event_obj.default_callback)

    def add_callback(self, event: Event | str, callback: Callable, priority: int = 0, once: bool = False, override: bool = False):
        """
        添加回调函数

        Args:
            event: 事件对象或事件名称
            callback: 回调函数
            priority: 优先级，默认为0
            once: 是否只执行一次，默认为False
            override: 是否覆盖现有回调，默认为False
        """
        if isinstance(event, str):
            event_name = event
        else:
            event_name = event.name

        wrapper = CallbackWrapper(callback, priority, once)
        if override:
            self._callbacks[event_name] = [wrapper]
        else:
            if event_name not in self._callbacks:
                self._callbacks[event_name] = []
            self._callbacks[event_name].append(wrapper)
            self._callbacks[event_name].sort()

    def set_callback(self, event: Event | str, callback: Callable):
        """
        设置回调函数（覆盖模式）

        清除该事件的所有现有回调，只保留新设置的回调。

        Args:
            event: 事件对象或事件名称
            callback: 回调函数
        """
        self.add_callback(event, callback, override=True)

    def remove_callback(self, event: Event | str, callback: Callable):
        """
        移除指定的回调函数

        Args:
            event: 事件对象或事件名称
            callback: 要移除的回调函数
        """
        event_name = event.name if isinstance(event, Event) else event
        if event_name in self._callbacks:
            self._callbacks[event_name] = [
                w for w in self._callbacks[event_name] if w.fn != callback
            ]

    def run_callback(self, event: Event | str, *args, **kwargs):
        """
        执行指定事件的所有回调函数

        按照优先级顺序执行回调函数，参数不匹配时会抛出详细异常。

        Args:
            event: 事件对象或事件名称
            *args: 传递给回调函数的位置参数
            **kwargs: 传递给回调函数的关键字参数

        Raises:
            TypeError: 当回调函数参数不匹配时
        """
        event_name = event.name if isinstance(event, Event) else event
        for wrapper in self._callbacks.get(event_name, []):
            wrapper(*args, **kwargs)

    def get_callback_prototype(self, event: Event | str, callback_index: int = 0) -> str:
        """
        获取指定回调函数的原型

        Args:
            event: 事件对象或事件名称
            callback_index: 回调函数索引，默认为0

        Returns:
            str: 回调函数的原型字符串，如果未找到则返回提示信息
        """
        event_name = event.name if isinstance(event, Event) else event
        wrappers = self._callbacks.get(event_name, [])
        if callback_index < len(wrappers):
            return wrappers[callback_index].prototype
        return "未找到对应的回调函数"

    def list_callbacks_with_prototypes(self, event: Event | str) -> List[Dict]:
        """
        列出指定事件的所有回调函数及其原型

        Args:
            event: 事件对象或事件名称

        Returns:
            List[Dict]: 包含回调函数信息的字典列表
        """
        event_name = event.name if isinstance(event, Event) else event
        result = []
        for i, wrapper in enumerate(self._callbacks.get(event_name, [])):
            result.append({
                'index': i,
                'function': wrapper.fn.__name__,
                'prototype': wrapper.prototype,
                'priority': wrapper.priority
            })
        return result

    def get_event_by_name(self, event_name: str) -> Event | None:
        """
        通过事件名获取事件对象

        Args:
            event_name: 事件名称

        Returns:
            Event | None: 事件对象，如果未找到则返回None
        """
        return next((e for e in self._events.values() if e.name == event_name), None)

    def summary(self):
        """
        获取回调统计信息

        Returns:
            Dict[str, int]: 事件名称到回调数量的映射（只包含有回调的事件）
        """
        return {e: len(cbs) for e, cbs in self._callbacks.items() if cbs}
