from __future__ import annotations

import sys
import time
import random
import threading

from typing import Iterable, Optional

from TinyTrain.global_var import RANK

# -----------------------------------------------------------
# 全局配置（按需修改）
# -----------------------------------------------------------
# 全局锁：多线程场景下保证 stdout 串行输出
_LOCK = threading.Lock()
# 线程本地存储：用来记录当前线程内的嵌套深度，实现多线程嵌套进度条
_LOCAL = threading.local()

# ANSI 颜色字典（只读，全局缓存）——避免每次重新生成
RESET = "\033[0m"
_COLORS = {
    "black": "\033[30m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[37m",
    "bright_red": "\033[91m", "bright_green": "\033[92m", "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m", "bright_magenta": "\033[95m", "bright_cyan": "\033[96m",
}
# 预生成 (start, end) 元组，减少运行时字符串拼接
_COLOR_CODES = {k: (v, RESET) for k, v in _COLORS.items()}
# 预生成颜色值元组，random_color 模式下随机挑选
_COLOR_VALUES = tuple(_COLORS.values())

# -----------------------------------------------------------
# 模块级开关：是否全局禁用颜色
# -----------------------------------------------------------
_DISABLE_COLOR = False  # 会被 TTProgressBar 的实例参数覆盖


# -----------------------------------------------------------
# 工具函数
# -----------------------------------------------------------
def _color(text: str, color: str) -> str:
    """
    给文本添加 ANSI 颜色码，若颜色不存在或全局禁用颜色则原样返回。
    """
    if _DISABLE_COLOR:
        return text
    start, end = _COLOR_CODES.get(color, ("", ""))
    return f"{start}{text}{end}"


def _format_time(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS。
    """
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


class TTProgressBar:
    """
    TinyTrain ProgressBar 是高性能、彩色、线程安全、可嵌套进度条，API 与 tqdm 类似。
    支持自适应刷新间隔、批量更新、单线程 fast-path 无锁输出。

    已做 CPU 级微优化：
    1. 缓存不变字符串、颜色码
    2. 支持 update_bulk 批更新
    3. 自适应 min_interval
    4. 单线程 fast-path 无锁
    5. 其他细节：% 格式化、预生成颜色表

    参数
    ----
    iterable : Iterable, optional
        需要迭代的对象。若提供，可直接在 for 循环中使用；否则需手动调用 update。
    total : int, optional
        总步数。当 `iterable` 本身没有 `__len__` 时必须显式指定。
    desc : str, default ""
        进度条左侧的描述文字，支持运行时通过 `set_description` 修改。
    title : str, optional
        额外标题，仅在第一次渲染时打印一次，可用于区分多个进度条。
    bar_width : int, default 20
        进度条宽度（字符数）。
    info_color : str, default "cyan"
        描述文字的颜色，必须是 _COLORS 中的键名。
    bar_color : str, default "green"
        进度条填充块的颜色，必须是 _COLORS 中的键名。
    fill_char : str, default "█"
        进度填充字符。
    show_percent : bool, default False
        是否显示百分比进度。
    random_color : bool, default False
        若设为 True，每次渲染时随机挑选颜色填充进度条。
    min_interval : float, default 0.05
        最小刷新间隔（秒）。低于此间隔的更新将被跳过。
    adaptive_interval : bool, default True
        是否启用自适应刷新。启用后会根据实际输出速率动态调整 `min_interval`，
        以控制终端输出频率在 ~30 行/秒左右，减少 CPU 占用。
    disable : bool, default False
        若设为 True，则完全静默，不打印任何进度条/标题/换行。（新增）
    disable_color : bool, default False
        若设为 True，则禁用所有 ANSI 颜色转义码，输出纯文本。（新增）
    """

    def __init__(
            self,
            iterable: Optional[Iterable] = None,
            total: Optional[int] = None,
            desc: str = "",
            title: Optional[str] = None,
            bar_width: int = 20,
            info_color: str = "cyan",
            bar_color: str = "green",
            fill_char: str = "█",
            show_percent: bool = False,
            random_color: bool = False,
            min_interval: float = 0.05,
            *,
            adaptive_interval: bool = True,
            disable: bool = False,
            disable_color: bool = False,
    ):
        # 新增：在非主进程中直接退出初始化，避免后续逻辑报错
        if RANK not in {-1, 0}:
            self._passthrough = True
            self.iterable = iterable
            return
        else:
            self._passthrough = False

        self.iterable = iterable
        # 若用户未提供 total，则尝试用 len(iterable) 自动推导
        self.total = (
            total
            if total is not None
            else (len(iterable) if hasattr(iterable, "__len__") else None)
        )
        if self.total is None:
            raise ValueError("total must be specified when iterable has no __len__")

        self.desc = desc
        self.title = title
        self.bar_width = bar_width
        self.info_color = info_color
        self.bar_color = bar_color
        self.fill_char = str(fill_char) or "█"
        self.show_percent = show_percent
        self.random_color = random_color
        self.min_interval = float(min_interval)
        self.adaptive_interval = bool(adaptive_interval)

        # 新增开关
        self.disable = bool(disable)
        self.disable_color = bool(disable_color)

        # 【新增】根据实例级别决定是否关闭颜色
        global _DISABLE_COLOR
        _DISABLE_COLOR = self.disable_color

        # 预先生成不变的前缀，减少每次渲染的字符串拼接
        self._info_prefix = _color(desc, info_color) + " "
        self._bar_left = "|"
        self._bar_right = "| "

        # 内部状态
        self._n = 0  # 当前已完成步数
        self._start_ts = time.perf_counter()
        self._last_print_ts = 0.0  # 上一次打印时间
        self._closed = False
        self._indent = " " * self._depth()  # 根据嵌套深度自动缩进

        self._header_printed = False  # 防止标题重复打印

        # 自适应刷新：目标 30 行/秒
        self._target_lines_per_sec = 30.0

        self._am_i_printing = RANK in {-1, 0}
        # 多进程/多机不能再用无锁路径
        self._fast = False

    # ---------- 上下文管理器 ----------
    def __enter__(self):
        """
        with 语句入口：
        1. 打印标题（如果需要）
        2. 自动增加嵌套深度
        3. 返回 self，以便用户手动 update 或继续 for 迭代
        """
        if not self.disable and self._am_i_printing:
            self._print_title()
        self._inc_depth(1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        with 语句出口：
        无论是否异常，都确保 close()，并减少嵌套深度
        """
        try:
            self.close()
        finally:
            self._inc_depth(-1)

    # ---------- 嵌套深度 ----------
    @staticmethod
    def _depth() -> int:
        """返回当前线程的嵌套深度（用于缩进）"""
        return getattr(_LOCAL, "tt_pbar_depth", 0)

    @staticmethod
    def _inc_depth(delta: int):
        """增减嵌套深度计数器"""
        _LOCAL.tt_pbar_depth = TTProgressBar._depth() + delta

    # ---------- 迭代器 ----------
    def __iter__(self):
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接透传
            yield from self.iterable
            return

        # 若禁用，则直接空转
        if self.disable:
            for _ in self.iterable:
                pass
            return

        # 非打印 rank：空转一遍，保持同步
        if not self._am_i_printing:
            for _ in self.iterable:
                pass
            return  # 结束迭代器

        self._inc_depth(1)
        try:
            for idx, item in enumerate(self.iterable, 0):  # 从 0 开始索引
                yield item
                self.update(idx + 1 - self._n)  # 更新进度条
        finally:
            self.close()
            self._inc_depth(-1)

    # ---------- 更新 ----------
    def update(self, n: int = 1):
        """
        单步更新，若间隔小于 min_interval 或 disable=True 则跳过渲染。
        """
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return

        if self.disable:
            self._n += n
            return
        self._n += n
        now = time.perf_counter()
        if now - self._last_print_ts < self.min_interval and self._n < self.total:
            return
        self._last_print_ts = now
        self._render()

    def update_bulk(self, n: int):
        """
        一次性累加 n 步，仅渲染一次，适合批量操作场景。
        若 disable=True 则直接跳过。
        """
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return

        if self.disable:
            self._n += n
            return
        self._n += n
        now = time.perf_counter()
        if now - self._last_print_ts >= self.min_interval or self._n >= self.total:
            self._last_print_ts = now
            self._render()

    # ---------- 渲染 ----------
    def _render(self):
        """
        核心渲染逻辑：计算百分比、ETA、速率，拼出整行并输出。
        若 disable=True 则直接返回。
        """
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return

        if self.disable:
            return

        elapsed = time.perf_counter() - self._start_ts
        rate = self._n / elapsed if elapsed else 0.0
        eta = 0.0 if self.total == 0 else (self.total - self._n) / (rate + 1e-9)

        pct = 100.0 if self.total == 0 else 100.0 * self._n / self.total
        fill_len = min(self.bar_width, max(0, int(round(self.bar_width * pct / 100.0))))

        # 始终通过 _COLOR_CODES 获取颜色码，受 disable_color 控制
        # 当 disable_color=True 时，start_color 与 end_color 均为空字符串
        if self.random_color:
            start_color, end_color = (
                (random.choice(_COLOR_VALUES), RESET)
                if not self.disable_color
                else ("", "")
            )
        else:
            start_color, end_color = (
                _COLOR_CODES.get(self.bar_color, ("", ""))
                if not self.disable_color
                else ("", "")
            )

        # 第一次渲染时才打印标题，确保无论 for/with/手动更新都只出现一次
        if not self._header_printed and self.title and not self.disable:
            self._print_title()
            self._header_printed = True

        # 构造单行
        percent_part = f"{pct:5.1f}%" if self.show_percent else ""
        info_text = f"{self._indent}{self._info_prefix}{percent_part}{self._bar_left}"

        # 使用 start_color / end_color 包裹填充字符，保证 disable_color=True 时无色
        filled = self.fill_char * fill_len
        empty = "-" * (self.bar_width - fill_len)
        bar_part = f"{start_color}{filled}{end_color}{empty}{self._bar_right}"

        tail = f"{self._n}/{self.total} [{_format_time(elapsed)}<{_format_time(eta)}, {rate:.2f}it/s]"
        line = info_text + bar_part + tail

        # 仅 rank 0 / -1 真正输出到终端
        if self._am_i_printing:
            with _LOCK:
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()

        # 自适应刷新间隔：根据实际速率动态调整 min_interval
        if self.adaptive_interval and self._n:
            lines_per_sec = 1.0 / max(elapsed, 1e-6)
            self.min_interval = max(0.01, 1.0 / max(lines_per_sec, self._target_lines_per_sec))

    # ---------- 其他 ----------
    def _print_title(self):
        """打印标题（仅一次），仅 rank 0 / -1 真正输出。若 disable=True 则跳过。"""
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return

        if self.disable or not self.title or not self._am_i_printing:
            return
        with _LOCK:
            print(_color(self.title, self.info_color))
            # 【新增】立即标记标题已打印，防止 _render 再次打印
            self._header_printed = True

    def close(self):
        """
        手动关闭进度条：最后一次渲染并换行。
        若已关闭或非主进程或 disable=True 则不操作。
        """
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return

        if self._closed or RANK not in {-1, 0} or self.disable:
            return
        self._closed = True
        self._render()  # 最后一次渲染
        if self._am_i_printing:
            with _LOCK:
                sys.stdout.write("\n")
                sys.stdout.flush()

    # ---------- 兼容 API ----------
    def set_description(self, desc: str):
        """运行时修改描述文字"""
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return
        self.desc = desc
        self._info_prefix = _color(desc, self.info_color) + " "

    def set_title(self, title: Optional[str]):
        """运行时修改标题"""
        if getattr(self, '_passthrough', False):  # 如果是“空壳”模式，直接返回
            return
        self.title = title

    @staticmethod
    def print(text):
        print(text)
