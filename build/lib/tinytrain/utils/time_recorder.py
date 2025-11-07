import time
import torch

from typing import Optional, Union

from tinytrain.global_var import TIMER_ENABLED
from tinytrain.utils import LOGGER


class TimeRecorder:
    """
    通用 CPU/GPU 时间计时器
    用法：
        with TimeRecorder(device, event_name="preprocess"):
            ...  # 待测代码
    """
    __slots__ = ("device", "event_name", "cpu_start", "cuda_event_start",
                 "cuda_event_end", "elapsed_ms", "_sync_before", "_disabled")

    def __init__(self,
                 device: Optional[Union[torch.device, str]] = None,
                 event_name: str = "",
                 sync_before: bool = True):
        """
        初始化计时器。

        Args:
            device : torch.device 或 str，若为 cuda 设备则使用 cudaEvent 计时，否则用 time.perf_counter()
            event_name : 仅用于日志/调试
            sync_before : 是否在进入上下文时同步设备，避免之前的异步 kernel 影响计时
        """
        self.device = torch.device(device) if device else torch.device("cpu")
        self.event_name = event_name
        self._sync_before = sync_before
        self._disabled = not TIMER_ENABLED
        self.cpu_start = 0.0
        self.cuda_event_start: Optional[torch.cuda.Event] = None
        self.cuda_event_end: Optional[torch.cuda.Event] = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self._disabled:
            return self

        if self.device.type == "cuda":
            if self._sync_before:
                torch.cuda.synchronize(self.device)
            self.cuda_event_start = torch.cuda.Event(enable_timing=True)
            self.cuda_event_end = torch.cuda.Event(enable_timing=True)
            self.cuda_event_start.record()
        else:
            self.cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disabled:
            return

        if self.device.type == "cuda":
            self.cuda_event_end.record()
            torch.cuda.synchronize(self.device)
            self.elapsed_ms = self.cuda_event_start.elapsed_time(self.cuda_event_end)  # ms
        else:
            self.elapsed_ms = (time.perf_counter() - self.cpu_start) * 1000.0  # ms → ms
        LOGGER.info(f"[Timer] {self.event_name} -> {self.elapsed_ms:.3f} ms")
