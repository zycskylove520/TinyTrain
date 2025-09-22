"""
通用工具函数合集
提供随机种子、分布式同步、目录生成、数值规整等最常用的小工具，零第三方依赖（除 PyTorch 外）。
"""

import math
import os
import random
import shutil
import uuid
import torch
import numpy as np
import torch.distributed as dist

from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Union, Sequence, TypeVar

from tinytrain.utils import LOGGER

T = TypeVar('T')


def make2tuple(x: Union[T, Sequence[T]]) -> Tuple[T, T]:
    """
    将标量或长度为 2 的序列统一成二元组。

    Args:
        x: 单个值或长度为 2 的 list/tuple

    Returns:
        Tuple[T, T]: 保证返回两个元素的元组
    """
    if isinstance(x, (tuple, list)):
        # 此时 x 只能是 tuple 或 list；长度不为 2 时让运行时抛 ValueError
        if len(x) != 2:
            raise ValueError("Expected exactly 2 elements")
        return x[0], x[1]
    return x, x


def make2list(x: Union[T, Sequence[T]]) -> list[T]:
    """
    将标量或长度为 2 的序列统一成二元列表。

    Args:
        x: 单个值或长度为 2 的 list/tuple

    Returns:
        list[T]: 保证返回两个元素的列表
    """
    if isinstance(x, (tuple, list)):
        # 此时 x 只能是 tuple 或 list；长度不为 2 时让运行时抛 ValueError
        if len(x) != 2:
            raise ValueError("Expected exactly 2 elements")
        return list(x)
    return [x, x]


def generate_unique_id(file_name, timestamp):
    """
    根据文件名和时间戳生成全局唯一 UUID（版本 5）。

    Args:
        file_name: 原始文件名
        timestamp: datetime 对象

    Returns:
        str: 32 位十六进制字符串
    """
    unique_str = f"{file_name}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"
    unique_id = uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=unique_str).hex
    return unique_id


def set_random_seed(seed: int = 0, deterministic: bool = False) -> None:
    """
    统一设置 Python / NumPy / PyTorch 的随机种子，支持 CUDA 确定性。

    Args:
        seed: 随机种子
        deterministic: 是否启用 CUDA 确定性算法（会牺牲速度）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用CUDA，额外设置CUDA的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

        # 启用CUDA的非确定性算法
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = deterministic
            if deterministic:
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    分布式训练上下文：让所有非 rank-0 进程等待 rank-0 完成特定任务后再继续。

    Args:
        local_rank: 当前进程 local rank
    """
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])


def create_iter_directory(base_dir, start_string="train_"):
    """
    在指定目录下创建带递增编号的新子目录。

    Args:
        base_dir: 基础目录
        start_string: 子目录前缀，默认为 "train_"

    Returns:
        Path: 新创建的目录路径
    """
    # 检查 base_dir 是否存在，如果不存在则创建
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 获取 base_dir 下的所有子目录
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 过滤出以 start_string 开头的目录，并提取编号
    project_dirs = [d for d in existing_dirs if d.startswith(start_string)]
    project_numbers = []
    for d in project_dirs:
        try:
            # 尝试提取编号部分并转换为整数
            number_part = d[len(start_string):]
            project_numbers.append(int(number_part))
        except ValueError:
            # 如果编号部分无法转换为整数，则忽略该目录
            continue

    # 确定新目录的编号
    if not project_numbers:
        # 如果没有找到任何以 start_string 开头的目录，则创建 start_string0
        new_project_number = 0
    else:
        # 如果有以 start_string 开头的目录，则创建下一个编号的目录
        new_project_number = max(project_numbers) + 1

    # 创建新目录
    new_project_dir = os.path.join(base_dir, f"{start_string}{new_project_number}")
    os.makedirs(new_project_dir)

    return Path(new_project_dir)


def make_divisible(x, divisor=8):
    """
    将输入整数向上取整为 divisor 的最近倍数，常用于网络输入尺寸对齐。

    Args:
        x: 待规整数值
        divisor: 除数，默认 8

    Returns:
        int: 规整后的数值
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def _get_free_shm_mb() -> float:
    """返回 /dev/shm 剩余空间（单位 MB）。"""
    shm_path = "/dev/shm"
    if not os.path.exists(shm_path):
        return float("inf")  # Windows / 特殊环境
    return shutil.disk_usage(shm_path).free / 1024 / 1024


def maybe_limit_num_workers(requested_workers: int, safe_threshold_mb: int = 2048) -> int:
    """
    根据 /dev/shm 剩余空间决定是否要降低 num_workers。
    如果可用共享内存 < safe_threshold_mb，就把 num_workers 降到 0。
    返回最终 num_workers。
    """
    free_mb = _get_free_shm_mb()
    if free_mb < safe_threshold_mb and requested_workers > 0:
        LOGGER.warning(
            f"Available shared memory ({free_mb:.0f} MB) < "
            f"{safe_threshold_mb} MB. "
            f"Forcing num_workers=0 to avoid Bus Error."
        )
        return 0
    return requested_workers
