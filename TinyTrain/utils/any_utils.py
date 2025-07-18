import math
import os
import random
import uuid
import torch
import numpy as np
import torch.distributed as dist

from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Union, Sequence, TypeVar

T = TypeVar('T')

def make2tuple(x: Union[T, Sequence[T]]) -> Tuple[T, T]:
    if isinstance(x, (tuple, list)):
        # 此时 x 只能是 tuple 或 list；长度不为 2 时让运行时抛 ValueError
        if len(x) != 2:
            raise ValueError("Expected exactly 2 elements")
        return x[0], x[1]
    return x, x


def make2list(x: Union[T, Sequence[T]]) -> list[T]:
    if isinstance(x, (tuple, list)):
        # 此时 x 只能是 tuple 或 list；长度不为 2 时让运行时抛 ValueError
        if len(x) != 2:
            raise ValueError("Expected exactly 2 elements")
        return list(x)
    return [x, x]


def generate_unique_id(file_name, timestamp):
    """
    使用文件名和时间戳生成UUID。
    这样可以确保即使文件名相同，不同时间生成的UUID也不同。
    """
    unique_str = f"{file_name}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"
    unique_id = uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=unique_str).hex
    return unique_id


def set_random_seed(seed: int = 0, deterministic: bool = False) -> None:
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
            # 启用 CUDNN（如果可用）
            torch.backends.cudnn.benchmark = True


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensures all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])


def create_train_directory(base_dir, start_string="train_"):
    """
    在指定的基础目录下创建新的训练目录。
    如果基础目录下没有以 start_string 开头的目录，则创建 start_string0。
    如果有以 start_string 开头的目录，则创建下一个编号的目录。

    :param base_dir: 基础目录路径
    :param start_string: 目录的起始字符串，默认为 "train_"
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
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor
