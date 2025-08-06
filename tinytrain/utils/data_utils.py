"""
轻量级缓存与文件 IO 工具箱
- 零依赖（除 cv2/numpy）
- 原子写入防损坏
- 支持 mmap 加载大图像
- 统一中文路径读取
"""

import gc
import os
import tempfile
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]

def add_prefix_to_path(img_file: Path, prefix:str="") -> Path:
    """将 img_file 的最后一级目录名作为前缀拼接到文件名前"""
    new_name = f"{prefix}_{img_file.name}"
    return img_file.parent / new_name

def remove_prefix_from_path(img_file: Path, prefix: str = "") -> Path:
    """移除文件名中的前缀，还原原始文件名"""
    name = img_file.name
    prefix_part = f"{prefix}_"
    if name.startswith(prefix_part):
        new_name = name[len(prefix_part):]      # 只去掉前缀以及紧随的下划线
    else:
        raise ValueError
    return img_file.parent / new_name

# ------------------------------------------------------------------
# 1. 上下文装饰器：临时关闭 GC，减少大数组加载时的内存峰值
# ------------------------------------------------------------------
def _gc_safe_load(load_fn):
    """
    装饰器：关闭垃圾回收后执行 IO，异常时务必恢复 GC，防止泄露。
    """
    def wrapper(*args, **kwargs):
        gc.disable()
        try:
            return load_fn(*args, **kwargs)
        finally:
            gc.enable()
    return wrapper

# ------------------------------------------------------------------
# 2. 私有安全加载函数
# ------------------------------------------------------------------
@_gc_safe_load
def _load_npy(path: PathLike, *, mmap_mode=None, allow_pickle=False):
    return np.load(str(path), mmap_mode=mmap_mode, allow_pickle=allow_pickle)

@_gc_safe_load
def _load_npy_dict(path: PathLike) -> Any:
    return np.load(str(path), allow_pickle=True).item()

def _atomic_save_npy(arr: np.ndarray, path: PathLike, allow_pickle=False):
    """
    原子保存：先写临时文件再重命名，避免并发/中断导致的文件损坏。
    """
    path = Path(path)
    tmp = Path(tempfile.mktemp(suffix=".npy", dir=path.parent))
    try:
        np.save(str(tmp), arr, allow_pickle=allow_pickle)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

# ------------------------------------------------------------------
# 3. 公开接口
# ------------------------------------------------------------------
from os import PathLike
import cv2
import numpy as np

def cv_imread(file_path: PathLike, flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    支持中文路径的 cv2.imread 封装，并可指定读取模式。

    Args:
        file_path : PathLike
            待读取图像文件路径，支持中文。
        flag : int, optional
            cv2 读取标志，默认 cv2.IMREAD_COLOR(BGR)。
            常用取值：
            - cv2.IMREAD_COLOR 或 1   -> BGR, 忽略透明通道
            - cv2.IMREAD_GRAYSCALE or 0 -> 单通道灰度
            - cv2.IMREAD_UNCHANGED or -1 -> 原图，包括 alpha 通道

    Returns:
        读取到的图像数组，形状与 dtype 由 cv2.imdecode 决定。
    """
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
    if img is None:
        raise IOError(f'Cannot read image: {file_path}')
    return img

def get_hash(paths):
    """
    对路径列表计算 SHA256 摘要（按路径排序，加入大小+mtime，顺序无关）。
    常用于缓存键生成。
    """
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        st = os.stat(p)
        h.update(f"{st.st_size}:{st.st_mtime_ns}:{p}".encode())
    return h.hexdigest()

# ------------------------------------------------------------------
# 4. 字典缓存
# ------------------------------------------------------------------
def save_dict_cache_file(file: PathLike, obj: Any, allow_pickle=False):
    """
    将 Python 对象原子写入 *.npy（pickle 模式）。
    """
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(obj, file, allow_pickle=allow_pickle)

def load_dict_cache_file(file: PathLike) -> Any:
    """
    从 *.npy 反序列化字典对象（pickle 模式）。
    """
    return _load_npy_dict(file)

# ------------------------------------------------------------------
# 5. 图像缓存
# ------------------------------------------------------------------
def save_image_cache_file(file: PathLike, img: np.ndarray):
    """
    将 uint8 图像数组原子写入 *.npy 缓存。
    """
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _atomic_save_npy(img, file)

def load_image_cache_file(file: PathLike, *, mmap: bool = True) -> np.ndarray:
    """
    从 *.npy 加载图像数组；mmap=True 时启用内存映射，节省 RAM。
    """
    mmap_mode = "r" if mmap else None
    return _load_npy(file, mmap_mode=mmap_mode)