import gc
import os
import tempfile
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]

# ---------- 工具 ----------
def _gc_safe_load(load_fn):
    """装饰器：在关闭 GC 的情况下执行 load_fn，并保证异常时恢复 GC"""
    def wrapper(*args, **kwargs):
        gc.disable()
        try:
            return load_fn(*args, **kwargs)
        finally:
            gc.enable()
    return wrapper

# ---------- I/O ----------
@_gc_safe_load
def _load_npy(path: PathLike, *, mmap_mode=None, allow_pickle=False):
    return np.load(str(path), mmap_mode=mmap_mode, allow_pickle=allow_pickle)

@_gc_safe_load
def _load_npy_dict(path: PathLike) -> Any:
    return np.load(str(path), allow_pickle=True).item()

def _atomic_save_npy(arr: np.ndarray, path: PathLike, allow_pickle=False):
    """原子写入：先写临时文件再 rename，防止并发/中断导致损坏"""
    path = Path(path)
    tmp = Path(tempfile.mktemp(suffix=".npy", dir=path.parent))
    try:
        np.save(str(tmp), arr, allow_pickle=allow_pickle)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

# ---------- 公开接口 ----------
def cv_imread(file_path: PathLike) -> np.ndarray:
    """支持中文路径读取"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)

def get_hash(paths):
    """返回路径列表的 SHA256。加入 size+mtime，顺序无关。"""
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        st = os.stat(p)
        h.update(f"{st.st_size}:{st.st_mtime_ns}:{p}".encode())
    return h.hexdigest()

# ---------- dict cache ----------
def save_dict_cache_file(file: PathLike, obj: Any, allow_pickle=False):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(obj, file, allow_pickle=allow_pickle)

def load_dict_cache_file(file: PathLike) -> Any:
    return _load_npy_dict(file)

# ---------- image cache ----------
def save_image_cache_file(file: PathLike, img: np.ndarray):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _atomic_save_npy(img, file)

def load_image_cache_file(file: PathLike, *, mmap: bool = True) -> np.ndarray:
    mmap_mode = "r" if mmap else None
    return _load_npy(file, mmap_mode=mmap_mode)