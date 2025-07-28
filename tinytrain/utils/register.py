# ---------- 全局注册器 ----------
from typing import Dict, Type, Any


class TTRegistry:
    """负责 task × engine_type × backend 的多级注册"""
    _map: Dict[str, Dict[str, Dict[str, Type[Any]]]] = {}

    @classmethod
    def register(cls,
                 task: str,
                 engine_type: str,
                 backend: str | None = None):
        """
        装饰器：
        @Registry.register(task="detect", engine_type="exporter", backend="onnx")
        class MyExporter(...): ...
        若 engine_type 本身无 backend 概念（如 model/trainer），backend 传 None
        """
        def decorator(subcls: Type[Any]):
            cls._map.setdefault(task, {}).setdefault(engine_type, {})
            if backend is None:
                cls._map[task][engine_type] = subcls
            else:
                cls._map[task][engine_type][backend] = subcls
            return subcls
        return decorator

    @classmethod
    def get(cls,
            task: str,
            engine_type: str,
            backend: str | None = None) -> Type[Any]:
        """根据 task / engine_type / backend 取出注册类"""
        try:
            bucket = TTRegistry._map[task][engine_type]
            if backend is None:          # 如 model/trainer
                return bucket
            return bucket[backend]       # 如 exporter/inference_server
        except KeyError as e:
            raise NotImplementedError(
                f"No implementation for task={task}, engine_type={engine_type}, backend={backend}"
            ) from e