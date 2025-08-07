# ---------- 全局注册器 ----------
from __future__ import annotations

import torch.nn as nn
from typing import Dict, Iterable, Set, Type, List, Tuple, Any, Optional, ClassVar, Callable


class TTEngineRegistry:
    _cores: ClassVar[Set[str]] = set()
    _impl: ClassVar[Dict[str,  # core
    Dict[str,  # task
    Dict[str,  # engine_type
    Dict[str, Type]  # backend -> class
    ]]]] = {}

    # ---------- 1. 核心注册 ----------
    @classmethod
    def register_core(cls, core_name: str):
        if core_name in cls._cores:
            raise ValueError(f"Core {core_name} already registered")
        cls._cores.add(core_name)

    # ---------- 2. 类装饰器 ----------
    @classmethod
    def register(
            cls,
            core: Type,
            task: str,
            engine_type: str,
            backend: Optional[str] = None,
    ) -> Callable[[Type], Type]:
        core_name = core.__name__

        def decorator(impl_cls: Type) -> Type:
            if core_name not in cls._cores:
                cls.register_core(core_name)

            bucket = (
                cls._impl
                .setdefault(core_name, {})
                .setdefault(task, {})
                .setdefault(engine_type, {})
            )
            key = "" if backend is None else backend
            bucket[key] = impl_cls
            return impl_cls

        return decorator

    # ---------- 3. 统一 get ----------
    @classmethod
    def get(
            cls,
            config_manager,
            engine_type: str,
            backend: Optional[str] = None,
    ) -> Type:
        core_name = config_manager.register_name
        task = config_manager.core["task"]
        if core_name not in cls._cores:
            raise KeyError(f"Core {core_name} has not been registered")

        bucket = (
            cls._impl
            .get(core_name, {})
            .get(task, {})
            .get(engine_type, {})
        )
        key = "" if backend is None else backend
        try:
            return bucket[key]
        except KeyError:
            raise NotImplementedError(
                f"No implementation for core={core_name}, task={task}, "
                f"engine_type={engine_type}, backend={backend}"
            )


# class TTEngineRegistry:
#     """负责 task × engine_type × backend 的多级注册"""
#     _map: Dict[str, Dict[str, Dict[str, Type[Any]]]] = {}
#
#     @classmethod
#     def register(cls,
#                  task: str,
#                  engine_type: str,
#                  backend: str | None = None):
#         """
#         装饰器：
#         @Registry.register(task="detect", engine_type="exporter", backend="onnx")
#         class MyExporter(...): ...
#         若 engine_type 本身无 backend 概念（如 model/trainer），backend 传 None
#         """
#
#         def decorator(subcls: Type[Any]):
#             cls._map.setdefault(task, {}).setdefault(engine_type, {})
#             if backend is None:
#                 cls._map[task][engine_type] = subcls  # type: ignore[arg-type]
#             else:
#                 cls._map[task][engine_type][backend] = subcls
#             return subcls
#
#         return decorator
#
#     @classmethod
#     def get(cls,
#             task: str,
#             engine_type: str,
#             backend: str | None = None) -> Type[Any]:
#         """根据 task / engine_type / backend 取出注册类"""
#         try:
#             bucket = TTEngineRegistry._map[task][engine_type]
#             if backend is None:  # 如 model/trainer
#                 return bucket
#             return bucket[backend]  # 如 exporter/inference_server
#         except KeyError as e:
#             raise NotImplementedError(
#                 f"No implementation for task={task}, engine_type={engine_type}, backend={backend}"
#             ) from e


class TTModuleRegistry:
    """
    TTModuleRegistry
    ================
    模块级类注册器，支持：
    - 一个类注册多个别名（字符串）
    - 装饰器与函数式两种接口
    - 批量冲突检测，一次性报告所有冲突别名
    - 反向索引（类 → 已注册别名）

    Examples:
        >>> from torch import nn
        >>> from tinytrain.utils.register import TTModuleRegistry
        >>>
        >>> # 1) 装饰器方式：无参
        >>> @TTModuleRegistry.register
        ... class DWConv(nn.Module):
        ...     pass
        >>>
        >>> # 2) 装饰器方式：指定别名
        >>> @TTModuleRegistry.register("ResB", "ResBasic")
        ... class ResidualBlock(nn.Module):
        ...     pass
        >>>
        >>> # 3) 函数式方式
        >>> class MyLinear(nn.Module):
        ...     pass
        >>> TTModuleRegistry.register_name(MyLinear, "FC", "Linear1d")
        >>>
        >>> # 4) 查询
        >>> TTModuleRegistry.get("ResB") is ResidualBlock
        True
        >>> TTModuleRegistry.aliases_of(ResidualBlock)
        {'ResB', 'ResBasic'}
    """

    """别名 -> 类对象 的全局映射。"""
    MODULE_REGISTRY: Dict[str, Type[nn.Module]] = {}

    """类对象 -> 已注册别名 的反向映射。"""
    _CLASS2ALIASES: Dict[Type[nn.Module], Set[str]] = {}

    # ---------------- 装饰器 ----------------
    @classmethod
    def register(cls, name: str = None, *aliases: str):
        """
        装饰器，支持两种调用方式：
        1. 无参装饰器，用类名作为别名：
            @TTModuleRegistry.register
            class MyModule(nn.Module): ...

        2. 指定别名：
            @TTModuleRegistry.register("Alias1", "Alias2")
            class MyModule(nn.Module): ...

        Args:
            name(str, optional): 主别名。当装饰器无参调用时，此位置会收到类对象本身。
            *aliases(str): 额外别名列表。

        Returns:
            Type[nn.Module]: 被装饰的类本身，保证装饰器语义正确。
        """

        # 无参分支：@TTModuleRegistry.register
        if (
                isinstance(name, type)
                and issubclass(name, nn.Module)
        ):
            module_cls, aliases = name, (name.__name__,)
            cls._register_aliases(module_cls, aliases)
            return module_cls  # 立即返回类

        # 有参分支：@TTModuleRegistry.register(...)
        def _decorator(module_cls: type):
            # 如果 name 为 None，则把类名也加入别名列表
            final_aliases = (name,) + aliases if name is not None else aliases
            if not final_aliases:  # 极端情况：@register() 没给任何别名
                final_aliases = (module_cls.__name__,)
            cls._register_aliases(module_cls, final_aliases)  # type: ignore[arg-type]
            return module_cls

        return _decorator

    # ---------------- 函数式注册 ----------------
    @classmethod
    def register_name(cls, module_cls: Type[nn.Module], *aliases: str) -> None:
        """
        函数式接口，批量给类注册别名。

        Args:
            module_cls(Type[nn.Module]): 需要注册的 nn.Module 子类。
            *aliases(str): 要关联的别名列表。
        """
        cls._register_aliases(module_cls, aliases)

    # ---------------- 内部注册逻辑 ----------------
    @classmethod
    def _register_aliases(cls, module_cls: Type[nn.Module], aliases: Iterable[str]) -> None:
        """
        执行真正的注册逻辑，并检测别名冲突。

        Args:
            module_cls(Type[nn.Module]): 待注册的类。
            aliases(Iterable[str]): 要绑定的别名集合。

        Raises:
            TypeError: 如果 module_cls 不是 nn.Module 的子类。
            KeyError: 如果有任何一个别名已被其他类占用。
        """
        if not issubclass(module_cls, nn.Module):
            raise TypeError(f"{module_cls} is not a subclass of torch.nn.Module")

        # 1. 收集所有冲突
        conflicts: List[Tuple[str, Type[nn.Module]]] = []
        for alias in aliases:
            if alias in cls.MODULE_REGISTRY:
                conflicts.append((alias, cls.MODULE_REGISTRY[alias]))

        # 2. 如果冲突，一次性报错
        if conflicts:
            lines = ["Module alias conflict detected:"]
            for alias, existing_cls in conflicts:
                lines.append(
                    f"  alias='{alias}' "
                    f"already registered by {existing_cls.__module__}.{existing_cls.__name__}, "
                    f"but now being registered by {module_cls.__module__}.{module_cls.__name__}"
                )
            raise KeyError("\n".join(lines))

        # 3. 写入索引
        if module_cls not in cls._CLASS2ALIASES:
            cls._CLASS2ALIASES[module_cls] = set()
        for alias in aliases:
            cls.MODULE_REGISTRY[alias] = module_cls
            cls._CLASS2ALIASES[module_cls].add(alias)

    # ---------------- 查询 ----------------
    @classmethod
    def get(cls, alias: str) -> Type[nn.Module]:
        """
        根据别名获取对应的类。

        Args:
            alias(str): 已注册的别名。

        Returns:
            Type[nn.Module]: 对应的 nn.Module 子类。

        Raises:
            KeyError: 如果别名不存在。
        """
        if alias not in cls.MODULE_REGISTRY:
            raise KeyError(f"Module alias '{alias}' not found.")
        return cls.MODULE_REGISTRY[alias]

    @classmethod
    def aliases_of(cls, module_cls: Type[nn.Module]) -> Set[str]:
        """
        查询某个类已注册的所有别名。

        Args:
            module_cls(Type[nn.Module]): 目标类。

        Returns:
            Set[str]: 该类关联的所有别名集合，若未注册返回空集合。
        """
        return cls._CLASS2ALIASES.get(module_cls, set())

    @classmethod
    def clear(cls) -> None:
        """
        清空整个注册表，常用于单元测试或热重载。
        """
        cls.MODULE_REGISTRY.clear()
        cls._CLASS2ALIASES.clear()
