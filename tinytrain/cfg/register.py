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

# ---------- 全局注册器 ----------
from __future__ import annotations

from pathlib import Path
from torch import nn
from typing import Dict, Iterable, Set, Type, List, Tuple, Optional, ClassVar, Callable, Union, Final

from tinytrain.global_var import ASSETS_PATH
from tinytrain.utils import LOGGER


class TTEngineRegistry:
    """
    TTEngineRegistry
    ================
    引擎级实现注册器，用于统一管理 **核心(core) × 任务(task) × 引擎类型(engine_type) × 后端(backend)** 的四维实现注册与查询。

    支持：
    - 装饰器注册
    - 显式注册核心
    - 按路径精确/模糊查询实现类
    - 批量冲突检测（注册阶段即报错）
    - 7大引擎类型：model、trainer、validator、predictor、exporter、tuner和distiller

    数据结构示意：
        _impl[core][task][engine_type][backend] = impl_cls
        当 backend 为 None 时，key 固定为 ""，实现“默认实现”语义。

    Examples:
        >>> from tinytrain.cfg.register import TTEngineRegistry
        >>>
        >>> # 1) 装饰器方式注册
        >>> @TTEngineRegistry.register(core=MyCore, task="classify", engine_type="model")
        ... class MyClassificationModel:
        ...     pass
        >>>
        >>> # 2) 在Core类中注册
        >>> class MyCore(TTBaseCore):
        >>>     @classmethod
        >>>     def register_components(cls):
        >>>         TTEngineRegistry.register(cls, "classify", "model")(MyClassificationModel)
        >>>
        >>> # 3) 指定后端
        >>> @TTEngineRegistry.register(core=MyCore, task="detect", engine_type="inference_server", backend="onnx")
        ... class MyClassificationOnnxInferenceServer:
        ...     pass
        >>>
        >>> # 4) 查询实现
        >>> cls = TTEngineRegistry.get(config_manager, engine_type="inference_server", backend="onnx")
        >>> assert cls is MyClassificationOnnxInferenceServer
    """

    # ENGINE_ASSERT = (
    #     "model",
    #     "trainer",
    #     "validator",
    #     "predictor",
    #     "exporter",
    #     "tuner",
    #     "distiller",
    #     "inference_server",
    #     "export_server",
    #     "track_server"
    # )

    """已注册的核心名称集合。"""
    _cores: ClassVar[Set[str]] = set()

    """
    多级嵌套实现仓库：
        _impl[core_name][task][engine_type][backend] -> 实现类
    """
    _impl: ClassVar[Dict[str,  # core
    Dict[str,  # task
    Dict[str,  # engine_type
    Dict[str, Type]  # backend -> class
    ]]]] = {}

    # ---------- 1. 核心注册 ----------
    @classmethod
    def register_core(cls, core_name: str):
        """
        显式注册一个核心名称，防止拼写错误或重复注册。

        Args:
            core_name(str): 核心类的名称（通常使用 core.__name__）。

        Raises:
            ValueError: 如果该核心已被注册。
        """
        if core_name in cls._cores:
            raise ValueError(f"TTBaseCore {core_name} already registered")
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
        """
        装饰器，用于把某个实现类注册到指定四维路径下。

        如果对应核心尚未注册，会自动调用 register_core 进行注册。

        Args:
            core(Type): 核心类（取其 __name__ 作为核心名称）。
            task(str): 任务名称，如 "classify"、"detect"、"pose"。
            engine_type(str): 引擎类型，如 "model"、"trainer"、"validator"。
            backend(str, optional): 后端名称，如 "tensorrt"、"onnx"、"ncnn"。
                若为 None 或空字符串，则视为默认实现，key 固定为 ""。

        Returns:
            Callable[[Type], Type]: 装饰器函数，返回被装饰的类本身。
        """
        # assert engine_type in TTEngineRegistry.ENGINE_ASSERT, f"engine_type {engine_type} is not supported, must be one of {TTEngineRegistry.ENGINE_ASSERT}"

        core_name = core.__name__

        def decorator(impl_cls: Type) -> Type:
            # 核心未注册时自动注册
            if core_name not in cls._cores:
                cls.register_core(core_name)

            # 构建多级字典
            bucket = (
                cls._impl
                .setdefault(core_name, {})
                .setdefault(task, {})
                .setdefault(engine_type, {})
            )
            key = "" if backend is None else backend

            # 冲突检测：同一路径只能注册一次
            if key in bucket:
                raise KeyError(
                    f"Implementation already registered at "
                    f"core={core_name}, task={task}, engine_type={engine_type}, backend={backend}"
                )
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
    ) -> Type | None:
        """
        根据配置管理器与引擎类型，精确查询实现类。

        Args:
            config_manager: 必须实现以下属性：
                - register_name -> 核心名称
                - core["task"]   -> 任务名称
            engine_type(str): 引擎类型。
            backend(str, optional): 后端名称，若为 None 则查询默认实现。

        Returns:
            Type: 查询到的实现类。

        Raises:
            KeyError: 核心未注册。
            NotImplementedError: 在指定四维路径下找不到实现。
        """
        core_name = config_manager.register_name
        task = config_manager.core["task"]
        if core_name not in cls._cores:
            raise KeyError(f"TTBaseCore {core_name} has not been registered")

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
            LOGGER.warning(
                f"No implementation for core={core_name}, task={task}, "
                f"engine_type={engine_type}, backend={backend}"
            )
            return None


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
        >>> from tinytrain import TTModuleRegistry
        >>>
        >>> # 1) 装饰器方式：无参
        >>> @TTModuleRegistry.register
        ... class DWCBA(nn.Module):
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

    # ------------------------------------------------------------------
    # 全局别名 -> 类对象 的正向映射表。
    # 所有通过 @register 或 register_name 注册的模块都会记录于此，
    # 用于运行期通过别名快速获取对应类。
    # ------------------------------------------------------------------
    MODULE_REGISTRY: Dict[str, Type[nn.Module]] = {}

    # ------------------------------------------------------------------
    # 类对象 -> 已注册别名集合 的反向映射表。
    # 方便根据类反查所有别名，支持 aliases_of() 等查询/调试功能。
    # ------------------------------------------------------------------
    _CLASS2ALIASES: Dict[Type[nn.Module], Set[str]] = {}

    # ------------------------------------------------------------------
    # 缓存文件路径：用于持久化 alias -> 全类名 的映射快照。
    # 启动时若文件存在则直接加载，避免重复扫描磁盘。
    # ------------------------------------------------------------------
    _CACHE_FILE: Final[Path] = ASSETS_PATH / "snapshots/registry_snapshot.py"

    # ------------------------------------------------------------------
    # 收集所有待扫描的 (root, exclude) 需求。
    # key   : 要扫描的顶级包/模块名，例如 "my_models"
    # value : 该 root 下需要排除的子包或子目录名集合。
    # 多次调用 register_plugin() 时会合并到此处，最终由 launch() 统一处理。
    # ------------------------------------------------------------------
    _ROOT_EXCLUDE: Dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # 防止无限重扫的守护标志。
    # 当 get() 发现别名不存在时会触发一次 launch() 重扫，随后立即置 True，
    # 同一进程生命周期内不再二次扫描，避免真正缺失的别名陷入死循环。
    # ------------------------------------------------------------------
    _RESCAN_GUARD: bool = False

    # ---------------- 装饰器 ----------------
    @classmethod
    def register(cls, name: str = None, *aliases: str):
        """
        装饰器，支持两种调用方式：
        1. 无参装饰器，用类名作为别名：
            @TTModuleRegistry.register
            class MyModule(nn.Module): ...

        2. 指定别名：
            @TTModuleRegistry.register("Alias1")
            class MyModule(nn.Module): ...

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
    def register_name(cls, module_cls: Type[nn.Module], name: str = None, *aliases: str) -> None:
        """
        函数式接口，批量给类注册别名。

        Args:
            module_cls(Type[nn.Module]): 需要注册的 nn.Module 子类。
            name(str, optional): 主别名。当装饰器无参调用时，此位置会收到类对象本身。
            *aliases(str): 要关联的别名列表。
        """
        final_aliases = (name,) + aliases if name is not None else aliases
        if not final_aliases:  # 极端情况：没给任何别名
            final_aliases = (module_cls.__name__,)
        cls._register_aliases(module_cls, final_aliases)

    # ---------------- 内部注册逻辑 ----------------
    @classmethod
    def _register_aliases(cls, module_cls: Type[nn.Module], aliases: Iterable[str]) -> None:
        """
        真正的注册逻辑 + 即时落盘缓存。
        每成功注册一个别名，就把  alias -> 全类名  追加写入快照文件，
        保证后续进程可以直接秒加载。

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

        # 3. 写入内存
        for alias in aliases:
            cls.MODULE_REGISTRY[alias] = module_cls
            cls._CLASS2ALIASES.setdefault(module_cls, set()).add(alias)

    # ---------------- 查询 ----------------
    @classmethod
    def get(cls, alias: str) -> Type[nn.Module]:
        """
        根据别名获取类。若别名不存在且尚未触发过重扫，则自动重新扫盘一次；
        重扫后仍找不到才抛 KeyError。

        Args:
            alias(str): 已注册的别名。

        Returns:
            Type[nn.Module]: 对应的 nn.Module 子类。

        Raises:
            KeyError: 如果别名不存在。
        """
        # 1. 内存命中直接返回
        if alias in cls.MODULE_REGISTRY:
            return cls.MODULE_REGISTRY[alias]

        # 2. 未命中，且还没触发过重扫 → 扫一次
        if not cls._RESCAN_GUARD:
            cls._RESCAN_GUARD = True  # 置位，保证只扫一次
            LOGGER.info("Alias '%s' not found in cache, rescanning disk once...", alias)
            cls.launch()  # 复用之前统一入口
            # 再次查询
            if alias in cls.MODULE_REGISTRY:
                return cls.MODULE_REGISTRY[alias]

        # 3. 仍然找不到 → 抛错
        raise ValueError(
            f"Unrecognized module string '{alias}'. "
            f"Please check spelling, add candidate package, or use @register_module."
        )

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

    @classmethod
    def register_plugin(cls, root: str = "tinytrain", exclude: Union[str, Iterable[str], None] = None) -> None:
        """
        仅把本次扫描需求登记到 _ROOT_EXCLUDE，不真正扫描。
        可以多次调用，最终由 launch() 统一执行。

        用法：
            TTModuleRegistry.register_plugin("my_models.blocks")  # 整包
            TTModuleRegistry.register_plugin("my_models.blocks.block")   # 单模块
            TTModuleRegistry.register_plugin("my_models.blocks.block.block.py")   # 单文件
        """
        exclude_set = {exclude} if isinstance(exclude, str) else set(exclude or [])
        # 合并多次调用对同一个 root 的排除项
        cls._ROOT_EXCLUDE.setdefault(root, set()).update(exclude_set)

    @classmethod
    def launch(cls) -> None:
        """
        汇总之前所有 register_plugin 的登记，
        一次性扫描、导入、写缓存。
        """
        import importlib, pkgutil, os
        cls.clear()

        # ---------- 缓存命中则直接加载 ----------
        cls._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls._RESCAN_GUARD and cls._CACHE_FILE.exists():
            try:
                cache_module_name = ".".join(
                    cls._CACHE_FILE.with_suffix("").parts[-3:]
                )
                spec = importlib.util.spec_from_file_location(
                    cache_module_name, cls._CACHE_FILE
                )
                if spec is None or spec.loader is None:
                    raise ImportError("bad snapshot spec")
                snap_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(snap_mod)

                for alias, full_path in snap_mod.REG.items():
                    mod_path, cls_name = full_path.rsplit(".", 1)
                    mod = importlib.import_module(mod_path)
                    module_cls = getattr(mod, cls_name)
                    cls.MODULE_REGISTRY[alias] = module_cls
                    cls._CLASS2ALIASES.setdefault(module_cls, set()).add(alias)
                cls.write_cache()
                LOGGER.info(f"Loaded module registry from snapshot: {cls._CACHE_FILE}.")
                return
            except Exception as e:
                LOGGER.warning(f"Snapshot broken ({e}), rescanning...")
                cls._CACHE_FILE.unlink(missing_ok=True)

        # ---------- 缓存未命中：现场扫描 ----------
        # 对登记过的 (root,exclude) 去重、排序，保证可复现
        for root, exclude_set in sorted(cls._ROOT_EXCLUDE.items()):
            root_mod = importlib.import_module(root)
            root_dir = os.path.dirname(root_mod.__file__ or root_mod.__path__[0])

            # 排除目录不存在直接抛错，方便早发现
            missing = {d for d in exclude_set if not os.path.isdir(os.path.join(root_dir, d))}
            if missing:
                raise FileNotFoundError(
                    f"Excluded folder(s) {missing} not found in root directory '{root_dir}'."
                )

            exclude_prefixes = {f"{root}.{d}" for d in exclude_set}

            # 导入子模块触发装饰器
            if hasattr(root_mod, "__path__"):
                for _, subname, _ in pkgutil.walk_packages(
                        root_mod.__path__, root_mod.__name__ + "."
                ):
                    if any(subname == pfx or subname.startswith(pfx + ".") for pfx in exclude_prefixes):
                        continue
                    importlib.import_module(subname)
            else:
                importlib.import_module(root)

            cls.write_cache()
            LOGGER.info(f"Registry snapshot saved to: {cls._CACHE_FILE}")

    @classmethod
    def write_cache(cls):
        reg_lines = ["# Auto-generated by TTModuleRegistry.launch\nREG = {"]
        for alias, module_cls in sorted(cls.MODULE_REGISTRY.items()):
            full = f"{module_cls.__module__}.{module_cls.__name__}"
            reg_lines.append(f'    "{alias}": "{full}",')
        reg_lines.append("}")
        cls._CACHE_FILE.write_text("\n".join(reg_lines), encoding="utf8")
