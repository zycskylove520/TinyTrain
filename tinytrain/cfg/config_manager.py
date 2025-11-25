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

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from tinytrain import ROOT
from tinytrain.utils.checks import check_file


class TTConfigManager(SimpleNamespace):
    """
    配置管理器，用于加载和解析配置文件。

    主要功能：
    1. 支持链式配置文件（通过 link 文件加载其他配置文件）。
    2. 提供便捷的属性访问方式。
    3. 支持深度拷贝和序列化。

    Args:
        link_file (str | Path): 链式配置文件路径。
        **kwargs: 其他初始化参数。

    示例：
        >>> config = TTConfigManager("config.toml")
        >>> print(config.model)
        >>> print(config.dataset)
    """

    def __init__(self, link_file: str | Path, **kwargs):
        """
        初始化 TTConfigManager 实例。

        Args:
            link_file (str | Path): 链式配置文件路径。
            **kwargs: 其他初始化参数。
        """
        super().__init__(link_file=link_file, **kwargs)
        self.link = None

        self.register_name = self._infer_register_name()
        self.parse_link_file(link_file)

    def parse_link_file(self, link_file: str | Path):
        """
        解析链式配置文件。

        Args:
            link_file (str | Path): 链式配置文件路径。

        Raises:
            ValueError: 如果文件类型不支持或文件内容无效。
        """

        # 检查文件是否存在
        link_file = check_file(link_file)

        self.link = self.load_config_file(link_file)
        # 检查是否提供了core.toml
        if "core" not in self.link:
            data = self.load_config_file(ROOT / "cfg/core.toml")
            setattr(self, "core", data)

        # 不允许与已存在成员冲突
        for key in {"link", "register_name"}:
            if key in self.link:
                raise ValueError(f" key: '{key}' cannot use in link file")

        for key, value in self.link.items():
            if not isinstance(value, str):
                raise ValueError(f"key: '{key}' must be a file path")

            value = check_file(value)
            self.link[key] = value
            data = self.load_config_file(value)
            setattr(self, key, data)

    def load_config_file(self, config_file: str | Path):
        """
        加载配置文件并根据文件类型调用相应的解析器。

        支持的文件格式：
        - .toml: 使用 toml 库解析
        - .yaml/.yml: 使用 yaml 库解析

        Args:
            config_file (str | Path): 配置文件路径，支持字符串或 Path 对象

        Returns:
            Dict: 解析后的配置数据字典

        Raises:
            ValueError: 当文件类型不支持时抛出异常
            FileNotFoundError: 当配置文件不存在时抛出异常
            toml.TomlDecodeError: 当 TOML 文件格式错误时抛出异常
            yaml.YAMLError: 当 YAML 文件格式错误时抛出异常
        """
        if Path(config_file).suffix == ".toml":
            return self.load_toml(config_file)
        elif Path(config_file).suffix == ".yaml" or Path(config_file).suffix == ".yml":
            return self.load_yaml(config_file)
        else:
            raise ValueError(f"TTConfigManager unsupported file type: {config_file.suffix}")

    def rebuild_link_type(self, **kwargs):
        """
        重新构建link文件内部指向文件。
        """
        for link_file_name, link_file_path in kwargs.items():
            self.link[link_file_name] = check_file(link_file_path)
            data = self.load_config_file(link_file_path)
            setattr(self, link_file_name, data)

    @staticmethod
    def load_toml(toml_file: str | Path) -> Dict:
        """
        加载 TOML 文件。

        Args:
            toml_file (str | Path): TOML 文件路径。

        Returns:
            Dict: 加载后的数据。

        Raises:
            FileNotFoundError: 如果文件不存在。
            toml.TomlDecodeError: 如果文件格式错误。
        """
        import toml
        try:
            return toml.load(toml_file)
        except toml.TomlDecodeError as e:
            raise ValueError(f"Error loading TOML file {toml_file}: {e}")

    @staticmethod
    def load_yaml(yaml_file: str | Path) -> Dict:
        """
        加载 YAML 文件。

        Args:
            yaml_file (str | Path): YAML 文件路径。

        Returns:
            Dict: 加载后的数据。
        """
        import yaml
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    @staticmethod
    def _infer_register_name() -> str | None:
        """
        自动推断调用者类名（例如 TTBaseCore 的子类）。
        如果无法推断，返回 None。
        """
        import inspect

        frame = inspect.currentframe()
        try:
            # 向上查找调用栈，跳过 TTConfigManager 自身的 __init__
            while frame:
                locals_dict = frame.f_locals
                self_arg = locals_dict.get('self')
                if self_arg and not isinstance(self_arg, TTConfigManager):
                    return self_arg.__class__.__name__
                frame = frame.f_back
        finally:
            del frame  # 避免引用泄漏
        return None

    def __iter__(self):
        """
        返回配置管理器的迭代器。
        """
        return iter(vars(self).items())

    def __deepcopy__(self, memo):
        """
        实现深度拷贝。

        Args:
            memo: 深度拷贝的备忘录。

        Returns:
            TTConfigManager: 深度拷贝后的实例。
        """
        # 创建一个新的 TTConfigManager 实例，并传递 link_file 参数
        new_instance = TTConfigManager(self.link_file)
        memo[id(self)] = new_instance

        # 显式地复制所有属性
        for key, value in self.__dict__.items():
            if key != "link_file":  # link_file 已经在构造函数中处理
                setattr(new_instance, key, deepcopy(value, memo))

        return new_instance

    def __reduce__(self):
        """
        返回用于序列化的元组。
        """
        # 返回一个元组，包含用于重新构建对象的类和参数
        return self.__class__, (self.link_file,), self.__dict__
