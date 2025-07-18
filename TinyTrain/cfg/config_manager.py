from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from TinyTrain.utils.checks import check_file


class ConfigManager(SimpleNamespace):
    def __init__(self, link_file: str | Path, **kwargs):
        super().__init__(link_file=link_file, **kwargs)
        self.link = None
        self.parse_link_file(link_file)

    def parse_link_file(self, link_file: str | Path):
        # 检查文件是否存在
        link_file = check_file(link_file)

        if link_file.suffix == ".toml":
            self.link = self.load_toml(link_file)
            for key, value in self.link.items():
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a valid file")
                value = check_file(value)
                if value.suffix != ".toml":
                    raise ValueError(f"{key} must be '.toml' file")
                data = self.load_toml(value)
                setattr(self, key, data)
        else:
            raise ValueError(f"Unsupported file type: {link_file.suffix}")

    @staticmethod
    def load_toml(toml_file: str | Path) -> Dict:
        """
        加载 TOML 文件
        :param toml_file: TOML 文件路径
        :return: 加载后的数据
        """
        import toml
        return toml.load(toml_file)

    def __iter__(self):
        return iter(vars(self).items())

    def __deepcopy__(self, memo):
        # 创建一个新的 ConfigManager 实例，并传递 link_file 参数
        new_instance = ConfigManager(self.link_file)
        memo[id(self)] = new_instance

        # 显式地复制所有属性
        for key, value in self.__dict__.items():
            if key != "link_file":  # link_file 已经在构造函数中处理
                setattr(new_instance, key, deepcopy(value, memo))

        return new_instance

    def __reduce__(self):
        # 返回一个元组，包含用于重新构建对象的类和参数
        return self.__class__, (self.link_file,), self.__dict__
