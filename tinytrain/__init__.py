from .global_var import *
from .cfg import TTModuleRegistry, TTEngineRegistry

__version__ = "0.2.4"

# 自动扫描注册算子
TTModuleRegistry.register_plugin(root="tinytrain", exclude=["labeling"])
