from .cfg import TTModuleRegistry, TTEngineRegistry
from .models import *

__version__ = "0.2.4"

# 自动扫描注册算子
TTModuleRegistry.register_plugin(exclude=["labeling"])
