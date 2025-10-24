from .cfg import TTModuleRegistry, TTEngineRegistry
from .models import *

# 自动扫描注册算子
TTModuleRegistry.register_plugin(exclude=["labeling"])
