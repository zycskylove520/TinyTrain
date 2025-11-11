# TTEngineRegistry使用指南

TTEngineRegistry类负责管理、注册所有的engine类到Core类中进行绑定。任何未注册的engine类将无法参与训练、验证、推理、导出等流程。

## 使用方式

推荐在Core类中注册engine类，而不是直接通过TTEngineRegistry装饰器的方式注册，其好处在于：

- 便于统一管理
- 清晰可读
- 可动态增减



