# 什么是TTConfigManager

TTConfigManager类是用于管理所有配置文件的全局配置类，从模型的训练到推理到部署，通过TTConfigManager可以在任何地方获取到需要的配置参数。

考虑到单个脚本同时训练多个模型的情况，因此TTConfigManager类不是单例类，建议开发者在搭建自己的模型时，使用一个TTConfigManager类的实例贯穿整个流程。

# 入门

使用TTConfigManager类需要一个toml或yaml/yml文件作为link文件，link文件用于链接用户根据自己需求创建的不同的配置文件。请注意，配置文件仅支持toml或yaml/yml。

任意创建一个配置文件，这里取名为:`my_config.toml`：

```toml
dataset_name = "COCO datset"
ip = "localhost"
```

接下来创建一个link文件，这里取名为:`my_link.toml`：

```toml
cfg = "my_config.toml"
```

`my_link.toml`文件中的键名可以是任意名称，值则是指向的toml或yaml/yml文件。

使用方式如下：

```python
from tinytrain.cfg import TTConfigManager

config_manager = TTConfigManager(link_file="my_link.toml")

# 访问my_config.toml文件中的键值
dataset_name = config_manager.cfg["dataset_name"]

# 修改my_config.toml文件中的键值
ip = config_manager.cfg["ip"] = "127.0.0.0"
```

构建完TTConfigManager实例后，得到的config_manager.XXX，XXX就是link.toml文件里定义的键名，通过config_manager.XXX访问到对应的键名所对应的整个toml或yaml/yml文件将会以字典的形式返回并接受访问。

# 额外

在tinytrain框架中，每个内置的模型都已经定义好了对应的link.toml文件和所需要的其他配置文件。如果需要创建新的配置文件，请注意键名不要与某些定制好的键名冲突，比如`core`、`model`。