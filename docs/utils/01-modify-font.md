# 字体配置说明

tinytrain 框架默认使用 matplotlib 在不同系统下的默认英文字体。

## 支持第三方字体

如需使用自定义字体，请按照以下步骤操作：

### 1. 准备字体文件

字体文件可以存放在任何地方，如果希望统一管理，可以：

* 在 assets 目录下创建 fonts 文件夹（如不存在）
* 将您喜欢的字体文件（支持 .ttf 或 .otf 格式）放入该目录

### 2. 配置字体路径

#### 方法一：修改配置文件

编辑 `tinytrain/global_var/__init__.py` 文件中的 `DEFAULT_FONT` 变量：

```python
import os

# 支持相对路径或绝对路径
# 如直接填写字体文件名，框架会从 tinytrain/assets/fonts 目录中查找
DEFAULT_FONT = os.getenv("TINYTRAIN_FONT", "您的字体文件.ttf")
```

#### 方法二：通过环境变量设置

在运行脚本前设置环境变量：

```python
# 在脚本最上方设置字体
import os

os.environ["TINYTRAIN_FONT"] = "您的字体文件.ttf"

if __name__ == '__main__':
    # 您的脚本代码
    ...
```

## 注意事项

* 请确保字体文件路径正确且文件可访问
* 如遇到字体加载问题，请检查字体文件格式和路径设置