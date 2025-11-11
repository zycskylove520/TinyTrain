tinytrain框架默认使用matplotlib在不同系统下使用的默认字体，一般为英文。

如果需要支持第三方字体，请在assets目录下的fonts目录(如果fonts目录不存在，请手动创建)放置自己喜欢的字体文件，支持`ttf`或`otf`两种字体格式。

可直接修改global_var目录下的__init__文件中的DEFAULT_FONT变量的值：
```python
DEFAULT_FONT = "xxx.ttf" # 注意直接使用字体名称，无需额外路径
```

也可通过外界启动脚本时预先设定DEFAULT_FONT的值：
```python
from tinytrain import global_var
global_var.DEFAULT_FONT = "xxx.ttf"

if __name__ == '__main__':
    # 脚本训练代码之类
    ...
```