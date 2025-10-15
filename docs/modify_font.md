tinytrain框架默认使用matplotlib在不同系统下使用的默认字体，一般为英文。

如果需要支持第三方字体，请在assets目录下的fonts目录(如果fonts目录不存在，请手动创建)放置自己喜欢的字体文件，支持`ttf`或`otf`两种字体格式。

并在global_var目录下的__init__文件中修改以下内容：
```python
# 修改localization()，注意直接使用字体名称，无需额外路径
localization("xxx.ttf")
```