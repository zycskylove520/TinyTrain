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

from kivy.uix.label import Label

from tinytrain.labeling.components.info_bar import TTInfoBar


class Image2DClassifyInfoBar(TTInfoBar):
    def __init__(self, screen, **kwargs):
        super(Image2DClassifyInfoBar, self).__init__(**kwargs)
        self.screen = screen

        # 绑定事件
        self.screen.bind(on_read_file_info=self._on_read_file_info)

        self.label=Label()
        self.add_widget(self.label)

    def _on_read_file_info(self, screen, file_info):
        self.label.text = f"width: {file_info["width"]}, height: {file_info["height"]}"
