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

from kivy.app import App
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen

from .content_area import Image2DClassifyContentArea
from .info_bar import Image2DClassifyInfoBar
from .menu_bar import Image2DClassifyMenuBar


class Image2DClassifyLabelScreen(Screen):
    def __init__(self, **kwargs):
        super(Image2DClassifyLabelScreen, self).__init__(**kwargs)
        # 注册事件
        self.register_event_type('on_files_changed')
        self.register_event_type('on_read_file_info')

        self.sm = App.get_running_app().screen_manager

        v_layout = BoxLayout(orientation='vertical', size=self.size, pos=self.pos, spacing=dp(8), padding=[dp(8), dp(8)])
        self.add_widget(v_layout)

        # 菜单栏
        self.menu_bar = Image2DClassifyMenuBar(self)
        v_layout.add_widget(self.menu_bar)

        # 信息栏
        self.info_bar = Image2DClassifyInfoBar(self)
        v_layout.add_widget(self.info_bar)

        # 标注内容区
        self.content_area = Image2DClassifyContentArea(self)
        v_layout.add_widget(self.content_area)

    # -------------自定义事件-----------------
    def on_files_changed(self, files: ListProperty):
        pass

    def on_read_file_info(self, file_info: dict):
        pass
