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
from kivy.uix.screenmanager import Screen, ScreenManager

from tinytrain.labeling import IMAGE2D_CLASSIFY_SCREEN
from tinytrain.labeling.components.dynamic_grid_layout import TTDynamicGridLayout
from tinytrain.labeling.components.nav_button import TTNavButton
from tinytrain.labeling.components.tab_panel import TTTabPanel


class HomeScreen(Screen):
    """
    该页面负责进行标注任务选择。
    """

    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        self.sm: ScreenManager = App.get_running_app().screen_manager

        tab_panel = TTTabPanel(size=self.size, pos=self.pos, padding=dp(5))
        self.add_widget(tab_panel)

        # 1 2D图像任务
        tab_item1 = TTDynamicGridLayout()
        tab_panel.add_item(f"2D图像", tab_item1)
        # 1.1 图像分类任务
        button1_1 = TTNavButton(sm=self.sm, screen_name=IMAGE2D_CLASSIFY_SCREEN, text=f"图像分类", size_hint_y=None, height=dp(40))
        tab_item1.add_widget(button1_1)

