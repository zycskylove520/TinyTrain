# -*- coding: utf-8 -*-
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

"""A simple tab-like panel built with Kivy.

左侧为固定宽度滚动列，右侧为对应内容区。
新增选项只需调用 `add_item(text, content_widget)`。
"""

from functools import partial

from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView


class TTTabPanel(BoxLayout):
    """左右分栏式“选项卡”面板。

    左侧为按钮列表（ScrollView），右侧为对应内容区（ScrollView）。
    通过 `add_item()` 动态添加选项；首次添加时会自动显示第一项。
    """

    def __init__(self, **kwargs):
        """初始化面板布局与左侧空列表。

        Args:
            **kwargs: 透传给父类 BoxLayout。
        """
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.spacing = dp(12)  # 左右两栏之间距离

        # 左侧按钮列
        sc1 = ScrollView(size_hint_x=None, width=dp(112))
        self.add_widget(sc1)

        self.left_grid_layout = GridLayout(cols=1, size_hint_y=None, spacing=dp(4), padding=[dp(8),0,0,0])
        # 保证左侧 GridLayout 高度能随子按钮动态变化
        self.left_grid_layout.bind(minimum_height=self.left_grid_layout.setter('height'))
        sc1.add_widget(self.left_grid_layout)

        # 右侧内容区
        self.sc2 = ScrollView()
        self.add_widget(self.sc2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_item(self, text: str, widget):
        """新增一个左侧选项按钮，并绑定点击后切换右侧内容。

        如果是第一个选项，会立即自动显示其内容。

        Args:
            text: 按钮显示文字。
            widget: 该选项对应的内容面板（TTTabPanelItem 实例）。
        """
        button = Button(text=text, size_hint_y=None, height=dp(48), font_size=dp(14))
        button.bind(on_press=partial(self.show_content, widget=widget))
        self.left_grid_layout.add_widget(button)

        # 首次添加时默认展示
        if not self.sc2.children:
            self.show_content(button, widget)

    def show_content(self, instance: Button, widget):
        """将右侧内容区切换为指定面板。

        内部会移除旧内容（如有）并添加新内容。

        Args:
            instance: 触发事件的按钮实例（未使用，可留作扩展）。
            widget: 待展示的内容面板。
        """
        if widget:
            if self.sc2.children:
                self.sc2.remove_widget(self.sc2.children[0])
            self.sc2.add_widget(widget)
