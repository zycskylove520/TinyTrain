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

from kivy.metrics import dp
from kivy.uix.gridlayout import GridLayout


class TTDynamicGridLayout(GridLayout):
    """
    右侧内容载体，使用 GridLayout 按列排布子控件。
    新增能力：
    1. 窗口宽度变化时自动调整 cols；
    2. 高度仍保持 minimum_height 绑定，ScrollView 正常滚动。
    """

    # 每列允许的最小宽度，按需要改
    COL_MIN_WIDTH = dp(80)

    def __init__(self, size_hint_y=None, **kwargs):
        """初始化内容面板。

        Args:
            size_hint_y: 垂直方向 size_hint，通常保持 None 以便手动设高。
            **kwargs: 透传给父类 GridLayout。
        """
        super().__init__(size_hint_y=size_hint_y, **kwargs)
        self.spacing = dp(8), dp(8)
        self.bind(minimum_height=self.setter('height'))
        # 监听窗口宽度变化
        self.bind(width=self._recompute_cols)
        # 第一次手动触发，保证初始状态也正确
        self._recompute_cols()

    def _recompute_cols(self, *args):
        """
        根据当前 GridLayout 的实际可用宽度计算列数，
        并一次性把子控件宽度设成“列宽”，保证横向正好填满。
        """
        # 1. 计算真正可用来放列的净宽
        avail = self.width - self.padding[0] - self.padding[2]  # 减去左右 padding
        cols = max(1, int(avail // (self.COL_MIN_WIDTH + self.spacing[0])))  # 每列占宽 = 列宽 + 列间 spacing
        net_w = avail - (cols - 1) * self.spacing[0]  # 去掉 (cols-1) 份列间距
        col_w = net_w / cols  # 每列实际应分到的宽度

        # 2. 同步列数
        self.cols = cols

        # 3. 统一设置子控件宽度（size_hint_x 必须关掉）
        for child in self.children:
            child.size_hint_x = None
            child.width = col_w