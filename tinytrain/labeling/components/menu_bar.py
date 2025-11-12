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
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.actionbar import ActionBar, ActionView, ActionPrevious, ActionGroup, ActionButton

from tinytrain.labeling import HOME_SCREEN


class TTMenuBar(ActionBar):
    files = ListProperty()  # 存放打开文件读取到的文件路径

    def __init__(self, **kwargs):
        super(TTMenuBar, self).__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(48)

        self.sm = App.get_running_app().screen_manager

        # 视图
        self.view = ActionView()
        self.add_widget(self.view)

        # 1 返回/标题区域
        self.ap = ActionPrevious(with_previous=False)
        self.ap.bind(on_release=self.on_action_previous_release)
        self.view.add_widget(self.ap)

        # 2 文件下拉菜单
        self.file_group = ActionGroup(text='文件', mode='spinner', use_separator=True)
        self.view.add_widget(self.file_group)

        # 2.1 选择文件按钮
        open_btn = ActionButton(text='选择文件')
        open_btn.bind(on_release=self._ask_files)
        self.file_group.add_widget(open_btn)

        # 2 保存下拉菜单
        self.save_group = ActionGroup(text='保存', mode='spinner', use_separator=True)
        self.view.add_widget(self.save_group)

    def on_action_previous_release(self, _):
        # 按下图标按钮跳转回任务选择页面
        self.sm.current = HOME_SCREEN

    def on_files_selected(self, files):
        self.files = files

    def _ask_files(self, _, filetypes: list[tuple[str, str]] = None):
        # 必须阻塞主线程，防止用户误操作
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="请选择文件（可多选）",
            filetypes=filetypes if filetypes else [("所有文件", "*.*")]
        )
        root.destroy()
        # 刷新 UI
        Clock.schedule_once(lambda dt: self.on_files_selected(paths), 0)
