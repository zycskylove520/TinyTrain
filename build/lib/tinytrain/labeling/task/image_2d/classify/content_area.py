from pathlib import Path

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.stacklayout import StackLayout
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton

from tinytrain.labeling.components.content_area import TTContentArea
from tinytrain.labeling.components.dynamic_grid_layout import TTDynamicGridLayout
from tinytrain.utils.data_utils import cv_imread


class Image2DClassifyContentArea(TTContentArea):
    files = ListProperty()  # 存放打开文件读取到的文件路径

    def __init__(self, screen, **kwargs):
        super(Image2DClassifyContentArea, self).__init__(**kwargs)
        self.screen = screen

        # 绑定事件
        screen.bind(on_files_changed=self._on_files_changed)

        # 1 左侧文件选择区
        self.left_area = ScrollView(size_hint_x=0.15)
        self.add_widget(self.left_area)

        self.files_container = BoxLayout(orientation="vertical", size_hint_y=None)
        self.files_container.bind(minimum_height=self.files_container.setter('height'))
        self.left_area.add_widget(self.files_container)

        # 2 中间内容标注区
        self.center_area = Image()
        self.add_widget(self.center_area)

        # 3 右侧标签选择区
        self.right_area = ScrollView(size_hint_x=0.2)
        self.add_widget(self.right_area)

        self.label_container = StackLayout(orientation='lr-tb')
        self.label_container.bind(minimum_height=self.label_container.setter('height'))
        self.right_area.add_widget(self.label_container)

        # 3.1 添加新增标签按钮
        self.add_label_button = Button(
            text="+",
            size_hint=(None, None),  # 关键：完全放弃父布局的 hint
            height=dp(28),
            width=dp(30)  # 先随便给一个窄值
        )
        # 渲染后把宽度改成文字实际宽度
        self.add_label_button.bind(texture_size=self._set_label_width)
        self.add_label_button.bind(on_release=self._on_add_label_button_release)
        self.label_container.add_widget(self.add_label_button)

    def _on_files_changed(self, screen, files):
        self.files = files
        self.files_container.clear_widgets()

        # 1. 先全部创建，但先不设 down
        for file in files:
            file_button = ToggleButton(
                text=Path(file).stem,
                size_hint_y=None,
                height=dp(20),
                group='file_buttons',
                state='normal'  # 先全部 normal
            )
            file_button.img_file = file
            file_button.bind(state=self._on_file_button_state_changed)
            self.files_container.add_widget(file_button)

        # 2. 等 UI 落地后，再真正选中第一个
        if len(self.files_container.children) > 0:
            Clock.schedule_once(lambda dt: setattr(self.files_container.children[-1], "state", "down"), 0)

    def _on_file_button_state_changed(self, btn, state):
        # 只处理“被选中”
        if state == 'down':
            # 中心标注区设置图片
            self.center_area.source = btn.img_file

            # 广播图片信息
            img = cv_imread(btn.img_file)
            h, w, _ = img.shape

            file_info = {"width": w, "height": h}
            self.screen.dispatch("on_read_file_info", file_info)

    def _set_label_width(self, btn, tex_size):
        """texture_size 变化后把按钮宽度设成文字宽度 + 少量 padding"""
        btn.width = tex_size[0] + dp(16)  # 左右各 8 dp 留白

    def _on_add_label_button_release(self, btn):
        """带类别索引的标签管理弹窗"""

        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))

        # ---------- 顶部：新增 ----------
        add_box = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        new_label_input = TextInput(hint_text='新标签名称', multiline=False)
        confirm_btn = Button(text='增加', size_hint_x=None, width=dp(60))
        add_box.add_widget(new_label_input)
        add_box.add_widget(confirm_btn)
        content.add_widget(add_box)

        # ---------- 中部：列表 ----------
        scroll = ScrollView(size_hint_y=1)
        list_box = BoxLayout(orientation='vertical', size_hint_y=None)
        list_box.bind(minimum_height=list_box.setter('height'))
        scroll.add_widget(list_box)
        content.add_widget(scroll)

        # 读取主界面标签（去掉索引部分，只保留纯名称）
        old_names = [w.text.split(':', 1)[-1] for w in self.label_container.children
                     if isinstance(w, Button) and w.text != '+']

        def rebuild_list():
            """带独立序号 Label 的列表"""
            list_box.clear_widgets()
            for idx, name in enumerate(old_names, 1):
                item = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(5))

                # ① 序号
                idx_label = Label(text=f'{idx}:', size_hint_x=None, width=dp(30),
                                  halign='right', valign='center')
                idx_label.bind(size=idx_label.setter('text_size'))

                # ② 名称
                name_label = Label(text=name, halign='left', valign='center')
                name_label.bind(size=name_label.setter('text_size'))

                # ③ 删除按钮
                del_btn = Button(text='×', size_hint_x=None, width=dp(36),
                                 background_color=(1, 0, 0, .8))
                del_btn.bind(on_release=lambda _, n=name: delete_name(n))

                item.add_widget(idx_label)
                item.add_widget(name_label)
                item.add_widget(del_btn)
                list_box.add_widget(item)

        def delete_name(name):
            old_names.remove(name)
            rebuild_list()

        def add_name(text):
            t = text.strip()
            if t and t not in old_names:
                old_names.append(t)
                rebuild_list()
            new_label_input.text = ''

        confirm_btn.bind(on_release=lambda _: add_name(new_label_input.text))
        new_label_input.bind(on_text_validate=lambda _: add_name(new_label_input.text))
        rebuild_list()

        # ---------- 底部：关闭 ----------
        close_btn = Button(text='完成', size_hint_y=None, height=dp(40))
        content.add_widget(close_btn)

        popup = Popup(title='管理标签', content=content,
                      size_hint=(None, None), width=dp(300), height=dp(400))
        close_btn.bind(on_release=popup.dismiss)
        popup.open()

        # ---------- 关闭后同步 ----------
        def on_dismiss(*_):
            # 清掉非“+”按钮
            for c in list(self.label_container.children):
                if c is not self.add_label_button:
                    self.label_container.remove_widget(c)
            # 按顺序插入纯名称
            for name in reversed(old_names):
                btn = Button(text=name,  # ← 不带序号
                             size_hint=(None, None), height=dp(28))
                btn.bind(texture_size=lambda b, _: setattr(b, 'width', b.texture_size[0] + dp(16)))
                self.label_container.add_widget(btn, index=len(self.label_container.children) - 1)

        popup.bind(on_dismiss=on_dismiss)