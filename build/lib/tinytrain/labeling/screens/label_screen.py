from kivy.app import App
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, ScreenManager

from tinytrain.labeling.components.content_area import TTContentArea
from tinytrain.labeling.components.info_bar import TTInfoBar
from tinytrain.labeling.components.menu_bar import TTMenuBar


class LabelScreen(Screen):
    def __init__(self, menu_bar=None, info_bar=None, content_area=None, **kwargs):
        super(LabelScreen, self).__init__(**kwargs)
        self.sm: ScreenManager = App.get_running_app().screen_manager

        v_layout = BoxLayout(orientation='vertical', size=self.size, pos=self.pos, spacing=dp(8), padding=[dp(8), dp(8)])
        self.add_widget(v_layout)

        # 菜单栏
        self.menu_bar = menu_bar if menu_bar else TTMenuBar()
        v_layout.add_widget(self.menu_bar)

        # 信息栏
        self.info_bar = info_bar if info_bar else TTInfoBar()
        v_layout.add_widget(self.info_bar)

        # 标注内容区
        self.content_area = content_area if content_area else TTContentArea()
        v_layout.add_widget(self.content_area)
