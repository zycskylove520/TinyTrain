from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager

from tinytrain.labeling import HOME_SCREEN
from tinytrain.labeling.components.nav_button import TTNavButton


class StartScreen(Screen):
    def __init__(self, **kwargs):
        super(StartScreen, self).__init__(**kwargs)
        self.sm: ScreenManager = App.get_running_app().screen_manager

        self.button = TTNavButton(sm=self.sm, screen_name=HOME_SCREEN, text="点击开始")
        self.add_widget(self.button)
