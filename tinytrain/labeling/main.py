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

from kivy.config import Config

FONT = '../../tinytrain/assets/fonts/QingguaMoeSans.ttf'
Config.set('kivy', 'default_font', ['TTFont', FONT])

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager

from tinytrain.labeling import START_SCREEN, HOME_SCREEN, IMAGE2D_CLASSIFY_SCREEN
from tinytrain.labeling.screens.home_screen import HomeScreen
from tinytrain.labeling.screens.start_screen import StartScreen
from tinytrain.labeling.task.image_2d.classify.screen import Image2DClassifyLabelScreen


# ---------- ① 自动 import 所有自定义模块（类注册） ----------
# from pathlib import Path
# import importlib
# pkg_dir = Path(__file__).with_name('components')
# for py in pkg_dir.glob('*.py'):
#     if py.name.startswith('_'):
#         continue
#     importlib.import_module(f'components.{py.stem}')
#
# pkg_dir = Path(__file__).with_name('ui')
# for py in pkg_dir.glob('*.py'):
#     if py.name.startswith('_'):
#         continue
#     importlib.import_module(f'ui.{py.stem}')


class TTLabelingApp(App):
    def __init__(self, **kwargs):
        super(TTLabelingApp, self).__init__(**kwargs)
        self.screen_manager = ScreenManager()
        self.add_screens()
        self.screen_manager.current = IMAGE2D_CLASSIFY_SCREEN

    def add_screens(self):
        self.screen_manager.add_widget(StartScreen(name=START_SCREEN))
        self.screen_manager.add_widget(HomeScreen(name=HOME_SCREEN))
        self.screen_manager.add_widget(Image2DClassifyLabelScreen(name=IMAGE2D_CLASSIFY_SCREEN))

    def build(self):
        return self.screen_manager


if __name__ == '__main__':
    TTLabelingApp().run()
