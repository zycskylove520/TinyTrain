from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.button import Button

class TTInfoBar(BoxLayout):
    def __init__(self, **kwargs):
        super(TTInfoBar, self).__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(48)

        self.orientation = 'horizontal'

        with self.canvas.before:
            Color(0.5, 0.5, 0.5)
            self.bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(size=self.update_bg, pos=self.update_bg)

    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
