from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout


class TTContentArea(BoxLayout):
    def __init__(self, **kwargs):
        super(TTContentArea, self).__init__(**kwargs)
        self.spacing = dp(1)
