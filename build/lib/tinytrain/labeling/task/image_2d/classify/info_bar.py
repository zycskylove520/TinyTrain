from kivy.uix.label import Label

from tinytrain.labeling.components.info_bar import TTInfoBar


class Image2DClassifyInfoBar(TTInfoBar):
    def __init__(self, screen, **kwargs):
        super(Image2DClassifyInfoBar, self).__init__(**kwargs)
        self.screen = screen

        # 绑定事件
        self.screen.bind(on_read_file_info=self._on_read_file_info)

        self.label=Label()
        self.add_widget(self.label)

    def _on_read_file_info(self, screen, file_info):
        self.label.text = f"width: {file_info["width"]}, height: {file_info["height"]}"
