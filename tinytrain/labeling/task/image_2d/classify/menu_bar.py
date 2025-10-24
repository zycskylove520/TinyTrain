from tinytrain.labeling.components.menu_bar import TTMenuBar


class Image2DClassifyMenuBar(TTMenuBar):
    def __init__(self, screen, **kwargs):
        super(Image2DClassifyMenuBar, self).__init__(**kwargs)
        self.screen = screen

    def on_files_selected(self, files):
        super(Image2DClassifyMenuBar, self).on_files_selected(files)
        self.screen.dispatch("on_files_changed", files)