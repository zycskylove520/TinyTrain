from kivy._event import EventDispatcher


class FilesEvent(EventDispatcher):
    def __init__(self, **kwargs):
        super(FilesEvent, self).__init__(**kwargs)
        self.register_event_type('on_files_changed')

    def files_changed(self, files):
        self.dispatch('on_files_changed', files)

    def on_files_changed(self, files):
        pass
