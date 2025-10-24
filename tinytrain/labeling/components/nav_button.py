from kivy.uix.button import Button


class TTNavButton(Button):
    """
    通用导航按钮。
    点击后自动切换到 ScreenManager 中指定名称的屏幕，无需额外回调。
    """

    def __init__(self, sm, screen_name: str, **kwargs):
        """
        初始化导航按钮。

        Args:
            sm: 全局 ScreenManager 实例，用于执行屏幕切换。
            screen_name: 目标屏幕在 ScreenManager 中注册的唯一名称（str）。
            **kwargs: 透传给 Button 父类，可设置 text、size_hint 等常规属性。
        """
        super().__init__(**kwargs)
        self.sm = sm
        self.screen_name = screen_name

    def on_press(self):
        """按钮按下时触发，切换到预先绑定的屏幕。"""
        self.sm.current = self.screen_name
