"""
统一回调管理器
将训练、验证、推理、导出四阶段的所有钩子集中管理，支持动态增删与覆盖。
"""
from enum import Enum
from typing import Callable, Any, List, Dict

class Events(str, Enum):
    """训练阶段"""
    ON_PREPARE_TRAIN_START = "on_prepare_train_start"
    ON_PREPARE_TRAIN_END   = "on_prepare_train_end"
    ON_TRAIN_START         = "on_train_start"
    ON_TRAIN_EPOCH_START   = "on_train_epoch_start"
    ON_TRAIN_BATCH_START   = "on_train_batch_start"
    ON_BEFORE_ZERO_GRAD    = "on_before_zero_grad"
    ON_TRAIN_BATCH_END     = "on_train_batch_end"
    ON_TRAIN_EPOCH_END     = "on_train_epoch_end"
    ON_MODEL_SAVE          = "on_model_save"
    ON_TRAIN_END           = "on_train_end"

    """验证阶段"""
    ON_VAL_START       = "on_val_start"
    ON_VAL_BATCH_START = "on_val_batch_start"
    ON_VAL_BATCH_END   = "on_val_batch_end"
    ON_VAL_END         = "on_val_end"

    """推理阶段"""
    ON_PREDICT_START            = "on_predict_start"
    ON_PREDICT_BATCH_START      = "on_predict_batch_start"
    ON_PREDICT_PREPROCESS_END   = "on_predict_preprocess_end"
    ON_PREDICT_INFERENCE_END    = "on_predict_inference_end"
    ON_PREDICT_BATCH_END        = "on_predict_batch_end"
    ON_PREDICT_END              = "on_predict_end"

    """导出阶段"""
    ON_EXPORT_START = "on_export_start"
    ON_EXPORT_END   = "on_export_end"


class TrainerCallback:
    """
    训练阶段钩子
    生命周期：prepare → train → epoch → batch → save → end
    """

    def __init__(self):
        self.callbacks = {
            "on_prepare_train_start": [self.on_prepare_train_start],
            "on_prepare_train_end": [self.on_prepare_train_end],
            "on_train_start": [self.on_train_start],
            "on_train_epoch_start": [self.on_train_epoch_start],
            "on_train_batch_start": [self.on_train_batch_start],
            "on_train_batch_end": [self.on_train_batch_end],
            "on_train_epoch_end": [self.on_train_epoch_end],
            "on_model_save": [self.on_model_save],
            "on_train_end": [self.on_train_end]
        }

    @staticmethod
    def on_prepare_train_start(trainer):
        """训练前准备开始时调用。"""
        pass

    @staticmethod
    def on_prepare_train_end(trainer):
        """训练前准备结束时调用。"""
        pass

    @staticmethod
    def on_train_start(trainer):
        """训练开始时调用。"""
        pass

    @staticmethod
    def on_train_epoch_start(trainer):
        """每个训练 epoch 开始时调用。"""
        pass

    @staticmethod
    def on_train_batch_start(trainer):
        """每个训练 batch 开始时调用。"""
        pass

    @staticmethod
    def on_before_zero_grad(trainer):
        """梯度清零前调用。"""
        pass

    @staticmethod
    def on_train_batch_end(trainer):
        """每个训练 batch 结束时调用。"""
        pass

    @staticmethod
    def on_train_epoch_end(trainer):
        """每个训练 epoch 结束时调用。"""
        pass

    @staticmethod
    def on_model_save(trainer):
        """模型保存时调用。"""
        pass

    @staticmethod
    def on_train_end(trainer):
        """训练结束时调用。"""
        pass


class ValidatorCallback:
    """
    验证阶段钩子
    生命周期：val → batch-start → batch-end → val-end
    """

    def __init__(self):
        self.callbacks = {
            "on_val_start": [self.on_val_start],
            "on_val_batch_start": [self.on_val_batch_start],
            "on_val_batch_end": [self.on_val_batch_end],
            "on_val_end": [self.on_val_end],
        }

    @staticmethod
    def on_val_start(validator):
        """验证开始时调用。"""
        pass

    @staticmethod
    def on_val_batch_start(validator):
        """验证 batch 开始时调用。"""
        pass

    @staticmethod
    def on_val_batch_end(validator):
        """验证 batch 结束时调用。"""
        pass

    @staticmethod
    def on_val_end(validator):
        """验证结束时调用。"""
        pass


class PredictorCallback:
    """
    推理阶段钩子
    生命周期：predict → batch-start → preprocess → inference → batch-end → predict-end
    """

    def __init__(self):
        self.callbacks = {
            "on_predict_start": [self.on_predict_start],
            "on_predict_batch_start": [self.on_predict_batch_start],
            "on_predict_preprocess_end": [self.on_predict_preprocess_end],
            "on_predict_inference_end": [self.on_predict_inference_end],
            "on_predict_batch_end": [self.on_predict_batch_end],
            "on_predict_end": [self.on_predict_end],
        }

    @staticmethod
    def on_predict_start(predictor):
        """推理开始时调用。"""
        pass

    @staticmethod
    def on_predict_batch_start(predictor):
        """推理 batch 开始时调用。"""
        pass

    @staticmethod
    def on_predict_preprocess_end(predictor):
        """预处理结束时调用。"""
        pass

    @staticmethod
    def on_predict_inference_end(predictor):
        """推理结束时调用。"""
        pass

    @staticmethod
    def on_predict_batch_end(predictor):
        """推理 batch 结束时调用。"""
        pass

    @staticmethod
    def on_predict_end(predictor):
        """推理结束时调用。"""
        pass


class ExporterCallback:
    """
    导出阶段钩子
    生命周期：export-start → export-end
    """

    def __init__(self):
        self.callbacks = {
            "on_export_start": [self.on_export_start],
            "on_export_end": [self.on_export_end],
        }

    @staticmethod
    def on_export_start(exporter):
        """导出开始时调用。"""
        pass

    @staticmethod
    def on_export_end(exporter):
        """导出结束时调用。"""
        pass


class CallbackWrapper:
    def __init__(
            self,
            fn: Callable[[Any], None],
            priority: int = 0,
            once: bool = False,
            swallow_exceptions: bool = True,
    ):
        self.fn = fn
        self.priority = priority
        self.once = once
        self.swallow_exceptions = swallow_exceptions
        self._called = False

    def __call__(self, engine):
        if self.once and self._called:
            return
        try:
            self.fn(engine)
        except Exception as e:
            if not self.swallow_exceptions:
                raise e
            # 可以换成 logging
            print(f"[Callback Error] {self.fn.__name__}: {e}")
        finally:
            self._called = True

    def __lt__(self, other):  # 用于排序
        return self.priority > other.priority  # 数字越大越优先


class Callback:
    """
    全局回调管理器,支持优先级 / once / 覆盖
    """

    def __init__(self):
        self._callbacks: Dict[str, List[CallbackWrapper]] = {
            event: [] for event in Events
        }
        # 加载内置钩子
        self._register_builtin()

    # --------------------------------------------------
    # 内置钩子注册
    def _register_builtin(self):
        for cls in (TrainerCallback, ValidatorCallback,
                    PredictorCallback, ExporterCallback):
            instance = cls()
            for event, funcs in instance.callbacks.items():
                # 内置钩子优先级默认 -100，方便用户 override
                for f in funcs:
                    self.add_callback(event, f, priority=-100)

    # --------------------------------------------------
    # 统一增删改查
    def add_callback(
            self,
            event: str | Events,
            callback: Callable[[Any], None],
            priority: int = 0,
            once: bool = False,
            override: bool = False,
    ):
        wrapper = CallbackWrapper(callback, priority, once)
        if override:
            self._callbacks[event] = [wrapper]
        else:
            self._callbacks[event].append(wrapper)
            self._callbacks[event].sort()

    def set_callback(self, event: str, callback: Callable[[Any], None]):
        self.add_callback(event, callback, override=True)

    def remove_callback(self, event: str, callback: Callable):
        self._callbacks[event] = [
            w for w in self._callbacks[event] if w.fn != callback
        ]

    def run_callback(self, engine, event: str):
        for wrapper in self._callbacks.get(event, []):
            wrapper(engine)

    # --------------------------------------------------
    # 帮助调试：列出所有已注册事件和钩子数量
    def summary(self):
        return {e: len(cbs) for e, cbs in self._callbacks.items() if cbs}
