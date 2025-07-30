"""
统一回调管理器
将训练、验证、推理、导出四阶段的所有钩子集中管理，支持动态增删与覆盖。
"""

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


class Callback:
    """
    全局回调容器
    整合 Trainer / Validator / Predictor / Exporter 的全部钩子，对外提供：
    - add_callback:  追加自定义函数
    - set_callback:  覆盖原有函数
    - run_callback:  按事件名批量执行
    """
    def __init__(self):
        self._callbacks = {
            **TrainerCallback().callbacks,
            **ValidatorCallback().callbacks,
            **PredictorCallback().callbacks,
            **ExporterCallback().callbacks
        }

    def add_callback(self, event: str, callback):
        """在指定事件后追加一个回调函数。"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise KeyError(f"Callback event '{event}' does not exist.")

    def set_callback(self, event: str, callback):
        """用新回调函数覆盖指定事件的所有回调。"""
        if event in self._callbacks:
            self._callbacks[event] = [callback]
        else:
            raise KeyError(f"Callback event '{event}' does not exist.")

    def run_callback(self, engine, event: str):
        """执行与事件关联的所有回调函数。"""
        for callback in self._callbacks.get(event, []):
            callback(engine)
