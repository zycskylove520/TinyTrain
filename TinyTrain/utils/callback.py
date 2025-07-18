class TrainerCallback:
    """
    Trainer callbacks
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
        """Called before the pretraining routine starts."""
        pass

    @staticmethod
    def on_prepare_train_end(trainer):
        """Called after the pretraining routine ends."""
        pass

    @staticmethod
    def on_train_start(trainer):
        """Called when the training starts."""
        pass

    @staticmethod
    def on_train_epoch_start(trainer):
        """Called at the start of each training epoch."""
        pass

    @staticmethod
    def on_train_batch_start(trainer):
        """Called at the start of each training batch."""
        pass

    @staticmethod
    def on_before_zero_grad(trainer):
        """Called before the gradients are set to zero."""
        pass

    @staticmethod
    def on_train_batch_end(trainer):
        """Called at the end of each training batch."""
        pass

    @staticmethod
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        pass

    @staticmethod
    def on_model_save(trainer):
        """Called when the model is saved."""
        pass

    @staticmethod
    def on_train_end(trainer):
        """Called when the training ends."""
        pass


class ValidatorCallback:
    """
    Validator callbacks
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
        """Called when the validation starts."""
        pass

    @staticmethod
    def on_val_batch_start(validator):
        """Called at the start of each validation batch."""
        pass

    @staticmethod
    def on_val_batch_end(validator):
        """Called at the end of each validation batch."""
        pass

    @staticmethod
    def on_val_end(validator):
        """Called when the validation ends."""
        pass


class PredictorCallback:
    """
    Predictor callbacks
    """

    def __init__(self):
        self.callbacks = {
            "on_predict_start": [self.on_predict_start],
            "on_predict_end": [self.on_predict_end],
        }

    @staticmethod
    def on_predict_start(predictor):
        """Called when the prediction starts."""
        pass

    @staticmethod
    def on_predict_end(predictor):
        """Called when the prediction ends."""
        pass


class ExporterCallback:
    """
    Exporter callbacks
    """

    def __init__(self):
        self.callbacks = {
            "on_export_start": [self.on_export_start],
            "on_export_end": [self.on_export_end],
        }

    @staticmethod
    def on_export_start(exporter):
        """Called when the model export starts."""
        pass

    @staticmethod
    def on_export_end(exporter):
        """Called when the model export ends."""
        pass


class Callback:
    def __init__(self):
        self._callbacks = {
            **TrainerCallback().callbacks,
            **ValidatorCallback().callbacks,
            **PredictorCallback().callbacks,
            **ExporterCallback().callbacks
        }

    def add_callback(self, event: str, callback):
        """Append a callback event to the given callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            raise KeyError(f"Callback event '{event}' does not exist.")

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        if event in self._callbacks:
            self._callbacks[event] = [callback]
        else:
            raise KeyError(f"Callback event '{event}' does not exist.")

    def run_callback(self, engine, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self._callbacks.get(event, []):
            callback(engine)
