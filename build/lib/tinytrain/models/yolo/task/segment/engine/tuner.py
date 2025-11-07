from typing import Dict

from tinytrain.engine import TTBaseTuner


class YOLOSegmentTuner(TTBaseTuner):
    @staticmethod
    def build_param_tree() -> Dict[str, Dict[str, Dict]]:
        return {
            "core": {
                "batch_size": {"type": "discrete", "choices": [16, 32, 64], "low": 0, "high": 2},
                "optimizer": {"type": "discrete", "choices": ["SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop", "Adadelta", "Adagrad"], "low": 0, "high": 8},
                "scheduler": {"type": "discrete", "choices": ["auto", "LinearLR", "CosineLR", "ExponentialLR", "StepLR", "MultiStepLR"], "low": 0, "high": 5},
                "lr0": {"type": "continuous", "low": 1e-5, "high": 1e-1},
                "lr1": {"type": "continuous", "low": 1e-5, "high": 1e-1},
                "l1_norm": {"type": "continuous", "low": 0.0, "high": 1e-3},
                "momentum": {"type": "continuous", "low": 0.01, "high": 0.95},
                "weight_decay": {"type": "continuous", "low": 0.0, "high": 1e-4},
            },
            "loss": {
                "cls_loss_gain": {"type": "continuous", "low": 0.1, "high": 10},
                "box_loss_gain": {"type": "continuous", "low": 0.1, "high": 10},
                "dfl_loss_gain": {"type": "continuous", "low": 0.1, "high": 10},
                "seg_loss_gain": {"type": "continuous", "low": 0.1, "high": 10},
            }
        }
