"""
Copyright (c) 2025 zycskylove520

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict

from tinytrain.engine import TTBaseTuner


class LPRTuner(TTBaseTuner):
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
                "max_time_steps": {"type": "discrete", "choices": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "low": 0, "high": 10},
                "dropout_rate": {"type": "continuous", "low": 0.0, "high": 1.0}
            }
        }
