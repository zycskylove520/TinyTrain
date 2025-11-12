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

import torch

from tinytrain.engine import TTConfigModel
from tinytrain.loss import LPRCTCLoss
from tinytrain.models.lprnet.data_format import LPRBatchDataInfo


class LPRModel(TTConfigModel):
    """LPRNet 车牌识别模型"""
    def init_criterion(self):
        return LPRCTCLoss(
            lpr_loss_gain=self.config_manager.loss['lpr_loss_gain'],
            blank=self.config_manager.dataset["nc"] - 1
        )

    def loss(self, preds: torch.Tensor, batch_samples: LPRBatchDataInfo) -> tuple[float, dict]:
        return self.criterion(preds[0], batch_samples)

    def custom_parse_model_level(self, level, module_info):
        if self.config_manager.model["name"] == "LPRNet":
            if module_info["module"] == "LPRBackbone":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
                module_info["args"]["dropout_rate"] = self.config_manager.loss["dropout_rate"]

            elif module_info["module"] == "LPRHead":
                module_info["args"]["in_channels"] = 448 + self.config_manager.dataset["nc"]
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]

        elif self.config_manager.model["name"] == "TTLPRNet":
            if module_info["module"] == "torch.nn.Dropout":
                module_info["args"]["p"] = self.config_manager.loss["dropout_rate"]

            elif module_info["module"] == "LPRHead":
                module_info["args"]["nc"] = self.config_manager.dataset["nc"]
