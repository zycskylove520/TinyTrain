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

from tinytrain.engine import TTBaseTrainer
from tinytrain.models.lprnet.data_format import LPRBatchDataInfo
from tinytrain.models.lprnet.dataset import LPRNetDataset


class LPRTrainer(TTBaseTrainer):
    def build_dataset(self, mode="train"):
        if mode == "train":
            return LPRNetDataset(config_manager=self.config_manager,
                                           img_path=self.train_dir,
                                           mode="train"
                                           )
        elif mode == "val":
            return LPRNetDataset(config_manager=self.config_manager,
                                           img_path=self.val_dir,
                                           mode="val"
                                           )
        else:
            raise NotImplementedError

    def preprocess_data(self, batch_samples: LPRBatchDataInfo) -> LPRBatchDataInfo:
        batch_samples.data = batch_samples.data.to(self.device, non_blocking=True)
        batch_samples.target = batch_samples.target.to(self.device, non_blocking=True)
        return batch_samples
