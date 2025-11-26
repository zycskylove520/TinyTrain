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

from torch import nn

from tinytrain.engine import TTBaseModel
from tinytrain.engine.trainer import TTBaseTrainer


class YOLOTrainer(TTBaseTrainer):
    def make_param_groups(self, model: TTBaseModel, lr, weight_decay) -> dict:
        # 所有可训练参数，按 name 排序保证 DDP 一致性
        named_params = sorted(
            [(n, p) for n, p in self.model.named_parameters() if p.requires_grad],
            key=lambda x: x[0]
        )

        norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        groups = {
            "no_decay_bias": {"params": [], "weight_decay": 0.0, "lr": lr},
            "no_decay_norm": {"params": [], "weight_decay": 0.0, "lr": lr},
            "with_decay": {"params": [], "weight_decay": weight_decay, "lr": lr},
        }

        for name, param in named_params:
            # 找到该参数所属的模块
            parent_module = None
            for m_name, m in self.model.named_modules():
                # 判断 param 是否属于该模块的直属参数
                if any(id(p) == id(param) for p in m.parameters(recurse=False)):
                    parent_module = m
                    break

            if "bias" in name:
                groups["no_decay_bias"]["params"].append(param)
            elif parent_module is not None and isinstance(parent_module, norm_types):
                groups["no_decay_norm"]["params"].append(param)
            else:
                groups["with_decay"]["params"].append(param)

        return groups
