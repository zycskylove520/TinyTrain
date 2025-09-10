from torch import nn

from tinytrain.engine import BaseModel
from tinytrain.engine.trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    def make_param_groups(self, model: BaseModel, lr, weight_decay, **kwargs) -> dict:
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
