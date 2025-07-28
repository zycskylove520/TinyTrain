import os

import torch

from torch import nn
from copy import deepcopy
from queue import Queue

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.utils import LOGGER


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models."""
    DEPTH_GAIN = None  # 深度增益

    def __init__(self, config_manager: ConfigManager, *args, **kwargs):
        super().__init__()
        self.criterion = None
        self.config_manager = config_manager
        self.module_list, self.record_list, self.ask_set, self.log_info = self.parse_model(config_manager)
        # 只有第一次启动时打印模型信息
        if "LOCAL_RANK" not in os.environ:
            self._model_log()

    def forward(self, data, **kwargs):
        if isinstance(data, BaseBatchDataInfo):
            return self.loss(data, **kwargs)
        else:
            return self.inference(data, **kwargs)

    def inference(self, data, **kwargs):
        # 检查输入是否为多输入（list 或 tuple）
        if isinstance(data, (list, tuple)):
            inputs = {index: item for index, item in enumerate(data)}
        else:
            inputs = {0: data}
        inputs_idx = 0
        entry_idx_mapping = dict()

        outputs: list = []  # 存放模型推理最终的输出
        for i, layer in enumerate(self.module_list):
            try:
                # 拿到第i层的info
                record_info = self.record_list[i]

                # entry层特殊部分
                if record_info["type"] == "entry":
                    try:
                        entry_idx_mapping[i] = inputs_idx
                    except KeyError as e:
                        LOGGER.error(f"model input num != entry num, inputs num: {len(inputs)}, entry num: {inputs_idx}")
                        raise e
                    inputs_idx += 1

                    if len(record_info["from"]) == 1:
                        rf = record_info["from"][0]
                        assert rf == -1, f"if entry type from_list only have one element, must be -1."
                        # 拿对应的input的输入数据
                        data = layer(inputs[entry_idx_mapping[i]])
                    elif len(record_info["from"]) > 1:
                        rfs = record_info["from"]
                        for rf in [j for j in rfs if j != -1]:
                            assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"
                        temp_list: list = []
                        for rf in rfs:
                            if rf != -1:
                                temp_list.append(self.record_list[rf]["data"])
                            else:
                                temp_list.append(inputs[entry_idx_mapping[i]])
                        data = layer(temp_list)
                    else:
                        raise ValueError(f"from length must >=1.")

                    if i in self.ask_set:
                        self.record_list[i]["data"] = data  # add new key-value to record_list
                    continue

                if len(record_info["from"]) == 1:
                    rf = record_info["from"][0]
                    if rf == -1:
                        data = layer(data)
                    else:
                        assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"
                        data = layer(self.record_list[rf]["data"])
                elif len(record_info["from"]) > 1:
                    rfs = record_info["from"]
                    for rf in [j for j in rfs if j != -1]:
                        assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"

                    temp_list: list = []
                    for rf in rfs:
                        if rf != -1:
                            temp_list.append(self.record_list[rf]["data"])
                        else:
                            temp_list.append(data)
                    data = layer(temp_list)
                else:
                    raise ValueError(f"from length must >=1.")

                if record_info["type"] == "head":
                    outputs.append(data)

                if i in self.ask_set:
                    self.record_list[i]["data"] = data  # add new key-value to record_list
            except Exception as e:
                LOGGER.error(f"inference error: {e}, in layer: {i}.")
                raise e

        if len(outputs) == 0:
            raise RuntimeError("No output.")
        return outputs

    def loss(self, batch_samples: BaseBatchDataInfo, preds=None, **kwargs):
        """
        Compute loss.

        """
        if self.criterion is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch_samples.data) if preds is None else preds
        return self.criterion(preds, batch_samples)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")

    def load_model_state_dict(self, state_dict, force_load=True):
        """
        force_load为True，要求模型参数键对应的值的shape必须完全一致，适用于predict和export
        """
        model_state_dict = self.state_dict()

        match_state_dict = {}
        for key in state_dict:
            if key in self.state_dict():
                if state_dict[key].shape == model_state_dict[key].shape:
                    match_state_dict[key] = state_dict[key]
                else:
                    if not force_load:
                        raise KeyError(f"no match key:{key}, pt key shape:{state_dict[key].shape}, model key shape:{model_state_dict[key].shape}")
                    LOGGER.warning(f"no match key:{key}, pt key shape:{state_dict[key].shape}, model key shape:{model_state_dict[key].shape}")
            else:
                LOGGER.warning(f"not exist key:{key}")

        self.load_state_dict(match_state_dict, strict=False)

    def custom_parse_model(self, module_info):
        pass

    def parse_model(self, config_manager: ConfigManager):
        """

        :param config_manager:
        :return:
        """
        scale_info = config_manager.model["scales"][config_manager.model["scale"]]

        # 计算一次并写入类变量（所有实例共享）
        BaseModel.DEPTH_GAIN = scale_info["depth"]

        layers, record_list, log_info = nn.ModuleList(), [], []
        ask_set = set()

        # 直接读取类变量
        depth = BaseModel.DEPTH_GAIN

        for level, info in enumerate(config_manager.model["network"]):
            try:
                # deepcopy防止修改原始配置文件导致加载模型异常
                _info = deepcopy(info)
                _type: str = _info["type"]  # 当前模块的位置类型，目前三种:"entry"、"flow"、"head"
                _from: list = _info["from"]
                _module: str = _info["module"]
                _repeat: int = _info["repeat"]
                _args: dict = _info.get("args", {})

                # check network
                assert _type in {"entry", "flow", "head"}, f"level_{level}: {_module} 'type' must be 'entry', 'flow', or 'head'"
                assert len(_from) > 0, f"level_{level}: {_module} 'from' list length must greater than 0!"
                assert _repeat > 0, "level_{level}: {_module} 'repeat' must greater than 0!"

                # 限制第0层必须为entry层,且只能第0层为entry层
                if level == 0:
                    assert _type == "entry", f"level_0: {_module} 'type' must be 'entry'"
                    _from = [-1]

                if _type == "entry":  # entry层
                    # 子类可定制
                    pass
                elif _type == "flow":  # flow层
                    # 子类可定制
                    pass
                elif _type == "head":  # head层
                    # 子类可定制
                    pass

                # depth gain
                if _type == "flow":
                    _repeat = max(round(_repeat * depth), 1) if _repeat > 1 else _repeat

                # 用户可自定义模型解析方式
                self.custom_parse_model(_info)

                # 构造网络模块
                try:
                    layer = self._get_layer(_module)
                except (NameError, AttributeError) as e:
                    raise ValueError(f"Failed to get module {_module}: {e}")

                layer = torch.nn.Sequential(*(layer(**_args) for _ in range(_repeat))) if _repeat > 1 else layer(**_args)
                layers.append(layer)

                record_list.append({
                    "type": _type,  # type指明当前的模块类型
                    "module": _module,
                    "layer": level,  # layer指明当前是第几层
                    "from": _from,  # from指明接受第几层的输入
                    "repeat": _repeat
                })

                ask_set.update([f for f in _from if f != -1])

                log_info.append(_info)
            except Exception as e:
                LOGGER.error(f"parse_model error: {e}, in layer:{level}.")
                raise e

        return layers, record_list, ask_set, log_info

    def _model_log(self):
        # 获取对齐长度，打印会更好看
        align_len = dict({"layer": 5, "type": 4, "repeat": 6, "from": 4, "module": 6, "args": 4})
        for layer, info in enumerate(self.log_info):
            layer_len = len(str(layer))
            if align_len["layer"] < layer_len:
                align_len["layer"] = layer_len

            type_len = len(str(info["type"]))
            if align_len["type"] < type_len:
                align_len["type"] = type_len

            repeat_len = len(str(info["repeat"]))
            if align_len["repeat"] < repeat_len:
                align_len["repeat"] = repeat_len

            from_len = len(str(info["from"]))
            if align_len["from"] < from_len:
                align_len["from"] = from_len

            module_len = len(str(info["module"]))
            if align_len["module"] < module_len:
                align_len["module"] = module_len

            args_len = len(str(info.get("args", 0)))
            if align_len["args"] < args_len:
                align_len["args"] = args_len

        scale = self.config_manager.model["scale"]
        scale_info = self.config_manager.model["scales"][scale]
        LOGGER.info(f"start parse model...")
        print(f'model scale:{scale},'
              f' depth:{scale_info["depth"]},'
              f' width:{scale_info.get("width", None)}.')
        print(f"{self.__class__.__name__} struct:")
        print(
            f'|{'layer':^{align_len["layer"]}}'
            f'|{'type':^{align_len["type"]}}'
            f'|{'repeat':^{align_len["repeat"]}}'
            f'|{'from':^{align_len["from"]}}'
            f'|{'module':^{align_len["module"]}}'
            f'|{'args':^{align_len["args"]}}|'
        )
        for layer, info in enumerate(self.log_info):
            print(
                f'|{layer: ^{align_len["layer"]}}'
                f'|{info["type"]:^{align_len["type"]}}'
                f'|{info["repeat"]:^{align_len["repeat"]}}'
                f'|{str(info["from"]):^{align_len["from"]}}'
                f'|{info["module"]:^{align_len["module"]}}'
                f'|{str(info.get("args", {})):<{align_len["args"]}}|'
            )
        print(f"model summary: {scale_info['summary']}\n")

    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                # 自定义初始化
                nn.init.constant_(m.weight, 1.0)  # gamma 初始化为 1.0
                nn.init.constant_(m.bias, 0.0)  # beta 初始化为 0.0
                nn.init.constant_(m.running_mean, 0.0)  # running_mean 初始化为 0.0
                nn.init.constant_(m.running_var, 1.0)  # running_var 初始化为 1.0
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
                m.inplace = True

    @staticmethod
    def _get_layer(module_str: str):
        import importlib

        module_str = module_str.strip()
        try:
            # 1️⃣ torch.nn.*
            if module_str.lower().startswith("nn."):
                name = module_str[3:]
                return getattr(importlib.import_module("torch.nn"), name)

            # 2️⃣ torchvision.ops.*
            if module_str.lower().startswith("torchvision.ops."):
                name = module_str[16:]
                return getattr(importlib.import_module("torchvision.ops"), name)

            # 3️⃣ transformers.*
            if module_str.lower().startswith("transformers."):
                name = module_str[11:]
                return getattr(importlib.import_module("transformers"), name)

            # 4️⃣ tinytrain.modules.*
            return getattr(importlib.import_module("tinytrain.modules"), module_str)

        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(
                f"Unrecognized module string '{module_str}'. "
                f"Please check spelling. "
                f"Details: {e}"
            ) from None
