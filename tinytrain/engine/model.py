import torch

from torch import nn
from copy import deepcopy
from typing import Any, Dict

from tinytrain.cfg import ConfigManager, TTModuleRegistry
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.global_var import RANK
from tinytrain.utils import LOGGER


class BaseModel(nn.Module):
    """
    BaseModel 是所有深度学习模型的统一基类，负责：
    1. 根据配置文件动态解析网络结构（entry / flow / head）。
    2. 管理前向传播：支持推理模式（inference）与训练模式（loss）。
    3. 提供权重加载、初始化、日志打印等通用功能。
    4. 兼容绝大多数AI任务（视觉、自然语言处理等）。

    设计要点：
    - 结构配置完全由 ConfigManager 驱动，无需硬编码。
    - 支持深度增益（depth gain）自动缩放重复模块。
    - 内置模块缓存机制（record_list + ask_set）确保数据流正确。
    - 支持自定义模块解析（custom_parse_model）。
    - 自动初始化权重（initialize_weights），支持 Kaiming / BN / ReLU 等。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: ConfigManager = None, device: torch.device = None, *args, **kwargs):
        """
        初始化模型。

        Args:
            config_manager (ConfigManager): 配置管理器，包含模型结构、超参数、设备等信息。
            *args, **kwargs: 预留扩展参数，供子类使用。
        """
        super().__init__()
        self.DEPTH_GAIN = None  # 深度增益

        self.config_manager = config_manager
        self.device = device
        self.module_list, self.record_list, self.ask_set, self.log_info = self.parse_model()
        self.criterion = None

        # 只有第一次启动时打印模型信息
        if RANK in {-1, 0}:
            self._model_log()

    # ------------------------------------------------------------------
    # 2. 子类可重写钩子
    # ------------------------------------------------------------------
    def init_criterion(self):
        """
        初始化任务特定的损失函数。

        Returns:
            nn.Module: 损失模块。

        Raises:
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")

    def loss(self, preds: list[torch.Tensor], batch_samples: BaseBatchDataInfo) -> tuple[float, dict]:
        """
        训练/验证模式：计算损失。

        Args:
            preds (list[torch.Tensor] | None): 模型前向推理输出结果
            batch_samples (BaseBatchDataInfo): 包含输入与标签的数据对象。

        Returns:
            tuple[float, dict]: (总损失, 各分量损失字典),各分量损失字典例如：{"cls_loss", value}
        """
        return self.criterion(preds, batch_samples)

    def custom_parse_model(self, level, module_info):
        """
        钩子：子类可重写以动态修改模块配置。

        Args:
            level (int): 当前模块属于第几层
            module_info (dict): 当前模块的配置字典。
        """
        pass

    def initialize_weights(self):
        """
        初始化模型权重：
        - Conv2d: Kaiming 正态分布
        - BatchNorm2d: γ=1, β=0, running_mean=0, running_var=1
        - 激活函数: 设置为 inplace=True

        注意：该函数不会自动调用，需用户手动调用
        """

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

    # ------------------------------------------------------------------
    # 3. 统一前向入口（不建议重写）
    # ------------------------------------------------------------------
    def forward(self, data: BaseBatchDataInfo | torch.Tensor | list[torch.Tensor]):
        """
        统一入口：根据输入类型自动选择推理或训练模式。

        Args:
            data:
                - BaseBatchDataInfo：训练/验证模式，计算 loss。
                - torch.Tensor | list[torch.Tensor]：推理模式，执行 inference。

        Returns:
            Union[list[torch.Tensor], tuple]: 推理输出或 (loss, loss_items)。
        """

        if isinstance(data, BaseBatchDataInfo):
            outputs = self.inference(data.data, data.extra_data)
            return self.loss(outputs, data)
        elif isinstance(data, (torch.Tensor, list[torch.Tensor])):
            return self.inference(data)
        else:
            raise TypeError(f"type(data): {type(data)} is not supported")

    def inference(self, data: torch.Tensor | list[torch.Tensor], extra_data: Dict[str, Dict[str, Any]] = None):
        """
        推理模式前向传播。

        Args:
            data (torch.Tensor | list[torch.Tensor]): 输入张量或多输入列表。
            extra_data Dict[str, Dict[str,Any]]: 传递给模型的额外参数.

        Returns:
            list[torch.Tensor]: 模型输出列表（每个 head 对应一个输出）。

        Raises:
            RuntimeError: 如果未检测到任何输出。
        """

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
                current_data = data

                # 拿到第i层的info
                record_info = self.record_list[i]
                module_type: str = record_info["type"]
                module_name: str = record_info["module"]
                num_from: int = len(record_info["from"])

                # entry层特殊部分
                if module_type == "entry":
                    try:
                        entry_idx_mapping[i] = inputs_idx
                    except KeyError:
                        LOGGER.error(f"model input num != entry num, inputs num: {len(inputs)}, entry num: {inputs_idx}")
                        raise
                    inputs_idx += 1

                    if num_from == 1:
                        rf = record_info["from"][0]
                        assert rf == -1, f"if entry type from_list only have one element, must be -1."
                        # 拿对应的input的输入数据
                        current_data = inputs[entry_idx_mapping[i]]
                    elif num_from > 1:
                        rfs = record_info["from"]
                        for rf in [j for j in rfs if j != -1]:
                            assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"
                        temp_list: list = []
                        for rf in rfs:
                            if rf != -1:
                                temp_list.append(self.record_list[rf]["data"])
                            else:
                                temp_list.append(inputs[entry_idx_mapping[i]])
                        current_data = temp_list
                    else:
                        raise ValueError(f"from length must >=1.")
                else:
                    if num_from == 1:
                        rf = record_info["from"][0]
                        if rf == -1:
                            pass
                        else:
                            assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"
                            current_data = self.record_list[rf]["data"]
                    elif num_from > 1:
                        rfs = record_info["from"]
                        for rf in [j for j in rfs if j != -1]:
                            assert rf in self.ask_set, f"from index {rf} not found in ask_set {sorted(self.ask_set)}"

                        temp_list: list = []
                        for rf in rfs:
                            if rf != -1:
                                temp_list.append(self.record_list[rf]["data"])
                            else:
                                temp_list.append(data)
                        current_data = temp_list
                    else:
                        raise ValueError(f"from length must >=1.")

                if extra_data:
                    data = layer(current_data, **extra_data.get(module_name, {}))
                else:
                    data = layer(current_data)

                if module_type == "head":
                    outputs.append(data)

                if i in self.ask_set:
                    self.record_list[i]["data"] = data  # add new key-value to record_list
            except Exception as e:
                LOGGER.error(f"inference error: {e}, in layer: {i}.")
                raise

        if len(outputs) == 0:
            raise RuntimeError("Model no output!")
        return outputs

    # ------------------------------------------------------------------
    # 4. 模型结构解析（内部工具，不建议重写）
    # ------------------------------------------------------------------
    def parse_model(self):
        """
        解析配置文件，动态构建网络结构。

        Returns:
            tuple:
                - nn.ModuleList: 按顺序的模块列表。
                - list[dict]: 每层记录信息（type, from, repeat 等）。
                - set[int]: 需要缓存输出的层索引集合。
                - list[dict]: 日志信息（用于打印结构）。
        """
        scale_info = self.config_manager.model["scales"][self.config_manager.model["scale"]]

        # 获得深度增益
        self.DEPTH_GAIN = scale_info["depth"]

        layers, record_list, log_info = nn.ModuleList(), [], []
        ask_set = set()

        # 直接读取变量
        depth = self.DEPTH_GAIN

        for level, info in enumerate(self.config_manager.model["network"]):
            try:
                # deepcopy防止修改原始配置文件导致加载模型异常
                _info = deepcopy(info)
                _type: str = _info["type"]  # 当前模块的位置类型，目前三种:"entry"、"flow"、"head"
                _from: list = _info["from"]
                _module: str = _info["module"]
                _repeat: int = _info["repeat"]
                _allow_repeat: bool = _info.get("allow_repeat", False)  # 针对那些repeat次数为1，但希望能通过width进行repeat的模块
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
                    if _repeat > 1 or _allow_repeat:
                        _repeat = max(round(_repeat * depth), 1)

                # 用户可自定义模型解析方式
                self.custom_parse_model(level, _info)

                # 构造网络模块
                try:
                    layer = self._get_layer(_module)
                    layer.config_manager = self.config_manager
                except (NameError, AttributeError) as e:
                    raise ValueError(f"Failed to get module {_module}: {e}")

                layer = torch.nn.Sequential(*(layer(**_args) for _ in range(_repeat))) if _repeat > 1 else layer(**_args)
                layers.append(layer)

                record_list.append({
                    "type": _type,  # type指明当前的模块类型
                    "module": _module,  # module指明当前模块类
                    "layer": level,  # layer指明当前是第几层
                    "from": _from,  # from指明接受第几层的输入
                    "repeat": _repeat,  # repeat指明当前模块重复次数
                })

                ask_set.update([f for f in _from if f != -1])

                log_info.append(_info)
            except Exception as e:
                LOGGER.error(f"parse_model error: {e}, in layer:{level}.")
                raise e

        return layers, record_list, ask_set, log_info

    @staticmethod
    def _get_layer(module_str: str):
        import importlib
        module_str = module_str.strip()

        # 1. 完整包路径
        if "." in module_str:
            *pkg_parts, cls_name = module_str.split(".")
            pkg = ".".join(pkg_parts)
            try:
                mod = importlib.import_module(pkg)
                return getattr(mod, cls_name)
            except (ModuleNotFoundError, AttributeError):
                pass

        # 2. 候选包搜索（保持与之前一致）
        candidate_pkgs = [
            "torch.nn",
            "torchvision.ops",
            "transformers",
        ]
        for pkg in candidate_pkgs:
            try:
                mod = importlib.import_module(pkg)
                if hasattr(mod, module_str):
                    return getattr(mod, module_str)
            except ModuleNotFoundError:
                continue

        # 3. ⭐ 查全局注册表 ⭐
        if module_str in TTModuleRegistry.MODULE_REGISTRY:
            return TTModuleRegistry.get(module_str)

        raise ValueError(
            f"Unrecognized module string '{module_str}'. "
            f"Please check spelling, add candidate package, or use @register_module."
        )

    def _model_log(self):
        """
        打印模型结构摘要到日志与终端。
        """

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
        depth = scale_info["depth"]
        LOGGER.info(f"start parse model...")
        _struct_info = f"current model scale:{scale}," + f" depth:{depth}"
        print(_struct_info)
        print(f"{self.__class__.__name__} struct:")
        print(
            f"|{'layer':^{align_len['layer']}}"
            f"|{'type':^{align_len['type']}}"
            f"|{'repeat':^{align_len['repeat']}}"
            f"|{'from':^{align_len['from']}}"
            f"|{'module':^{align_len['module']}}"
            f"|{'args':^{align_len['args']}}"
            f"|"
        )
        for layer, info in enumerate(self.log_info):
            _repeat = max(round(info["repeat"] * depth), 1) if info["repeat"] > 1 else info["repeat"]
            print(
                f"|{layer: ^{align_len['layer']}}"
                f"|{info['type']:^{align_len['type']}}"
                f"|{_repeat:^{align_len['repeat']}}"
                f"|{str(info['from']):^{align_len['from']}}"
                f"|{info['module']:^{align_len['module']}}"
                f"|{str(info.get('args', {})):<{align_len['args']}}"
                f"|"
            )

        model_name = self.config_manager.model["name"]
        print(f"{model_name} model summary: {scale_info['summary']}\n")

    # ------------------------------------------------------------------
    # 5. 权重参数加载
    # ------------------------------------------------------------------
    def load_model_state_dict(self, state_dict, force_load=True):
        """
        加载权重，支持强制匹配或宽松匹配。

        Args:
            state_dict (dict[str, torch.Tensor]): 待加载的权重字典。
            force_load (bool, optional): True 时要求形状完全匹配，否则跳过；False 时抛出异常。默认 True。

        Raises:
            KeyError: force_load=False 且形状不匹配时抛出。
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
