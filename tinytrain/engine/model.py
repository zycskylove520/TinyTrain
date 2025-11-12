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

from __future__ import annotations

import torch

from torch import nn
from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

from tinytrain.cfg import TTConfigManager, TTModuleRegistry
from tinytrain.data.data_format import BaseBatchDataInfo
from tinytrain.global_var import RANK
from tinytrain.utils import LOGGER

if TYPE_CHECKING:
    from tinytrain.loss.base import TTBaseLoss


class TTBaseModel(nn.Module):
    """
    TinyTrain 框架中所有模型的抽象基类，承担统一建模规范、训练-推理流程控制、权重管理与初始化策略定义的核心职责。

    本类通过定义标准化接口，将模型结构定义、损失函数构造、前向传播行为、权重加载与初始化等关键能力解耦，使得子类仅需关注任务相关的网络结构实现与损失定义，即可自动获得完整的训练与推理能力。

    主要设计目标包括：
    - 提供统一的 forward 入口，根据输入类型自动区分训练与推理模式；
    - 规范损失函数初始化流程，确保与模型结构解耦；
    - 支持非严格权重加载机制，实现跨模型结构迁移与微调；
    - 内置通用权重初始化策略，覆盖卷积、批归一化与激活函数等常用模块；
    - 支持激活函数原地（inplace）化，降低显存占用。

    子类必须实现以下抽象方法：
    - `init_criterion()`: 返回任务特定的损失函数实例；
    - `inference()`: 定义模型在推理模式下的前向传播逻辑。

    本类不建议被直接实例化，仅应作为所有任务模型的公共父类使用。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: TTConfigManager, device: torch.device = None):
        """
        初始化模型。

        Args:
            config_manager (TTConfigManager): 配置管理器，包含模型结构、超参数、设备等信息。
            device (torch.device | None): 目标设备，仅做保存，不参与内部逻辑。
        """
        super(TTBaseModel, self).__init__()

        self.config_manager = config_manager
        self.device = device

        self.criterion = None

    # ------------------------------------------------------------------
    # 2. 子类必须实现的钩子
    # ------------------------------------------------------------------
    def init_criterion(self) -> TTBaseLoss:
        """
        初始化任务特定的损失函数。

        Returns:
            TTBaseLoss: 损失模块。

        Raises:
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError("init_criterion() needs to be implemented!")

    def inference(self, *args, **kwargs):
        """
        推理模式前向传播。

        Args:
            *args, **kwargs: 子类自定义输入。

        Returns:
            任意类型: 模型推理输出。

        Raises:
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError("inference() needs to be implemented!")

    # ------------------------------------------------------------------
    # 3. 统一前向入口（不建议重写）
    # ------------------------------------------------------------------
    def forward(self, data: BaseBatchDataInfo | torch.Tensor | list[torch.Tensor] | Any | list[Any]):
        """
        统一入口：根据输入类型自动选择推理或训练模式。

        Args:
            data:
                - BaseBatchDataInfo：训练/验证模式，计算 loss。
                - torch.Tensor | list[torch.Tensor] | Any | list[Any]：推理模式，执行 inference函数。如果为torch.Tensor|Any，则表示模型单输入，如果为list[torch.Tensor]|list[Any]，表示模型多输入

        Returns:
            Union[list[torch.Tensor], tuple]: 推理输出或 (loss, loss_items)。
        """

        if isinstance(data, BaseBatchDataInfo):
            outputs = self.inference(data.data)
            return self.loss(outputs, data)
        else:
            return self.inference(data)

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

    # ------------------------------------------------------------------
    # 4. 权重管理
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

    # ------------------------------------------------------------------
    # 5. 权重初始化
    # ------------------------------------------------------------------
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


class TTConfigModel(TTBaseModel):
    """
    面向配置文件驱动的结构化模型构建基类，支持通过配置描述文件（TOML/YAML）动态解析并构建深度学习模型，适用于绝大多数计算机视觉与自然语言处理任务。

    本类继承自 `TTBaseModel`，在保留其统一训练-推理流程与权重管理能力的基础上，进一步引入配置化建模能力。模型结构完全由 `TTConfigManager` 提供的配置数据驱动，无需硬编码网络定义，从而实现模型结构的声明式描述与高度可复现性。

    核心职责包括：
    - 解析配置中定义的 entry、flow、head 三类模块，构建有向无环计算图；
    - 支持深度增益（depth gain）自动缩放机制，实现模型复杂度的灵活调整；
    - 提供模块级缓存机制，确保多分支结构下的中间结果正确复用；
    - 支持自定义模块解析钩子（`custom_parse_model_level`）与结构修改钩子（`custom_modify_model`），允许子类在不破坏配置完整性的前提下干预模型构建过程；
    - 内置模块动态加载机制，支持从 `torch.nn`、`torchvision.ops`、`transformers` 等标准库或用户注册表中加载模块类；
    - 提供结构化日志输出，便于开发者审查模型构建结果与模块参数。

    配置语法要求：
    - 第 0 层必须为 entry 类型，且其输入来源必须为 `[-1]`；
    - 所有层必须声明其输入来源（`from` 字段），支持多输入融合；
    - head 类型模块为模型输出节点，至少需存在一个；
    - 支持 `repeat` 字段定义模块重复次数，并结合 `depth_gain` 自动缩放；
    - 支持 `allow_repeat` 字段允许宽度维度上的重复扩展。

    本类适用于需要高度可配置、可扩展、可复现的建模场景，尤其适合科研实验、模型结构搜索与多任务统一训练框架。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: TTConfigManager, device: torch.device = None):
        """
        初始化模型。

        Args:
            config_manager (TTConfigManager): 配置管理器，包含模型结构、超参数、设备等信息。
            device (torch.device | None): 目标设备，仅做保存，不参与内部逻辑。
        """
        super().__init__(config_manager, device)
        self.DEPTH_GAIN = None  # 深度增益

        self.module_list, self.record_list, self.ask_set, self.log_info = self.parse_model()

        # 只有第一次启动时打印模型信息
        if RANK in {-1, 0}:
            self._model_log()

    # ------------------------------------------------------------------
    # 2. 子类可重写钩子
    # ------------------------------------------------------------------
    def custom_modify_model(self, network):
        """
        钩子：子类可重写以动态修改整个模型结构，包括增加或删减模块，以及修改模块参数等操作。
        注意，这是一个危险的操作，除非你知道你在做什么。

        Args:
            network (dict): 整个模型结构的字典。
        """
        pass

    def custom_parse_model_level(self, level, module_info):
        """
        钩子：子类可重写以动态修改模块每一层的参数配置。

        Args:
            level (int): 当前模块属于第几层
            module_info (dict): 当前模块的配置字典。
        """
        pass

    # ------------------------------------------------------------------
    # 3. 统一前向入口（不建议重写）
    # ------------------------------------------------------------------
    def inference(self, data: torch.Tensor | list[torch.Tensor] | Any | list[Any]) -> list[torch.Tensor]:
        """
        推理模式前向传播。

        Args:
            data (torch.Tensor | list[torch.Tensor] | Any | list[Any]): 模型输入。
                单输入时可直接传入 ``torch.Tensor | Any``；
                多输入时请传入 ``list[torch.Tensor] | list[Any]``，列表顺序需与模型配置中 ``entry`` 层的顺序一一对应。

        Returns:
            list[torch.Tensor]: 模型输出列表（每个 head 对应一个输出）。

        Raises:
            RuntimeError: 如果未检测到任何输出。
        """
        # 检查输入是否为多输入（list 或 tuple）
        if not isinstance(data, (list, tuple)):
            inputs = [data]
        else:
            inputs = data
        outputs: list = []  # 存放模型推理最终的输出

        inputs_idx = 0
        current_data = None

        for i, layer in enumerate(self.module_list):
            try:
                record_info = self.record_list[i]
                module_type: str = record_info["type"]
                frm: list[int] = record_info["from"]
                num_from: int = len(frm)

                # entry层特殊部分
                if module_type == "entry":
                    if num_from == 1:
                        current_data = inputs[inputs_idx]
                    else:
                        current_data = tuple(inputs[inputs_idx] if rf == -1 else self.record_list[rf]["data"] for rf in frm)
                    inputs_idx += 1
                elif module_type == "flow" or module_type == "head":
                    if num_from == 1:
                        rf = frm[0]
                        if rf != -1:
                            current_data = self.record_list[rf]["data"]
                    else:
                        current_data = tuple(current_data if rf == -1 else self.record_list[rf]["data"] for rf in frm)

                current_data = layer(current_data)

                if module_type == "head":
                    outputs.append(current_data)

                if i in self.ask_set:
                    self.record_list[i]["data"] = current_data
            except Exception as e:
                LOGGER.error(f"inference error: {e}, in layer: {i}.")
                raise

        if len(outputs) == 0:
            raise RuntimeError("Model no output, 'model.toml' need head module!")
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

        # deepcopy防止修改原始配置文件导致加载模型异常
        network = deepcopy(self.config_manager.model["network"])
        for level, info in enumerate(network):
            try:
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
                assert _repeat > 0, f"level_{level}: {_module} 'repeat' must greater than 0!"
                assert not ask_set or level > max(ask_set), f"layer_{level} depends on future layer (max dependency: {max(ask_set)})."

                # 限制第0层必须为entry层,且只能第0层为entry层
                if level == 0:
                    assert _type == "entry", f"level_0: {_module} 'type' must be 'entry'"
                    if not (len(_from) == 1 and _from[0] == -1):
                        LOGGER.warning(f"level_0 'from' is not [-1], auto-correct to [-1].")
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

                # 用户可自定义每一层的模型参数解析方式
                self.custom_parse_model_level(level, _info)

                # depth gain
                # 如果custom_parse_model_level函数修改了repeat或allow_repeat，需要重新获取
                _repeat = _info["repeat"]
                _allow_repeat = _info.get("allow_repeat", False)
                if _repeat > 1 or _allow_repeat:
                    _repeat = max(round(_repeat * depth), 1)

                # 构造网络模块
                _layer = self.get_layer(_module)
                layer = nn.Sequential(*(_layer(**_args) for _ in range(_repeat))) if _repeat > 1 else _layer(**_args)
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
        return layers, record_list, sorted(ask_set), log_info

    @staticmethod
    def get_layer(module_str: str):
        """
        根据字符串名称动态导入并返回对应的 **模块类**（而非实例）。

        查找顺序（一旦匹配立即返回，不再继续）：
        1. 完整包路径：如 "torch.nn.Conv2d" → 直接 import torch.nn 并返回 Conv2d 类。
        2. 候选包搜索：依次在 ["torch.nn", "torchvision.ops", "transformers"] 等中查找同名类。
        3. 全局注册表：查询用户通过 `@register_module` 注册的自定义模块。

        Args:
            module_str (str): 模块名称，支持简写（"Conv2d"）或完整路径（"torch.nn.Conv2d"）。

        Returns:
            type: 对应的 **类对象**，可用于后续实例化。

        Raises:
            ValueError: 所有查找路径均未命中时抛出，提示检查拼写、补充候选包或使用 `@register_module` 注册。

        Examples:
            >>> TTConfigModel.get_layer("ReLU")           # 返回 torch.nn.ReLU
            >>> TTConfigModel.get_layer("my_pkg.MyBlock") # 返回自定义包中的 MyBlock
            >>> TTConfigModel.get_layer("CustomBlock")    # 返回 @register_module 注册的 CustomBlock
        """
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
            # 自定义继续添加更多第三方候选包
        ]
        for pkg in candidate_pkgs:
            try:
                mod = importlib.import_module(pkg)
                if hasattr(mod, module_str):
                    return getattr(mod, module_str)
            except ModuleNotFoundError:
                continue

        # 3. ⭐ 查全局注册表 ⭐
        return TTModuleRegistry.get(module_str)

    def _model_log(self):
        """
        打印模型结构摘要到日志与终端。

        输出示例
        --------
        |layer|type|repeat|from|module|args|
        |----|----|----|----|----|----|
        | 0  |entry| 1  |[-1]|Conv|{'k':6,'s':2,'p':2}|
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
        model_name = self.config_manager.model.get("name", "")
        _struct_info = f"{model_name} model scale:{scale}, depth:{depth}, struct:"
        print(_struct_info)
        # print(f"{self.__class__.__name__} struct:")
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
        print(f"model summary: {scale_info['summary']}\n")


class TTEasyModel(TTBaseModel):
    """
    极简手动建模基类，专为无需配置、快速原型开发、教学演示或高度定制化模型结构设计的轻量级建模入口。

    本类继承自 `TTBaseModel`，在保留其统一前向传播、权重加载与初始化能力的前提下，移除配置驱动的模型构建逻辑，允许开发者以最直接的方式定义模型结构。子类仅需在 `setup_model()` 方法中返回一个标准的 `torch.nn.Module` 实例，即可自动获得完整的训练与推理能力。

    核心设计目标包括：
    - 最小化建模开销，适用于脚本式开发、单元测试、算法验证等场景；
    - 完全兼容 `TTBaseModel` 的权重管理、初始化和训练流程；
    - 不引入任何配置依赖，模型结构由开发者显式定义，确保最大灵活性；
    - 支持任意复杂度的模型结构，包括多输入、多输出、动态图等高级特性。

    使用方式：
    - 子类继承 `TTEasyModel`；
    - 实现 `setup_model()` 方法，返回一个 `nn.Module` 实例；
    - 实现 `init_criterion()` 定义损失函数；
    - 模型将自动具备训练、推理、权重加载与初始化能力。

    本类适用于以下场景：
    - 快速验证算法可行性；
    - 构建无法通过配置描述的复杂结构；
    - 教学或演示目的，降低框架使用门槛；
    - 需要完全控制模型构建过程的高级开发者。

    注意：本类不提供结构化日志或配置解析能力，模型结构完全由开发者负责维护。
    """

    # ------------------------------------------------------------------
    # 1. 构造与入口
    # ------------------------------------------------------------------
    def __init__(self, config_manager: TTConfigManager, device: torch.device = None):
        """
        初始化模型。

        Args:
            config_manager (TTConfigManager | None): 预留参数，始终可传 None。
            device (torch.device | None): 目标设备，仅做保存，不参与内部逻辑。
        """
        super().__init__(config_manager, device)

        self.model = self.setup_model()

    # ------------------------------------------------------------------
    # 2. 子类必须实现的钩子
    # ------------------------------------------------------------------
    def setup_model(self) -> nn.Module:
        """
        子类在此返回一个现成的 nn.Module，作为推理核心。

        Returns:
            nn.Module: 任意 PyTorch 模型。

        Raises:
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError("set_model() needs to be implemented!")

    # ------------------------------------------------------------------
    # 3. 统一前向入口（不建议重写）
    # ------------------------------------------------------------------
    def inference(self, data: Any) -> Any:
        """
        推理模式：直接调用内部模型。

        Args:
            data (Any): 任意输入，会被原样传给 `self.model`。

        Returns:
            Any: `self.model(data)` 的返回值。
        """
        return self.model(data)
