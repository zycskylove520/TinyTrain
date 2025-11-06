"""
YOLOModel
=========
TinyTrain 框架中 YOLO 检测、分割、关键点等任务的通用基类实现。
"""

from copy import deepcopy
from torch import nn

from tinytrain.cfg import TTConfigManager
from tinytrain.engine import TTConfigModel
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import make_divisible
from tinytrain.utils.checks import check_img_size


class YOLOModel(TTConfigModel):
    """
    YOLOModel
    ~~~~~~~

    YOLO 系列检测/分割/关键点模型的 **通用基类**，承担以下职责：

    1. 根据 ``model.toml`` 动态解析网络结构（entry / flow / head）。
    2. 在解析阶段完成 **深度增益** (depth gain) 与 **宽度增益** (width gain) 缩放。
    3. 提供统一的 ``init_criterion`` / ``loss`` / ``forward`` 接口，子类仅需实现任务损失。
    4. 内置 stride 自动推算、权重初始化、结构摘要打印等公共能力。

    设计要点
    --------
    - 结构配置完全由 TTConfigManager 驱动，**零硬编码**。
    - 宽度增益统一调用 ``make_divisible()``，保证通道数为 8 / 16 的倍数，兼容 TensorRT。
    - 深度增益仅对 **in_channels == out_channels** 的模块生效，避免上采样 / 降采样层被误缩放。
    - 第 0 层强制为 entry 且 ``from = [-1]``，杜绝非法依赖。
    - 支持自定义解析钩子 ``custom_parse_model_level()``，子类可原地修改 ``module_info``。
    - 所有模块延迟实例化，先校验再构建，**配置错误立即抛异常**。
    """

    def __init__(self, config_manager: TTConfigManager, device):
        """
        初始化 YOLO 基类。

        步骤
        ----
        1. 提前占位 ``WIDTH_GAIN``，供后续解析阶段写入。
        2. 调用父类 ``TTBaseModel.__init__()``，触发 ``parse_model()`` 完成网络构建。
        3. 子类（检测/分割/关键点）可在 ``super()`` 之后继续初始化 stride、loss 等。

        Args
        ----
        config_manager : TTConfigManager
            必须包含 model.scales / model.network / dataset.nc 等字段。
        device : torch.device
            模型所在设备。
        """
        self.WIDTH_GAIN = None  # 宽度增益
        super().__init__(config_manager=config_manager, device=device)

        # 调整输入图片尺寸，以满足YOLO要求
        self.config_manager.dataset["img_size"] = check_img_size(self.config_manager.dataset["img_size"])

    def parse_model(self):
        """
        根据配置描述文件，构建 YOLO 网络。

        步骤
        ----
        1. 读取 scales 配置，计算并缓存 width / depth gain。
        2. 逐层解析 network，完成合法性校验、增益缩放、模块实例化。
        3. 生成 `layers`、`record_list`、`ask_set`、`log_info` 四元组，
           供后续 ForwardGraph 或手动特征融合使用。

        参数
        ----
        config_manager : TTConfigManager
            包含 model / dataset / augment 等完整配置。

        返回
        ----
        layers : nn.ModuleList
            按顺序排列的 PyTorch 层或子网络。
        record_list : list[dict]
            每层详细描述，便于特征路由与调试。
        ask_set : set[int]
            所有被其他层引用的层索引，用于剪枝或可视化。
        log_info : list[dict]
            解析后用于打印或序列化的最终配置。
        """

        scale_info = self.config_manager.model["scales"][self.config_manager.model["scale"]]

        # 计算一次并写入类变量（所有实例共享）
        self.WIDTH_GAIN = scale_info["width"]
        self.DEPTH_GAIN = scale_info["depth"]

        layers, record_list, log_info = nn.ModuleList(), [], []
        ask_set = set()

        # 直接读取类变量
        width = self.WIDTH_GAIN
        depth = self.DEPTH_GAIN

        for level, info in enumerate(self.config_manager.model["network"]):
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
            assert not ask_set or level > max(ask_set), f"layer_{level} depends on future layer (max dependency: {max(ask_set)})."

            # 限制第0层必须为entry层，且entry层只能出现在第0层
            if level == 0:
                assert _type == "entry", f"level_0: {_module} 'type' must be 'entry'"
                if not (len(_from) == 1 and _from[0] == -1):
                    LOGGER.warning(f"level_0 'from' is not [-1], auto-correct to [-1].")
                    _from = [-1]

            if _type == "entry":  # entry层
                assert -1 in _from, f"entry layer_{level} must depend on from index: -1."
                # assert "in_channels" in _args, f"level_{level}: {_module} 'in_channels' must exist!"
                # set model entry channels
                if _args.get("in_channels", None) is not None:
                    if _args["in_channels"] == -1:
                        entry_channels = self.config_manager.model["entry_channels"]
                        assert entry_channels is not None, "entry level: if {_module} 'in_channels' == -1, must set model config key:entry_channels!"
                        _args["in_channels"] = entry_channels
            elif _type == "flow":  # flow层
                # assert "in_channels" in _args, f"level_{level}: {_module} 'in_channels' must exist!"
                # assert "out_channels" in _args, f"level_{level}: {_module} 'out_channels' must exist!"
                pass
            elif _type == "head":  # head层
                # assert "in_channels" in _args, f"level_{level}: {_module} 'in_channels' must exist!"
                if _args.get("nc", None) is not None:
                    _args["nc"] = self.config_manager.dataset["nc"]

            # 用户可自定义模型解析方式
            self.custom_parse_model_level(level, _info)

            # depth gain
            # 如果custom_parse_model函数修改了repeat或allow_repeat，需要重新获取
            _repeat = _info["repeat"]
            _allow_repeat = _info.get("allow_repeat", False)
            if _args.get("in_channels", None) is not None and _args.get("out_channels", None) is not None:
                if _args["in_channels"] == _args["out_channels"]:
                    _repeat = max(round(_repeat * depth), 1) if _repeat > 1 or _allow_repeat else _repeat
                else:
                    if _repeat > 1:
                        _repeat = 1
                        LOGGER.warning(f"level_{level}: {_module} not support 'repeat'! set repeat value to 1!")
            else:
                if _repeat > 1:
                    LOGGER.warning(f"'in_channels' or 'out_channels' not exist, set repeat value to {_repeat} maybe throw exception!")

            # width gain
            if _type == "entry":
                if _args.get("out_channels", None) is not None:
                    _args["out_channels"] = make_divisible(_args.get("out_channels") * width)
            elif _type == "flow":
                if _args.get("in_channels", None) is not None:
                    _args["in_channels"] = make_divisible(_args.get("in_channels") * width)
                if _args.get("out_channels", None) is not None:
                    _args["out_channels"] = make_divisible(_args.get("out_channels") * width)
            elif _type == "head":
                if _args.get("in_channels", None) is not None:
                    _args["in_channels"] = make_divisible(_args.get("in_channels") * width)

            # 构造网络模块
            _layer = self.get_layer(_module)
            layer = nn.Sequential(*(_layer(**_args) for _ in range(_repeat))) if _repeat > 1 else _layer(**_args)
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
        return layers, record_list, sorted(ask_set), log_info

    def _model_log(self):
        """
        以表格形式将网络结构打印到终端与日志文件。

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
        width = scale_info["width"]
        LOGGER.info(f"start parse model...")
        model_name = self.config_manager.model.get("name", "")
        _struct_info = f"{model_name} model scale:{scale}," + f" depth:{depth}" + ("," if width else f", width:{width},") + f" struct:"
        print(_struct_info)
        print(
            f"|{'layer':^{align_len['layer']}}"
            f"|{'type':^{align_len['type']}}"
            f"|{'repeat':^{align_len['repeat']}}"
            f"|{'from':^{align_len['from']}}"
            f"|{'module':^{align_len['module']}}"
            f"|{'args':^{align_len['args']}}|"
        )
        for layer, info in enumerate(self.log_info):
            _repeat = max(round(info["repeat"] * depth), 1) if info["repeat"] > 1 else info["repeat"]
            print(
                f"|{layer: ^{align_len['layer']}}"
                f"|{info['type']:^{align_len['type']}}"
                f"|{_repeat:^{align_len['repeat']}}"
                f"|{str(info['from']):^{align_len['from']}}"
                f"|{info['module']:^{align_len['module']}}"
                f"|{str(info.get('args', {})):<{align_len['args']}}|"
            )
        print(f"model summary: {scale_info['summary']}\n")
