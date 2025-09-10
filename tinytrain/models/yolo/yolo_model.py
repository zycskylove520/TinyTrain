"""
YOLOModel
=========
YOLO 系列检测 / 分类模型的统一解析器。
在 TinyTrain 框架中，所有 YOLO 变体（v5 / v7 / v8 / NAS …）均可通过
同一套配置语法描述网络结构，并由本类自动完成深度、宽度缩放与模块实例化。

核心职责
--------
1. 依据模型规模（n/s/m/l/x）自动计算 depth / width 增益并一次性写入类变量，
   确保所有层共享同一缩放因子。
2. 逐层校验网络描述合法性（entry→flow→head 的顺序、from / repeat 约束）。
3. 对 entry / flow / head 中 `in_channels / out_channels / nc` 等关键字段
   执行 width-gain 调整，保证输出通道数为硬件对齐值（8 的倍数）。
4. 支持用户通过 `custom_parse_model` 钩子插入私有解析逻辑，实现零侵入扩展。
5. 自动生成 `layers / record_list / ask_set / log_info` 四元组，
   供 Backbone、Neck、Head 按需索引与特征融合。

配置约定
--------
network:
  - type: entry      # 第 0 层必须为 entry，且仅出现一次
    module: Conv     # 模块类名，必须能在 tinytrain.modules 中找到
    from: [-1]       # -1 表示「来自上一层」
    repeat: 1        # 重复次数，flow 层受 depth 增益影响
    args:
      in_channels: 3
      out_channels: 64
  - type: flow
    module: C2f
    from: [-1]
    repeat: 6
    args:
      in_channels: 64
      out_channels: 64
  - type: head
    module: Detect
    from: [15, 18, 21]
    repeat: 1
    args:
      nc: -1          # 自动替换为 dataset.nc
"""

from copy import deepcopy
from torch import nn

from tinytrain.cfg import ConfigManager
from tinytrain.engine import BaseModel
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import make_divisible


class YOLOModel(BaseModel):
    # YOLO模型专属类变量
    def __init__(self, config_manager: ConfigManager, device, *args, **kwargs):
        self.WIDTH_GAIN = None  # 宽度增益
        super().__init__(config_manager=config_manager, device=device, *args, **kwargs)

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
        config_manager : ConfigManager
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
            _args: dict = _info.get("args", {})

            # check network
            assert _type in {"entry", "flow", "head"}, f"level_{level}: {_module} 'type' must be 'entry', 'flow', or 'head'"
            assert len(_from) > 0, f"level_{level}: {_module} 'from' list length must greater than 0!"
            assert _repeat > 0, "level_{level}: {_module} 'repeat' must greater than 0!"

            # 限制第0层必须为entry层，且entry层只能出现在第0层
            if level == 0:
                assert _type == "entry", f"level_0: {_module} 'type' must be 'entry'"
                _from = [-1]
            else:
                assert _type != "entry", f"level_{level}: {_module} 'type' cannot be 'entry'"

            if _type == "entry":  # entry层
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

            # depth gain
            if _type == "flow":
                if _args.get("in_channels", None) is not None and _args.get("out_channels", None) is not None:
                    if _args["in_channels"] == _args["out_channels"]:
                        _repeat = max(round(_repeat * depth), 1) if _repeat > 1 else _repeat
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

            # 用户可自定义模型解析方式
            self.custom_parse_model(_info)

            # 构造网络模块
            try:
                layer = self._get_layer(_module)
                layer.config_manager = self.config_manager
            except (NameError, AttributeError) as e:
                raise ValueError(f"Failed to get module {_module}: {e}")

            layer = nn.Sequential(*(layer(**_args) for _ in range(_repeat))) if _repeat > 1 else layer(**_args)
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

        return layers, record_list, ask_set, log_info

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
        width = scale_info["width"]
        LOGGER.info(f"start parse model...")
        _struct_info = f"current model scale:{scale}," + f" depth:{depth}" + ("." if width else f", width:{width}")
        print(_struct_info)
        print(f"{self.__class__.__name__} struct:")
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