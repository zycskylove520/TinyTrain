from copy import deepcopy

from tinytrain.cfg.config_manager import ConfigManager
from tinytrain.engine import BaseModel
from tinytrain.modules import *
from tinytrain.utils import LOGGER
from tinytrain.utils.any_utils import make_divisible


class YOLOModel(BaseModel):
    # YOLO模型专属类变量
    WIDTH_GAIN = None  # 宽度增益

    def parse_model(self, config_manager: ConfigManager):
        """
        重写解析YOLO专用适配模型。
        :param config_manager:
        :return:
        """

        scale_info = config_manager.model["scales"][config_manager.model["scale"]]

        # 计算一次并写入类变量（所有实例共享）
        YOLOModel.WIDTH_GAIN = scale_info["width"]
        YOLOModel.DEPTH_GAIN = scale_info["depth"]

        layers, record_list, log_info = nn.ModuleList(), [], []
        ask_set = set()

        # 直接读取类变量
        width = YOLOModel.WIDTH_GAIN
        depth = YOLOModel.DEPTH_GAIN

        for level, info in enumerate(config_manager.model["network"]):
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
                        entry_channels = config_manager.model["entry_channels"]
                        assert entry_channels is not None, "entry level: if {_module} 'in_channels' == -1, must set model config key:entry_channels!"
                        _args["in_channels"] = entry_channels
            elif _type == "flow":  # flow层
                # assert "in_channels" in _args, f"level_{level}: {_module} 'in_channels' must exist!"
                # assert "out_channels" in _args, f"level_{level}: {_module} 'out_channels' must exist!"
                pass
            elif _type == "head":  # head层
                # assert "in_channels" in _args, f"level_{level}: {_module} 'in_channels' must exist!"
                if _args.get("nc", None) is not None:
                    _args["nc"] = config_manager.dataset["nc"]

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

        return layers, record_list, ask_set, log_info
