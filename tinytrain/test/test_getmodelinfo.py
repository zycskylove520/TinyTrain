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

from ptflops import get_model_complexity_info


def get_any_model_info(model: nn.Module,
                       input_shape=(3, 224, 224),
                       print_table=False):
    """
    返回一个 dict，包含：
        layers       : int   – 网络总层数（可训练+不可训练）
        parameters   : int   – 总参数量
        gradients    : int   – 需要梯度的参数量
        gflops       : float – 前向 GFLOPs
    """
    # 1. 层数
    layers = sum(1 for _ in model.modules())

    # 2. 参数量 & 梯度量
    params_total = sum(p.numel() for p in model.parameters())
    params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. FLOPs
    macs, _ = get_model_complexity_info(
        model, input_shape, as_strings=False, verbose=print_table
    )
    gflops = macs / 1e9 * 2  # 1 MAC ≈ 2 FLOPs

    return dict(
        layers=int(layers),
        parameters=int(params_total),
        gradients=int(params_train),
        gflops=round(gflops, 2)
    )


def get_tt_model_info(core, scale='nsmlx', input_shape=(3, 224, 224)):
    info_dict = {}
    for i in scale:
        model = core.get_model(model_scale=i)
        info = get_any_model_info(model, input_shape=input_shape)
        info_dict[i] = f"{info['layers']} layers, {info['parameters']} parameters, {info['gradients']} gradients, {info['gflops']} GFLOPs"

    print("summary: \n")
    for k, v in info_dict.items():
        print(f"{k}: {v}")


# ==================================================================
# 下面开始 pytest 用例
# ==================================================================
try:
    import pytest
except ImportError as e:
    raise ImportError(
        "TinyTrain 的测试依赖 pytest，请先安装：\n"
        "  pip install -U pytest\n"
    )


# ------------------------------------------------------------------
# 1. 获取任意模型信息
# ------------------------------------------------------------------
def test_get_any_model_info():
    from torchvision.models import resnet18

    net = resnet18(pretrained=False)
    info = get_any_model_info(net, input_shape=(3, 224, 224))
    print(f"{info['layers']} layers, {info['parameters']} parameters, {info['gradients']} gradients, {info['gflops']} GFLOPs")


def test_get_tt_model_info():
    import os

    from tinytrain.models.yolo import YOLOCore
    from tinytrain import ROOT

    os.chdir(ROOT.parent / "examples/cls")  # Tinytrain/examples/cls

    yolo = YOLOCore(link_file="link.toml")
    get_tt_model_info(yolo, scale="nsmlx", input_shape=(3, 640, 640))


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
