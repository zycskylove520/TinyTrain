import torch.nn as nn
from ptflops import get_model_complexity_info

def get_model_info(model: nn.Module,
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


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    from torchvision.models import resnet18
    net = resnet18(pretrained=False)
    info = get_model_info(net, input_shape=(3, 224, 224))
    print(f"{info['layers']} layers, {info['parameters']} parameters, {info['gradients']} gradients, {info['gflops']} GFLOPs")