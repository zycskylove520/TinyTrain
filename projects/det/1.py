import torch

from tinytrain.modules import ResNetLayer

if __name__ == '__main__':
    x = torch.randn(1, 64, 640, 640)
    layer = ResNetLayer(64, 256, n=2)
    y = layer(x)
    print(y.shape)