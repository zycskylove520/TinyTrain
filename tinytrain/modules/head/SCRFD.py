from torch import nn


class SCRFDHead(nn.Module):
    def __init__(self, nc, num_in_channels, stacked_convs=4,):
        super(SCRFDHead, self).__init__()
