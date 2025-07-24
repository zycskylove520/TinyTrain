import torch
from torch import nn, Tensor

from TinyTrain.modules import Conv, DWConv
from TinyTrain.utils.tal import make_anchors, dist2bbox


class TestOps(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: list[Tensor]) -> Tensor:
        return torch.sum(torch.stack(x, dim=0), dim=0)


class TestYOLODetect(nn.Module):
    """YOLO detection head."""
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc, from_channels: list):
        """
        @param nc: 输出类别个数
        @param from_channels: 接受的输入通道列表
        """
        super().__init__()
        self.nc = nc
        self.from_channels = from_channels
        # self.stride = torch.zeros(len(from_channels))
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

        c2, c3 = max((16, from_channels[0] // 4, self.reg_max * 4)), max(from_channels[0], min(self.nc, 100))  # channels

        # cv2用于DFLoss和IOULoss计算
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in from_channels
        )
        # cv3用于类别损失计算
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in from_channels
        )


    def forward(self, x: list[torch.Tensor]):
        """
        训练模式返回的shape为：[batch,
        """
        for i, (cv2_module, cv3_module) in enumerate(zip(self.cv2, self.cv3)):
            x[i] = torch.cat((cv2_module(x[i]), cv3_module(x[i])), 1)

        if self.training:
            return x
        x = self.inference(x)
        return x

    def inference(self, x: list[torch.Tensor]):
        # inference模式下，bboxes已解码

        # 解码锚框,self.stride是在构建模型时添加的属性，详见TinyTrain\models\yolo\detect\model中的__init__函数
        self.anchors, self.strides = make_anchors(x, self.stride)

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        box = box.permute(0, 2, 1).contiguous()
        cls = cls.permute(0, 2, 1).contiguous()
        b, a, c = box.shape  # batch, anchors, channels
        # [batch, num_anchors, 4]
        pred_dist = box.reshape(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(box.dtype).to(box.device))
        temp = dist2bbox(pred_dist, self.anchors, xywh=True, dim=-1)
        decode_box = temp * self.strides.repeat(1, 1, 2)  # [batch, num_anchors, 4]
        # # decode_box为cxcywh格式
        # return torch.cat((decode_box, cls.sigmoid()), -1)