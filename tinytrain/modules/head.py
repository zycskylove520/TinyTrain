import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Conv2Linear, Proto
from .conv import CBA, DWConv
from tinytrain.cfg import TTModuleRegistry
from tinytrain.utils.tal import dist2bbox
from tinytrain.utils.box_utils import make_anchors


@TTModuleRegistry.register
class Classify(nn.Module):
    """
    通用分类头。
    training模型下返回: [batch, nc]
    eval模式下返回: [batch, nc], nc已做softmax
    """

    def __init__(self, in_channels, nc, hidden_channels=1280, kernel_size=1, stride=1, padding=None, groups=1):
        """Initializes classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        self.conv = CBA(in_channels, hidden_channels, kernel_size, stride, padding, groups)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(batch, hidden_channels, 1, 1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(hidden_channels, nc)  # to x(batch, out_channels)

    def forward(self, x):
        """Performs a forward pass of the YOLO model_config on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

        if not self.training:
            x = self.inference(x)
        return x

    def inference(self, x):
        x = F.softmax(x, dim=1)
        return x


@TTModuleRegistry.register
class GDC(nn.Module):
    """
    MobileFaceNet 人脸识别头。
    training模型下返回: [batch, embedding_size]
    eval模式下返回: [batch, embedding_size], embedding_size已做L2正则化
    """

    def __init__(self, in_channels, embedding_size):
        super().__init__()
        self.layers = nn.Sequential(
            CBA(in_channels=in_channels, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            CBA(in_channels=512, out_channels=512, groups=512, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), act=False),
            nn.Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        x = self.layers(x)

        if not self.training:
            x = F.normalize(x, p=2, dim=-1)
        return x


@TTModuleRegistry.register
class GeneralFace(nn.Module):
    """
        通用人脸识别头。
        training模型下返回: [batch, embedding_size]
        eval模式下返回: [batch, embedding_size], embedding_size已做L2正则化
        """

    def __init__(self, in_channels, embedding_size):
        super().__init__()
        self.c2l = Conv2Linear(in_channels=in_channels, out_channels=embedding_size, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.c2l(x)

        if not self.training:
            x = F.normalize(x, p=2, dim=-1)
        return x


@TTModuleRegistry.register
class YOLODetect(nn.Module):
    """
    YOLO算法通用检测头。
    training模型下返回: [batch, 4*16+nc, h, w]
    eval模式下返回: [batch, num_anchors, 4+nc], bboxes已解码成推理时传入的图片尺寸
    """

    def __init__(self, nc, from_channels: list, reg_max=16):
        """
        @param nc: 输出类别个数
        @param from_channels: 接受的输入通道列表
        """
        super().__init__()
        self.nc = nc
        self.from_channels = from_channels
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

        c2, c3 = max((16, from_channels[0] // 4, self.reg_max * 4)), max(from_channels[0], min(self.nc, 100))  # channels

        # cv2用于DFLoss和IOULoss计算
        self.cv2 = nn.ModuleList(
            nn.Sequential(CBA(x, c2, 3), CBA(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in from_channels
        )
        # cv3用于类别损失计算
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), CBA(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), CBA(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in from_channels
        )

        # self.bias_init()

    def forward(self, x: list[torch.Tensor]):
        """
        训练模式返回的shape为：[batch, 4*16+nc, h, w]
        """
        for i, (cv2_module, cv3_module) in enumerate(zip(self.cv2, self.cv3)):
            x[i] = torch.cat((cv2_module(x[i]), cv3_module(x[i])), 1)

        if self.training:
            return x
        x = self.inference(x)
        return x

    def inference(self, x: list[torch.Tensor]):
        # inference模式下，bboxes已解码,返回的shape为：[batch, num_anchors, 4+nc]

        # 解码锚框,self.stride是在构建模型时添加的属性，详见TinyTrain\models\yolo\detect\model中的__init__函数
        anchors, strides = make_anchors(x, self.strides)

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        box = box.permute(0, 2, 1).contiguous()
        cls = cls.permute(0, 2, 1).contiguous()
        b, a, c = box.shape  # batch, anchors, channels
        # [batch, num_anchors, 4]
        pred_dist = box.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.view(-1, 1).to(box.device)).squeeze(-1)
        decode_box = dist2bbox(pred_dist, anchors, xywh=True, dim=-1) * strides  # [batch, num_anchors, 4]
        # decode_box为cxcywh格式
        return torch.cat((decode_box, cls.sigmoid()), -1)

    def bias_init(self, strides):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # 如果不做bias初始化，输出的cls loss会非常大，这会导致反向传播让权重参数迅速归0的问题
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(self.cv2, self.cv3, strides):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


@TTModuleRegistry.register
class YOLOPose(YOLODetect):
    """
    YOLO通用姿态估计头。
    training模型下返回: ([batch, 4*16+nc, h, w], [batch, 17*3, h*w])
    eval模式下返回: [batch, num_anchors, 4+nc+17*3], bboxes和keypoints已解码成推理时传入的图片尺寸,mask已做sigmoid
    """

    def __init__(self, nc, from_channels: list, kpt_shape=(17, 3), reg_max=16):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, from_channels, reg_max)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(from_channels[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(CBA(x, c4, 3), CBA(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in from_channels)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        batch_size = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(batch_size, self.nk, -1) for i in range(len(self.from_channels))], -1)  # (batch_size, 17*3, h*w)
        for i, (cv2_module, cv3_module) in enumerate(zip(self.cv2, self.cv3)):
            x[i] = torch.cat((cv2_module(x[i]), cv3_module(x[i])), 1)

        if self.training:
            return x, kpt

        # 解码锚框,self.strides是在构建模型时添加的属性，详见TinyTrain\models\yolo\detect\model中的__init__函数
        anchors, strides = make_anchors(x, self.strides)
        x = YOLODetect.inference(self, x)
        pred_kpt = self.kpts_decode(kpt, anchors, strides).permute(0, 2, 1).contiguous()
        return torch.cat([x, pred_kpt], -1)

    def kpts_decode(self, kpts, anchors, strides):
        """
        Decodes keypoints.
        kpts:     [B, 54, 8400]  最后一维是 anchor 维
        anchors:  [8400, 2]
        strides:  [8400, 1]
        """
        anchors = anchors.to(kpts).permute(1, 0)  # [2, 8400]
        strides = strides.to(kpts).permute(1, 0)  # [1, 8400]

        ndim = self.kpt_shape[1]  # 每个 key-point 的维度，通常是 2 或 3
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[0] - 0.5)) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[1] - 0.5)) * strides
        return y


@TTModuleRegistry.register
class YOLOSegment(YOLODetect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc, from_channels: list, nm=32, npr=256):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, from_channels)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(from_channels[0], self.npr, self.nm)  # protos

        c4 = max(from_channels[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(CBA(x, c4, 3), CBA(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in from_channels)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(len(self.from_channels))], 2)  # mask coefficients
        x = YOLODetect.forward(self, x)
        if self.training:
            return x, mc, p
        return torch.cat([x, mc], 1), p
