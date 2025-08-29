import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Conv2Linear
from .conv import Conv, DWConv
from tinytrain.cfg.TT_register import TTModuleRegistry
from tinytrain.utils.tal import make_anchors, dist2bbox


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
        self.conv = Conv(in_channels, hidden_channels, kernel_size, stride, padding, groups)
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
class ArcFace(nn.Module):
    def __init__(self, in_channels, feature_dim, nc, kernel_size=1, stride=1, padding=None, groups=1, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.features = Conv2Linear(in_channels=in_channels, out_channels=feature_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight = nn.Parameter(torch.FloatTensor(nc, feature_dim))

        nn.init.xavier_uniform_(self.weight)
        self.target = None

        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.export = False

    def forward(self, x):
        x = self.features(x)

        if self.export:
            return x

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # resnet网络最后一层输出的是全连接层，把全连接层权重W归一化。
        # 这里全连接层输出的是cosine的原因为，X和W都使用了l2范数变成了单位向量，因此计算出来的每一个值都是余弦值
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))  # 这里使用clamp可能是担心精度溢出
        phi = cosine * self.cos_m - sine * self.sin_m  # 计算：cos(theta + m)
        if self.easy_margin:
            # easy_margin可以理解为θ+m>pi，此时cos(θ+m)超过了0-pi的单调区间，那就不管了，直接用cos(θ)代替
            # cosine>0表示theta<pi/2，因此m+theta不会超过pi，在该区间使用cos(theta+m)可以将同类之间收的更紧
            # 如果cosine<0表示theta>pi/2，因此m+theta在最坏的情况下，比如theta=pi时，theta+m会超过pi，此时跳出了cos在0-pi的单调区间，
            # 为了保持单调性，这种情况直接使用cos(theta)，因为cos(theta)在单调区间内。所以就是不将同类收紧了，无所谓了
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # 非easy_margin可以理解为就算θ+m>pi，此时cos(θ+m)超过了0-pi的单调区间，也要坚持使用m收紧同类
            # cos在0-pi的单调区间内，cos(θ) > cos(pi-m)表示θ<pi-m即：θ+m<pi时，使用cos(θ+m)
            # 否则就使用类似cosface的损失函数来代替
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.target.device)
        one_hot.scatter_(1, self.target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def inference(self, x):
        x = self.features(x)
        return x


@TTModuleRegistry.register
class CosFace(nn.Module):
    def __init__(self, in_channels, feature_dim, nc, kernel_size=1, stride=1, padding=None, groups=1, s=30.0, m=0.40):
        super().__init__()
        self.features = Conv2Linear(in_channels=in_channels, out_channels=feature_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight = nn.Parameter(torch.FloatTensor(nc, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.target = None

        self.s = s
        self.m = m

        self.export = False

    def forward(self, x):
        x = self.features(x)

        if self.export:
            return x

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=self.target.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, self.target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def inference(self, x):
        x = self.features(x)
        return x


@TTModuleRegistry.register
class SphereFace(nn.Module):
    def __init__(self, in_channels, feature_dim, nc, m=4, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        self.features = Conv2Linear(in_channels=in_channels, out_channels=feature_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight = nn.Parameter(torch.FloatTensor(nc, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.target = None

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        self.export = False

    def forward(self, x):
        x = self.features(x)

        if self.export:
            return x

        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(x, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, self.target.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def inference(self, x):
        x = self.features(x)
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

        self.bias_init()

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
        anchors, strides = make_anchors(x, self.stride)

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

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        # 如果不做bias初始化，输出的cls loss会非常大，这会导致反向传播让权重参数迅速归0的问题
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(self.cv2, self.cv3, [8, 16, 32]):  # from
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
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in from_channels)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        batch_size = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(batch_size, self.nk, -1) for i in range(len(self.from_channels))], -1)  # (batch_size, 17*3, h*w)
        if self.training:
            x = super().forward(x)
            return x, kpt

        # 解码锚框,self.stride是在构建模型时添加的属性，详见TinyTrain\models\yolo\detect\model中的__init__函数
        anchors, strides = make_anchors(x, self.stride)
        x = super().forward(x)
        pred_kpt = self.kpts_decode(kpt, anchors, strides).permute(0, 2, 1).contiguous()
        return torch.cat([x, pred_kpt], 1)

    def kpts_decode(self, kpts, anchors, strides):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (anchors[:, 0] - 0.5)) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (anchors[:, 1] - 0.5)) * strides
        return y
