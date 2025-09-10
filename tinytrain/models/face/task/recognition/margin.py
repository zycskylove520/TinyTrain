import math
import torch

from torch import nn


class CombinedMargin(nn.Module):
    """
    只做 margin 变换的纯“无权重”模块，必须与 PartialFCLoss 配套使用。

    支持两种互斥的 margin 策略：
    1. ArcFace (additive angular margin)
    2. CosFace (additive cosine margin)

    此外提供“类间过滤”功能，可抑制高相似度负类 logits 对后续 softmax 的干扰。

    Args
    -----
    s : float, default 64.0
        缩放因子，最终输出 logits *= s。
    m_arc : float, default 0.0
        ArcFace 角度 margin。>0 时开启 ArcFace，与 m_cos 互斥。
    m_cos : float, default 0.0
        CosFace 余弦 margin。>0 时开启 CosFace，与 m_arc 互斥。
    interclass_filtering_threshold : float, default 0
        若 >0，则对负类 logits 做抑制：当 cosine > threshold 时将其置 0。
    eps : float, default 1e-7
        数值保护小量。

    Raises
    ------
    ValueError
        同时给 m_arc 和 m_cos 赋值（>0）时抛出。

    Note
    ----
    - 标签值为 -1 的样本会被跳过，保持原 logits。
    - 本模块 **不** 包含任何可训练参数，仅做 logits 的 in-place 修正。
    - 输出已乘以 s，可直接接 PartialFCLoss 或 CrossEntropyLoss。
    """

    def __init__(
            self,
            s: float = 64.0,
            m_arc: float = 0.0,
            m_cos: float = 0.0,
            interclass_filtering_threshold=0,
            eps: float = 1e-7,
    ):

        super().__init__()
        active = [m_arc > 0, m_cos > 0].count(True)
        if active > 1:
            raise ValueError("Only one margin type can be active at a time.")

        self.s = s
        self.m_arc = m_arc
        self.m_cos = m_cos
        self.interclass_filtering_threshold = interclass_filtering_threshold
        self.eps = eps

        if m_arc > 0:
            self.easy_margin = False
            self.cos_m = math.cos(m_arc)
            self.sin_m = math.sin(m_arc)
            self.th = math.cos(math.pi - m_arc)
            self.mm = math.sin(m_arc) * m_arc

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        对 cosine similarity logits 施加 margin 修正并缩放。

        Args
        ----
        logits : Tensor[batch, local_classes]
            当前卡/进程可见的局部 cosine similarity，范围 [-1, 1]。
        labels : Tensor[batch], dtype=int64
            全局标签，-1 表示忽略该样本。

        Returns
        ----
        Tensor[batch, local_classes]
            已施加 margin 并乘以 s 后的 logits，可直接用于后续 loss。
        """

        # 只处理标签不是 -1 的行
        index = torch.where(labels != -1)[0]
        if index.numel() == 0:
            return logits * self.s

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index], 0)
                dirty[index] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logits = logits[index, labels[index]].clamp(-1 + 1e-7, 1 - 1e-7)

        if self.m_arc > 0:  # ArcFace
            cosine, sine = target_logits, (1 - target_logits.square()).clamp(min=0).sqrt()
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            logits[index, labels[index]] = phi
        elif self.m_cos > 0:  # CosFace
            logits[index, labels[index]] = target_logits - self.m_cos

        return logits * self.s


class ArcFace(torch.nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    仅对 target logit 施加角度惩罚：
    cos(θ + m) = cosθ·cosm − sinθ·sinm
    不改变负类，输出已乘缩放因子 s，可直接接 CrossEntropyLoss。

    Args
    ----
    s : float, default 64.0
        输出 logits 的全局缩放因子。
    m_arc : float, default 0.5
        角度 margin（弧度）。建议 0.2~0.5。
    """

    def __init__(self, s=64.0, m_arc=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m_arc = m_arc
        self.cos_m = math.cos(m_arc)
        self.sin_m = math.sin(m_arc)
        self.th = math.cos(math.pi - m_arc)
        self.mm = math.sin(math.pi - m_arc) * m_arc
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        前向：对 target 位置执行 cos(θ + m) 变换。

        Args
        ----
        logits : Tensor[batch, local_classes]
            当前卡可见的 cosine similarity，范围 [-1, 1]。
        labels : Tensor[batch],  dtype=int64
            全局标签，-1 表示忽略该样本。

        Returns
        ----
        Tensor[batch, local_classes]
            已施加角度 margin 并乘以 s 的 logits。
        """
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        # 数值稳定：截断
        logits = torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7)

        # arccos
        with torch.no_grad():
            target_logit_arccos = torch.arccos(target_logit)
            logits_arccos = torch.arccos(logits)
            final_target_logit_arccos = target_logit_arccos + self.m_arc

            if self.easy_margin:
                final_target_logit_arccos = torch.where(
                    target_logit_arccos > 0,
                    final_target_logit_arccos,
                    target_logit_arccos
                )

            logits_arccos = logits_arccos.clone()
            logits_arccos[index, labels[index].view(-1)] = final_target_logit_arccos
            logits = torch.cos(logits_arccos)

        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    """
    CosFace: Additive Cosine Margin Loss
    仅对 target logit 减去固定 margin m，不改变负类。
    输出已乘缩放因子 s，可直接接 CrossEntropyLoss。

    Args
    ----
    s : float, default 64.0
        输出 logits 的全局缩放因子。
    m_cos : float, default 0.40
        余弦 margin。建议 0.2~0.4。
    """

    def __init__(self, s=64.0, m_cos=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m_cos = m_cos

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        前向：对 target 位置执行 cosθ - m 变换。

        Args
        ----
        logits : Tensor[batch, local_classes]
            当前卡可见的 cosine similarity，范围 [-1, 1]。
        labels : Tensor[batch],  dtype=int64
            全局标签，-1 表示忽略该样本。

        Returns
        ----
        Tensor[batch, local_classes]
            已施加余弦 margin 并乘以 s 的 logits。
        """
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m_cos
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
