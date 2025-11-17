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
import torch

import torch.nn.functional as F

from tinytrain.engine import TTBaseDistiller, TTBaseModel
from tinytrain.modules.assigner.taa_assigner import dist2bbox
from tinytrain.utils.box_utils import make_anchors


class YOLODetectionDistiller(TTBaseDistiller):
    def __init__(self, trainer, teacher_model: TTBaseModel):
        super().__init__(trainer, teacher_model)

        self.nc = self.config_manager.dataset["nc"]
        self.distill_weight = self.config_manager.loss["distill_loss_gain"]
        self.temperature = self.config_manager.loss["distill_temperature"]  # 增加温度参数，软化分布

        self.use_dfl = self.student_model.reg_max > 1
        self.proj = torch.arange(self.student_model.reg_max, dtype=torch.float, device=self.device)

    def execute_forward(self, batch_samples):
        student_outputs = self.student_model.inference(batch_samples.data)
        teacher_outputs = self.teacher_model.inference(batch_samples.data)

        anchor_points, stride_tensor = make_anchors(student_outputs[0], self.student_model.strides, 0.5)

        # 原始loss
        total_student_loss_with_weight, student_loss_items = self.student_model.loss(student_outputs, batch_samples)

        # 将多个头的输出拼接起来
        s_cat = torch.cat([s_out.flatten(2) for s_out in student_outputs[0]], dim=2)
        t_cat = torch.cat([t_out.flatten(2) for t_out in teacher_outputs[0]], dim=2)

        # 分离分类和回归输出
        s_dist, s_cls = s_cat.split((self.student_model.reg_max * 4, self.nc), 1)
        t_dist, t_cls = t_cat.split((self.student_model.reg_max * 4, self.nc), 1)

        s_cls = s_cls.permute(0, 2, 1).contiguous()  # [batch, num_anchors, cls_num]
        t_cls = t_cls.permute(0, 2, 1).contiguous()  # [batch, num_anchors, cls_num]
        s_dist = s_dist.permute(0, 2, 1).contiguous()  # [batch, num_anchors, reg_max*4]
        t_dist = t_dist.permute(0, 2, 1).contiguous()  # [batch, num_anchors, reg_max*4]

        s_box = self.bbox_decode(anchor_points, s_dist)  # [batch, num_anchors, 4]
        t_box = self.bbox_decode(anchor_points, t_dist)  # [batch, num_anchors, 4]

        # 计算各类蒸馏loss
        cls_loss = self._cls_distill_loss(s_cls, t_cls)
        reg_loss = self._regression_distill_loss(s_box, t_box)
        feat_loss = self._feature_distill_loss(s_cat, t_cat)

        total_distill_loss = cls_loss + reg_loss + feat_loss

        student_loss_items.update({
            'cls_distill': cls_loss,
            'reg_distill': reg_loss,
            'feat_distill': feat_loss
        })

        return total_student_loss_with_weight + total_distill_loss * self.distill_weight, student_loss_items

    def _cls_distill_loss(self, student_logits, teacher_logits):
        """
        KL散度损失
        """
        # 分类输出使用sigmoid+KL散度
        if self.nc > 1:
            # 应用softmax计算分布
            s_probs = torch.softmax(student_logits / self.temperature, dim=-1)
            t_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)

            # 添加微小值防止数值不稳定
            s_probs = torch.clamp(s_probs, min=1e-8, max=1 - 1e-8)
            t_probs = torch.clamp(t_probs, min=1e-8, max=1 - 1e-8)

            loss = F.kl_div(
                torch.log(s_probs),
                t_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
        else:
            # 应用sigmoid
            s_probs = torch.sigmoid(student_logits / self.temperature)  # 类别为1的概率
            t_probs = torch.sigmoid(teacher_logits / self.temperature)  # 类别为1的概率

            # 构建完整的二分类概率分布 [P(类别=1), P(类别≠1)]
            s_dist = torch.cat([s_probs, 1 - s_probs], dim=-1)  # [batch, num_anchors, 2]
            t_dist = torch.cat([t_probs, 1 - t_probs], dim=-1)  # [batch, num_anchors, 2]

            # 添加微小值防止数值不稳定
            s_dist = torch.clamp(s_dist, min=1e-8, max=1 - 1e-8)
            t_dist = torch.clamp(t_dist, min=1e-8, max=1 - 1e-8)

            loss = F.kl_div(
                torch.log(s_dist),
                t_dist,
                reduction='batchmean'
            ) * (self.temperature ** 2)

        return loss

    def _regression_distill_loss(self, student_box, teacher_box):
        """
        回归蒸馏loss
        """
        return F.mse_loss(student_box, teacher_box)

    def _feature_distill_loss(self, student_feat, teacher_feat):
        """
        特征蒸馏loss - 添加归一化
        """
        # 对特征进行归一化，避免尺度差异
        s_feat_norm = student_feat
        t_feat_norm =teacher_feat

        # 对特征进行归一化，避免尺度差异
        # s_feat_norm = F.normalize(student_feat, p=2, dim=1)
        # t_feat_norm = F.normalize(teacher_feat, p=2, dim=1)

        return F.mse_loss(s_feat_norm, t_feat_norm)

    def bbox_decode(self, anchor_points, pred_dist):
        """
        将锚点 + DFL 分布解码为框坐标。

        Args:
            anchor_points: [H*W, 2] 锚点中心
            pred_dist: [B, H*W, 4*reg_max]

        Returns:
            [B, H*W, 4] (lx,ly,rx,ry)
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 将最后一个维度归一化，计算ltrb四个坐标的预测值
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
