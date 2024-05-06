# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.eps=1e-7


    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'diou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2

            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2

            # 求diou
            diou = iou - d / c
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'ciou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最大外界矩形对角线长度c^2
            w_c = (c_br - c_tl)[:, 0]
            h_c = (c_br - c_tl)[:, 1]
            c = w_c ** 2 + h_c ** 2

            # 中心点距离平方d^2
            w_d = (pred[:, :2] - target[:, :2])[:, 0]
            h_d = (pred[:, :2] - target[:, :2])[:, 1]
            d = w_d ** 2 + h_d ** 2

            # 求diou
            diou = iou - d / c

            w_gt = target[:, 2]
            h_gt = target[:, 3]
            w = pred[:, 2]
            h = pred[:, 3]

            with torch.no_grad():
                arctan = torch.atan(w_gt / h_gt) - torch.atan(w / h)
                v = (4 / (math.pi ** 2)) * torch.pow(arctan, 2)
                s = 1 - iou
                alpha = v / (s + v)

            ciou = diou - alpha * v
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf

            b1_x1, b1_x2 = pred[:, 0] - pred[:, 2] / 2, pred[:, 0] + pred[:, 2] / 2
            b1_y1, b1_y2 = pred[:, 1] - pred[:, 3] / 2, pred[:, 1] + pred[:, 3] / 2
            b2_x1, b2_x2 = target[:, 0] - target[:, 2] / 2, target[:, 0] + target[:, 2] / 2
            b2_y1, b2_y2 = target[:, 1] - target[:, 3] / 2, target[:, 1] + target[:, 3] / 2

            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

            # Intersection area
            inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

            # Union Area
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
            union = w1 * h1 + w2 * h2 - inter + self.eps

            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
            loss = 1.0 - iou


        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss