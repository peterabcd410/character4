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

import numpy as np

import torch
import torch.nn as nn 
from simrec.layers.common import RepBlock, RepVGGBlock, BottleRep, BepC3, SimConv, Transpose, BiFusion
from simrec.layers.blocks import darknet_conv
from  simrec.layers.asff import ASFF

torch.backends.cudnn.enabled=False

class LrmREC_Trans(nn.Module):
    def __init__(
        self, 
        visual_backbone: nn.Module, 
        language_encoder: nn.Module,
        multi_scale_manner: nn.Module,
        fusion_manner: nn.Module,
        fusion_manner2: nn.Module,
        attention_manner: nn.Module,
        head: nn.Module,
    ):
        super(LrmREC, self).__init__()
        self.visual_encoder=visual_backbone
        self.lang_encoder=language_encoder
        self.multi_scale_manner = multi_scale_manner
        self.fusion_manner = fusion_manner
        self.fusion_manner2 = fusion_manner2
        self.attention_manner = attention_manner
        self.head=head

        #asff
        self.assf_5 = ASFF(level = 0)
        self.assf_4 = ASFF(level = 1)
        self.assf_3 = ASFF(level = 2)

    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    

    def forward(self, x, y, det_label=None,seg_label=None):

        # vision and language encoding
        x=self.visual_encoder(x)
        y=self.lang_encoder(y)

        x_lgv=self.lgv(x)
        x_vgl=self.vgl(x)

        v_feat = x_lgv.concat(x_vgl)

        

        if self.training:
            

            return loss,loss_det,loss_seg
        else:
            # box, mask=self.head(x,x)
            box, mask=self.head(x[-1],x[0])
            return box,mask


