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

class LrmREC(nn.Module):
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
        # x=self.visual_encoder(x)
        # y=self.lang_encoder(y)

        #!add lrm in bb 
        y=self.lang_encoder(y)
        x=self.visual_encoder(x,y['lang_feat'],y['lang_feat_mask'])

        # vision and language fusion
        # for i in range(len(self.fusion_manner)):
        #     x[i] = self.fusion_manner[i](x[i], y['flat_lang_feat'][-1])
            # x[i] = self.fusion_manner[i](x[i], y['lang_feat'])
            
        # x[-1] = self.fusion_manner[-1](x[-1], y['lang_feat'])
        
        # ! asff
        out_dark3, out_dark4, x0 = x
        out_assf_5 = self.assf_5(x0, out_dark4, out_dark3)
        out_assf_4 = self.assf_4(x0, out_dark4, out_dark3)
        out_assf_3 = self.assf_3(x0, out_dark4, out_dark3)

        x = [out_assf_3, out_assf_4, out_assf_5]

        for i in range(len(self.fusion_manner)):
            x[i] = self.fusion_manner[i](x[i], y['flat_lang_feat'][-1])

        # # multi-scale vision features
        x=self.multi_scale_manner(x)

        # for i in range(len(self.fusion_manner2)):
        #     x[i] = self.fusion_manner2[i](x[i], y['flat_lang_feat'][-1])

        # x=self.multi_scale_manner(x) 

        # for i in range(len(self.fusion_manner2)):
        #     x[i] = self.fusion_manner2[i](x[i], y['flat_lang_feat'][-1])

        # x=self.multi_scale_manner(x) 

        # for i in range(len(self.fusion_manner2)):
        #     x[i] = self.fusion_manner2[i](x[i], y['flat_lang_feat'][-1])

        # x=self.multi_scale_manner(x)
        
        # x=self.multi_scale_manner(x)
        # for i in range(len(self.fusion_manner2)):
        #     x[i] = self.fusion_manner2[i](x[i], y['flat_lang_feat'][-1])

        # x=self.attention_manner(y['flat_lang_feat'][-1],x[-1]) 

        # output
        if self.training:
            loss,loss_det,loss_seg=self.head(x[-1],x[0],det_label,seg_label)
            # loss,loss_det,loss_seg=self.head(out_assf_5,out_assf_5,det_label,seg_label)

            # loss,loss_det,loss_seg=self.head(x,x,det_label,seg_label)
            return loss,loss_det,loss_seg
        else:
            # box, mask=self.head(x,x)
            box, mask=self.head(x[-1],x[0])
            return box,mask


