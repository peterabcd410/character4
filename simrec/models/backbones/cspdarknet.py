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

import torch
import torch.nn as nn

from simrec.layers.blocks import ConvBnAct, C3Block
from simrec.layers.sppf import SPPF
from simrec.layers.fusion_layer import SimpleFusion2, TSRMFusion,MultiScaleFusion, GaranAttention
from simrec.layers.sa_layer import SA

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

# class CspDarkNet(nn.Module):
#     def __init__(
#         self, 
#         pretrained_weight_path=None,
#         pretrained=False, 
#         multi_scale_outputs=False,
#         freeze_backbone=True,
#     ):
#         super().__init__()
#         self.model = nn.Sequential(
#             ConvBnAct(c1=3, c2=64, k=6,s=2,p=2),
#             # i = 0 ch = [64]
#             ConvBnAct(c1=64, c2=128, k=3, s=2),
#             # i = 1 ch = [64,128]
#             C3Block(c1=128, c2=128, n=3),

#             # i = 2 ch =[64,128,128]
#             ConvBnAct(c1=128, c2=256, k=3, s=2),
#             # i = 3 ch =[64,128,128,256]
#             C3Block(c1=256, c2=256, n=6),
#             # i = 4 ch =[64,128,128,256,256]
#             TSRMFusion(v_planes=256, out_planes=512, q_planes=512),#5
            
#             ConvBnAct(c1=256, c2=512, k=3, s=2),
#             # i = 5 ch =[64,128,128,256,256,512]
#             C3Block(c1=512, c2=512, n=9),
#             # i = 6 ch =[64,128,128,256,256,512,512]
#             TSRMFusion(v_planes=512, out_planes=512, q_planes=512),#5

#             ConvBnAct(c1=512, c2=1024, k=3, s=2),
#             # i = 7 ch =[64,128,128,256,256,512,512,1024]
#             C3Block(c1=1024, c2=1024, n=3),
#             # i = 8 ch =[64,128,128,256,256,512,512,1024,1024]
#             TSRMFusion(v_planes=1024, out_planes=512, q_planes=512),#5

#             SPPF(c1=1024, c2=1024, k=5),
#             # i = 9 ch =[64,128,128,256,256,512,512,1024,1024,1024]
#         )
        
#         self.multi_scale_outputs=multi_scale_outputs
#         self.sa = SA(512, 8, 2048, 0.1)

#         if pretrained:
#             self.weight_dict = torch.load(pretrained_weight_path)
#             self.load_state_dict(self.weight_dict, strict=False)

#         if freeze_backbone:
#             self.frozen(self.model[:-1])

#     def frozen(self,module):
#             if getattr(module,'module',False):
#                 print("getattr")
#                 for i,child in enumerate(module.module()):
#                     if i in [5,8,11]:
#                         for param in child.parameters():
#                             param.requires_grad = True
#                     else:
#                         for param in child.parameters():
#                             param.requires_grad = False

#             else:
#                 for i,child in enumerate(module):
#                     if i in [5,8,11]:
#                         for param in child.parameters():
#                             param.requires_grad = True
#                     else:
#                         for param in child.parameters():
#                             param.requires_grad = False

#     def forward(self, x, y,y_mask):
#         outputs=[]
#         for i,module in enumerate(self.model):
            
#             # if i in [5, 8,11]:
#             if i in [5,8,11]:
#                 # y=self.sa(y,y_mask)
#                 x=module(x,y)
#                 if i in [5,8]:
#                     outputs.append(x)
#             else:
#                 x=module(x)

#                 # if i == 2:
#                 #     outputs.append(x)
#         outputs.append(x)
#         if self.multi_scale_outputs:
#             return outputs
#         else:
#             return x

# class CspDarkNet(nn.Module):
#     def __init__(
#         self, 
#         pretrained_weight_path=None,
#         pretrained=False, 
#         multi_scale_outputs=False,
#         freeze_backbone=True,
#     ):
#         super().__init__()
#         self.model = nn.Sequential(
#             ConvBnAct(c1=3, c2=64, k=6,s=2,p=2),
#             # i = 0 ch = [64]
#             ConvBnAct(c1=64, c2=128, k=3, s=2),
#             # i = 1 ch = [64,128]
#             C3Block(c1=128, c2=128, n=3),
#             # i = 2 ch =[64,128,128]
#             ConvBnAct(c1=128, c2=256, k=3, s=2),
#             # i = 3 ch =[64,128,128,256]
#             C3Block(c1=256, c2=256, n=6),
#             # i = 4 ch =[64,128,128,256,256]
#             GaranAttention(d_q=512,d_v=256),#5
            
#             ConvBnAct(c1=256, c2=512, k=3, s=2),
#             # i = 5 ch =[64,128,128,256,256,512]
#             C3Block(c1=512, c2=512, n=9),
#             # i = 6 ch =[64,128,128,256,256,512,512]
#             GaranAttention(d_q=512,d_v=512),#5

#             ConvBnAct(c1=512, c2=1024, k=3, s=2),
#             # i = 7 ch =[64,128,128,256,256,512,512,1024]
#             C3Block(c1=1024, c2=1024, n=3),
#             # i = 8 ch =[64,128,128,256,256,512,512,1024,1024]
#             GaranAttention(d_q=512,d_v=1024),#5

#             SPPF(c1=1024, c2=1024, k=5),
#             # i = 9 ch =[64,128,128,256,256,512,512,1024,1024,1024]
#         )
        
#         self.multi_scale_outputs=multi_scale_outputs
        
#         if pretrained:
#             self.weight_dict = torch.load(pretrained_weight_path)
#             self.load_state_dict(self.weight_dict, strict=False)

#         if freeze_backbone:
#             self.frozen(self.model[:-1])

#     def frozen(self,module):
#             if getattr(module,'module',False):
#                 print("getattr")
#                 for i,child in enumerate(module.module()):
#                     if i in [5,8,11]:
#                         for param in child.parameters():
#                             param.requires_grad = True
#                     else:
#                         for param in child.parameters():
#                             param.requires_grad = False

#             else:
#                 for param in module.parameters():
#                     param.requires_grad = False

#     def forward(self, x, y):
#         outputs=[]
#         j = 0
#         for i,module in enumerate(self.model):
#             # print(i)
#             # print(x.shape)
            
#             if i in [5, 8, 11]:
                
#                 # print(j,y[j].shape)
#                 x=module(y[j],x)[0]
#                 j=j+1
                
#             else:
#                 x=module(x)
#                 if i in [2,4, 7]:
#                     outputs.append(x)

#         outputs.append(x)
#         if self.multi_scale_outputs:
#             return outputs
#         else:
#             return x

class CspDarkNet(nn.Module):
    def __init__(
        self, 
        pretrained_weight_path=None,
        pretrained=False, 
        multi_scale_outputs=False,
        freeze_backbone=True,
    ):
        super().__init__()
        self.model = nn.Sequential(
            ConvBnAct(c1=3, c2=64, k=6,s=2,p=2),
            # i = 0 ch = [64]
            ConvBnAct(c1=64, c2=128, k=3, s=2),
            # i = 1 ch = [64,128]
            C3Block(c1=128, c2=128, n=3),
            # i = 2 ch =[64,128,128]
            ConvBnAct(c1=128, c2=256, k=3, s=2),
            # i = 3 ch =[64,128,128,256]
            C3Block(c1=256, c2=256, n=6),
            # i = 4 ch =[64,128,128,256,256]
            
            ConvBnAct(c1=256, c2=512, k=3, s=2),
            # i = 5 ch =[64,128,128,256,256,512]
            C3Block(c1=512, c2=512, n=9),
            # i = 6 ch =[64,128,128,256,256,512,512]

            ConvBnAct(c1=512, c2=1024, k=3, s=2),
            # i = 7 ch =[64,128,128,256,256,512,512,1024]
            C3Block(c1=1024, c2=1024, n=3),
            # i = 8 ch =[64,128,128,256,256,512,512,1024,1024]

            SPPF(c1=1024, c2=1024, k=5),
            # i = 9 ch =[64,128,128,256,256,512,512,1024,1024,1024]
        )
        
        self.multi_scale_outputs=multi_scale_outputs
        
        if pretrained:
            self.weight_dict = torch.load(pretrained_weight_path)
            self.load_state_dict(self.weight_dict, strict=False)

        if freeze_backbone:
            self.frozen(self.model[:-1])

    def frozen(self,module):
            if getattr(module,'module',False):
                print("getattr")
                for i,child in enumerate(module.module()):
                    for param in child.parameters():
                        param.requires_grad = False

            else:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        outputs=[]
        for i,module in enumerate(self.model):
            
            x=module(x)
            if i in [4,6]:
                outputs.append(x)

        outputs.append(x)
        if self.multi_scale_outputs:
            return outputs
        else:
            return x