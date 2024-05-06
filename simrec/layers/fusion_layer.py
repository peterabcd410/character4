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
import torch.nn.functional as F
from .common import RepBlock, RepVGGBlock, BottleRep, BepC3, SimConv, Transpose, BiFusion
from .blocks import darknet_conv


class CollectDiffuseAttention(nn.Module):
    ''' CollectDiffuseAttention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, kc,kd, v, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''

        attn_col = torch.bmm(q, kc.transpose(1, 2)) #n*b,1,h*w
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v) #n*b,1,d_o

        attn_dif = torch.bmm(kd,q.transpose(1, 2)) #n*b,h*w,1
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = torch.sigmoid(attn_dif_logit)
        attn_dif= self.dropout_d(attn_dif)
        output=torch.bmm(attn_dif,attn)
        return output, attn_col_logit.squeeze(1)

class GaranAttention(nn.Module):
    """
    Garan Attention Module
    """

    def __init__(self,d_q, d_v,n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k=d_v
        self.d_o=d_v
        d_o=d_v

        self.w_qs = nn.Linear(d_q, d_o)
        self.w_kc = nn.Conv2d(d_v, d_o,1)
        self.w_kd = nn.Conv2d(d_v, d_o,1)
        self.w_vs = nn.Conv2d(d_v, d_o,1)
        self.w_m=nn.Conv2d(d_o,1,3,1,padding=1)
        self.w_o=nn.Conv2d(d_o,d_o,1)

        self.attention = CollectDiffuseAttention(temperature=np.power(d_o//n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti= nn.LeakyReLU(0.1,inplace=True)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, v, mask=None):

        d_k, d_v, n_head,d_o = self.d_k, self.d_v, self.n_head,self.d_o
        q= q[0]
        sz_b, c_q = q.size()
        sz_b,c_v, h_v,w_v = v.size()
        residual = v

        q = self.w_qs(q)
        kc=self.w_kc(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        kd=self.w_kd(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        v=self.w_vs(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        q=q.view(sz_b,n_head,1,d_o//n_head)
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, 1, d_o//n_head) # (n*b) x lq x dk
        kc = kc.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        kd=kd.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        v = v.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lv x dv

        output, m_attn = self.attention(q, kc,kd, v)
        #n * b, h * w, d_o
        output = output.view(sz_b,n_head, h_v,w_v, d_o//n_head)
        output = output.permute(0,1,4,3,2).contiguous().view(sz_b,-1, h_v,w_v) # b x lq x (n*dv)
        m_attn=m_attn.view(sz_b,n_head, h_v*w_v)
        # m_attn=m_attn.mean(1)

        #residual connect
        output=self.w_o(output)
        attn=output
        m_attn=self.w_m(attn).view(sz_b, h_v*w_v)
        output=self.layer_norm(output)
        output= output+residual
        output=self.layer_acti(output)
        # print(output.shape)
        # output = self.dropout(self.fc(output))

        return output, m_attn,attn


class CollectDiffuseAttentionV2(nn.Module):
    """
    Collect Diffuse Attention Module (Version-2)
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout_c = nn.Dropout(attn_dropout)
        self.dropout_d = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q, kc,kd, v,flap_map, mask=None):
        '''
        q: n*b,1,d_o
        kc: n*b,h*w,d_o
        kd: n*b,h*w,d_o
        v: n*b,h*w,d_o
        '''


        attn_col = torch.bmm(q, kc.transpose(1, 2)) #n*b,l,h*w
        attn_col=(attn_col*flap_map).sum(1,keepdim=True)
        attn_col_logit = attn_col / self.temperature
        attn_col = self.softmax(attn_col_logit)
        attn_col = self.dropout_c(attn_col)
        attn = torch.bmm(attn_col, v) #n*b,1,d_o

        attn_dif = torch.bmm(kd,q.transpose(1, 2)) #n*b,h*w,1
        attn_dif = (attn_dif * flap_map.transpose(1,2)).sum(-1, keepdim=True)
        attn_dif_logit = attn_dif / self.temperature
        attn_dif = torch.sigmoid(attn_dif_logit)
        attn_dif= self.dropout_d(attn_dif)
        output=torch.bmm(attn_dif,attn)
        return output, attn_col_logit.squeeze(1)


class GaranAttentionV2(nn.Module):
    """
    Garan Attention V2
    """

    def __init__(self,d_q, d_v,n_head=2, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_v = d_v
        self.d_k=d_v
        self.d_o=d_v
        d_o=d_v

        self.w_qs = nn.Linear(d_q, d_o, 1)
        self.w_qs_att = nn.Linear(d_q, 1, 1)
        self.w_kc = nn.Conv2d(d_v, d_o,1)
        self.w_kd = nn.Conv2d(d_v, d_o,1)
        self.w_vs = nn.Conv2d(d_v, d_o,1)
        self.w_m=nn.Conv2d(d_o,1,3,1,padding=1)

        self.attention = CollectDiffuseAttentionV2(temperature=np.power(d_o//n_head, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_o)
        self.layer_acti= nn.LeakyReLU(0.1,inplace=True)
        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, v, mask=None):

        d_k, d_v, n_head,d_o = self.d_k, self.d_v, self.n_head,self.d_o

        sz_b, l,c_q = q.size()
        sz_b,c_v, h_v,w_v = v.size()
        residual = v

        q = self.w_qs(q)
        flat_map=self.w_qs_att(q)
        if mask is not None:
            mask=mask.view(sz_b,l,1)
            flat_map = flat_map.masked_fill(mask, -1e9).repeat(self.n_head,1,1)
            # flat_map=flat_map.view(sz_b,1,l)
        flat_map = F.softmax(flat_map, dim=-1)
        kc=self.w_kc(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        kd=self.w_kd(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        v=self.w_vs(v).view(sz_b,n_head,d_o//n_head,h_v*w_v)
        q=q.view(sz_b,-1,n_head,d_o//n_head).transpose(1,2).contiguous()
        # v=v.view(sz_b,h_v*w_v,n_head,c_v//n_head)

        q = q.view(-1, l, d_o//n_head) # (n*b) x lq x dk
        kc = kc.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        kd=kd.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lk x dk
        v = v.permute(0,1,3,2).contiguous().view(-1, h_v*w_v, d_o//n_head) # (n*b) x lv x dv

        output, m_attn = self.attention(q, kc,kd, v,flat_map)
        #n * b, h * w, d_o
        output = output.view(sz_b,n_head, h_v,w_v, d_o//n_head)
        output = output.permute(0,1,4,3,2).contiguous().view(sz_b,-1, h_v,w_v) # b x lq x (n*dv)
        m_attn=m_attn.view(sz_b,n_head, h_v*w_v)
        # m_attn=m_attn.mean(1)
        attn=output
        m_attn=self.w_m(attn).view(sz_b, h_v*w_v)


        #residual connect
        output=self.layer_norm(output)
        output= output+residual
        output=self.layer_acti(output)

        # output = self.dropout(self.fc(output))

        return output, m_attn,attn


class SimpleFusion(nn.Module):
    def __init__(self,v_planes=1024,q_planes=1024,out_planes=1024,rate=4):
        super().__init__()
        self.v_proj=nn.Sequential(
            nn.Conv2d(v_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.q_proj=nn.Sequential(
            nn.Conv2d(q_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.norm=nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(v_planes, int(v_planes / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(v_planes / rate), v_planes)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(v_planes, int(v_planes / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(v_planes / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(v_planes / rate), v_planes, kernel_size=7, padding=3),
            nn.BatchNorm2d(v_planes)
        )


    def forward(self, x, y):

        # b, c, h, w = x.shape
        # x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        # x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        # x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        # x = x * x_channel_att

        # x_spatial_att = self.spatial_attention(x).sigmoid()
        # out = x * x_spatial_att
        # x = out

        x=self.v_proj(x)
        # print(y)
        y=self.q_proj(y[0].unsqueeze(2).unsqueeze(2))
        return self.norm(x*y)

class SimpleFusion2(nn.Module):
    def __init__(self,v_planes=1024,q_planes=1024,out_planes=1024,rate=4):
        super().__init__()
        self.v_proj=nn.Sequential(
            nn.Conv2d(v_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.q_proj=nn.Sequential(
            nn.Conv2d(q_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
        self.norm=nn.Sequential(
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )


    def forward(self, x, y):

        x=self.v_proj(x)
        y=self.q_proj(y.unsqueeze(2).unsqueeze(2))
        return self.norm(x*y)


class MultiScaleFusion(nn.Module): #old
    def __init__(self,v_planes=[256,512,1024],hiden_planes=512,scaled=True):
        super().__init__()
        self.up_modules=nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2]+hiden_planes//2, hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            ),
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-1], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3)
            )]
        )

        self.down_modules=nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes//2+v_planes[0], hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes//2, 3),
            ),
                nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(hiden_planes+v_planes[1], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            )]
        )


        self.top_proj=darknet_conv(v_planes[-1]+hiden_planes//2,hiden_planes,1)
        self.mid_proj=darknet_conv(v_planes[1]+hiden_planes,hiden_planes,1)
        self.bot_proj=darknet_conv(v_planes[0]+hiden_planes//2,hiden_planes,1)

    def forward(self, x):
        l,m,s=x
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)
        # out=self.out_proj(l)

        m = torch.cat([self.down_modules[0](l), m], 1)

        s = torch.cat([self.down_modules[1](m), s], 1)

        #top prpj and bot proj
        top_feat=self.top_proj(s)
        mid_feat=self.mid_proj(m)
        bot_feat=self.bot_proj(l)
        return [bot_feat,mid_feat,top_feat]

# class MultiScaleFusion(nn.Module):#base
#     def __init__(self,v_planes=[256,512,1024], hiden_planes=512, scaled=True):
#         super().__init__()
#         self.up_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-2]+hiden_planes//2, hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3),
#             ),
#             nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-1], hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3)
#             )]
#         )

#         self.down_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 darknet_conv(hiden_planes//2+v_planes[0], hiden_planes // 2, ksize=1),
#                 darknet_conv(hiden_planes // 2, hiden_planes//2, 3),
#             ),
#                 nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 darknet_conv(hiden_planes+v_planes[1], hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3),
#             )]
#         )


#         self.top_proj=darknet_conv(v_planes[-1]+hiden_planes//2,hiden_planes,1)
#         self.mid_proj=darknet_conv(v_planes[1]+hiden_planes,hiden_planes,1)
#         self.bot_proj=darknet_conv(v_planes[0]+hiden_planes//2,hiden_planes,1)

#     def forward(self, x):
#         # l torch.Size([16, 512, 68, 68])
#         # m torch.Size([16, 1024, 34, 34])
#         # s torch.Size([16, 1280, 17, 17])

#         l,m,s=x
#         m = torch.cat([self.up_modules[1](s), m], 1)
#         l = torch.cat([self.up_modules[0](m), l], 1)
#         # out=self.out_proj(l)

#         m = torch.cat([self.down_modules[0](l), m], 1)

#         s = torch.cat([self.down_modules[1](m), s], 1)
        
#         # print("l",l.shape)
#         # print("m",m.shape)
#         # print("s",s.shape)

#         #top prpj and bot proj
#         top_feat=self.top_proj(s)
#         mid_feat=self.mid_proj(m)
#         bot_feat=self.bot_proj(l)
#         return [bot_feat,mid_feat,top_feat]


# class MultiScaleFusion(nn.Module):#change lrm
#     def __init__(self,v_planes=[256,512,1024], hiden_planes=512, scaled=True):
#         super().__init__()
#         self.up_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 # TSRMFusion(v_planes=v_planes[-2], q_planes=512,out_planes=512),
#                 darknet_conv(v_planes[-2]+hiden_planes//2, hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3)         
#             ),
#             nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 # TSRMFusion(v_planes=v_planes[-1], q_planes=512,out_planes=512),
#                 darknet_conv(v_planes[-1], hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3),
#             )]
#         )

#         self.down_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 # TSRMFusion(v_planes=hiden_planes//2+v_planes[0], q_planes=512,out_planes=512),
#                 darknet_conv(hiden_planes//2+v_planes[0], hiden_planes // 2, ksize=1),
#                 darknet_conv(hiden_planes // 2, hiden_planes//2, 3),
#             ),
#             nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 # TSRMFusion(v_planes=hiden_planes+v_planes[1], q_planes=512,out_planes=512),
#                 darknet_conv(hiden_planes+v_planes[1], hiden_planes//2, ksize=1),
#                 darknet_conv(hiden_planes//2, hiden_planes//2, 3),
#             )]
#         )

#         self.lrm=nn.Sequential(
#                 TSRMFusion(v_planes=v_planes[-2]+hiden_planes//2, q_planes=512,out_planes=512),
#                 TSRMFusion(v_planes=hiden_planes//2+v_planes[0], q_planes=512,out_planes=512),
#                 TSRMFusion(v_planes=hiden_planes+v_planes[1], q_planes=512,out_planes=512),      
#                 TSRMFusion(v_planes=v_planes[-1]+hiden_planes//2, q_planes=512,out_planes=512),
#             )

#         self.top_proj=darknet_conv(v_planes[-1]+hiden_planes//2,hiden_planes,1)
#         self.mid_proj=darknet_conv(v_planes[1]+hiden_planes,hiden_planes,1)
#         self.bot_proj=darknet_conv(v_planes[0]+hiden_planes//2,hiden_planes,1)

#     def forward(self, x,y):
#         l,m,s=x
#         m = torch.cat([self.up_modules[1](s), m], 1)

#         m=self.lrm[0](m,y)

#         l = torch.cat([self.up_modules[0](m), l], 1)
#         # out=self.out_proj(l)
#         l=self.lrm[1](l,y)

#         m = torch.cat([self.down_modules[0](l), m], 1)

#         m=self.lrm[2](m,y)

#         s = torch.cat([self.down_modules[1](m), s], 1)

#         s=self.lrm[3](s,y)

#         #top prpj and bot proj
#         top_feat=self.top_proj(s)
#         mid_feat=self.mid_proj(m)
#         bot_feat=self.bot_proj(l)
#         return [bot_feat,mid_feat,top_feat]



class MultiScaleFusion_(nn.Module):
    def __init__(self, v_planes=[128,256, 512, 1024], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules = nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2] + hiden_planes // 2, hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                    darknet_conv(v_planes[-1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3)
                )
                ,
                nn.Sequential(
                    darknet_conv(768, 128, ksize=1),
                    nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                    darknet_conv(128,128, 3)
                )
            ]
        )

        self.down_modules = nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(1024, hiden_planes // 2, ksize=1),
                darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
            ),
                nn.Sequential(
                    nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                    darknet_conv(hiden_planes + v_planes[1], hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                ),
                nn.Sequential(
                    nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                    darknet_conv(384, hiden_planes // 2, ksize=1),
                    darknet_conv(hiden_planes // 2, hiden_planes // 2, 3),
                )]
        )

        self.top_proj = darknet_conv(v_planes[-1] + hiden_planes // 2, hiden_planes, 1)
        self.mid_proj = darknet_conv(v_planes[1] + hiden_planes, hiden_planes, 1)
        self.bot_proj = darknet_conv(v_planes[0] + hiden_planes // 2, hiden_planes, 1)
        self.out_proj = nn.Sequential(nn.AvgPool2d(4, 4),
                                      darknet_conv(hiden_planes // 2 + v_planes[0], hiden_planes, ksize=1),
                                      )

    def forward(self, x):

        ll,l, m, s = x
        m = torch.cat([self.up_modules[1](s), m], 1)
        l = torch.cat([self.up_modules[0](m), l], 1)
        ll = torch.cat([self.up_modules[2](l), ll], 1)
        # out=self.out_proj(l)

        l = torch.cat([self.down_modules[2](ll), l], 1)

        m = torch.cat([self.down_modules[0](l), m], 1)

        s = torch.cat([self.down_modules[1](m), s], 1)

        # top prpj and bot proj
        top_feat = self.top_proj(s)
        # mid_feat = self.mid_proj(m)
        # bot_feat = self.bot_proj(l)
        return [top_feat, top_feat, top_feat]
class AdaptiveFeatureSelection(nn.Module):
    ''' AdaptiveFeatureSelection '''

    def __init__(self, down_num,down_ins,up_num,up_ins,cur_in,lang_in,hiddens,outs):
        super().__init__()
        self.afs_modules=[]
        for i in range(down_num):
            self.afs_modules.append(FeatureNormalize(down_ins[i],hiddens,outs,down_sample=True,scale_factor=2**(down_num-i)))
        self.afs_modules.append(FeatureNormalize(cur_in,hiddens,outs))
        for i in range(up_num):
            self.afs_modules.append(FeatureNormalize(up_ins[i],hiddens,outs,up_sample=True,scale_factor=2**(i+1)))
        self.afs_modules=nn.ModuleList(self.afs_modules)
        self.afs_weights=nn.Linear(lang_in,down_num+1+up_num)
    def forward(self, *input):
        lang=input[0]
        visuals=input[1]
        v_len=len(visuals)

        for i in range(v_len):
            visuals[i]=self.afs_modules[i](visuals[i]).unsqueeze(-1)
        v_size=visuals[0].size()
        visuals=torch.cat(visuals,-1).permute(0,4,1,2,3).contiguous().view(v_size[0],v_len,-1)

        weights=self.afs_weights(lang)
        weights=F.softmax(weights,dim=-1).unsqueeze(1)
        outputs=torch.bmm(weights,visuals).view(v_size[:-1])
        return outputs

class FeatureNormalize(nn.Module):
    ''' FeatureNormalize '''

    def __init__(self,ins,hiddens,outs,down_sample=False,up_sample=False,scale_factor=1.):
        super().__init__()
        self.normalize=None
        if down_sample:
            self.normalize=nn.AvgPool2d(scale_factor)
        elif up_sample:
            self.normalize = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv1=nn.Conv2d(ins, hiddens, 3, padding=1)
        self.norm1=nn.BatchNorm2d(hiddens)
        self.act1=nn.LeakyReLU(0.1, inplace=True)
        self.conv2=nn.Conv2d(hiddens, outs, 1)
        self.norm2=nn.BatchNorm2d(outs)
        self.act2=nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        if self.normalize is not None:
            x=self.normalize(x)
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.act2(x)
        return x

class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024] 
    # [256, 128, 128, 256, 256, 512]
    
    def __init__(
        self,
        channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512],
        num_repeats=[1,1,1,1,1,1,1,1,1],
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None  

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[5]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[5], channels_list[6]], # 256, 128
            out_channels=channels_list[6], # 128
        )
      
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )
        

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs

class TSRMFusion(nn.Module):
    def __init__(self,v_planes=1024,q_planes=1024,out_planes=1024):
        super().__init__()

        self.v_trans = nn.Sequential(
            nn.Conv2d(v_planes, out_planes, 1),
            nn.Tanh(),
        )
        self.t_trans = nn.Sequential(
            nn.Linear(q_planes, out_planes),
            nn.Tanh(),
        )
        self.f_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_planes, v_planes, 1),
            nn.BatchNorm2d(v_planes),
            nn.ReLU()
        )

        
    def forward(self, v_feat, t_feat):

        t_mask = torch.ones(v_feat.shape[0],t_feat.shape[2]).to("cuda")
        #new clim
        vis_feats = self.v_trans(v_feat)
        lang_feats = self.t_trans(t_feat)
        #torch.div 就是直接除以一个数字。
        sent_feat = torch.div(torch.sum(lang_feats, 1), t_mask).unsqueeze(2).unsqueeze(3)
        # expand_as()函数用来扩展张量中某维数据的尺寸,括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。
        vis_feats = self.f_out(vis_feats * sent_feat.expand_as(vis_feats))
        vis_feats = F.normalize(v_feat + vis_feats, p=2, dim=1)
        return vis_feats

# class TSRMFusion(nn.Module):#change out=512
#     def __init__(self,v_planes=1024,q_planes=1024,out_planes=1024):
#         super().__init__()

#         self.v_trans = nn.Sequential(
#             nn.Conv2d(v_planes, out_planes, 1),
#             nn.Tanh(),
#         )
#         self.t_trans = nn.Sequential(
#             nn.Linear(q_planes, out_planes),
#             nn.Tanh(),
#         )
#         self.f_out = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(out_planes, out_planes, 1),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU()
#         )

        
#     def forward(self, v_feat, t_feat):

#         t_mask = torch.ones(v_feat.shape[0],t_feat.shape[2]).to("cuda")
#         #new clim
#         vis_feats = self.v_trans(v_feat)
#         lang_feats = self.t_trans(t_feat)
#         #torch.div 就是直接除以一个数字。
#         sent_feat = torch.div(torch.sum(lang_feats, 1), t_mask).unsqueeze(2).unsqueeze(3)
#         # expand_as()函数用来扩展张量中某维数据的尺寸,括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。
#         vis_feats2 = self.f_out(vis_feats * sent_feat.expand_as(vis_feats))
#         vis_feats = F.normalize(vis_feats + vis_feats2, p=2, dim=1)
#         return vis_feats

#shape buxinagtong
# class LFPN(nn.Module):
#     def __init__(self,v_planes=[256,512,1024], hiden_planes=512, scaled=True):
#         super().__init__()
#         self.up_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-2], v_planes[-3], ksize=1),
#                 darknet_conv(v_planes[-3], v_planes[-3], 3),
#             ),
#             nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-1], v_planes[-2], ksize=1),
#                 darknet_conv(v_planes[-2], v_planes[-2], 3)
#             )]
#         )

#         self.down_modules=nn.ModuleList(
#             [nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-3], v_planes[-2], ksize=1),
#                 darknet_conv(v_planes[-2], v_planes[-2], 3),
#             ),
#             nn.Sequential(
#                 nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
#                 darknet_conv(v_planes[-2], v_planes[-1], ksize=1),
#                 darknet_conv(v_planes[-1], v_planes[-1], 3),
#             )]
#         )


#         self.top_proj=darknet_conv(v_planes[0]+v_planes[0],v_planes[0],1) #l
#         self.mid_proj=darknet_conv(v_planes[1]+v_planes[1]+v_planes[1],v_planes[1],1) #m
#         self.bot_proj=darknet_conv(v_planes[2]+v_planes[2],v_planes[2],1) #s

#     def forward(self, x):
#         # l torch.Size([16, 256, 68, 68])    'down [16, 512, 34, 34]'
#         # m torch.Size([16, 512, 34, 34])    'down [16, 1024, 17, 17] up[16, 256, 68, 68]'
#         # s torch.Size([16, 1024, 17, 17])   'up [16, 512, 34, 34]'
#         l,m,s=x
#         # print(l.shape,m.shape,s.shape)

#         s_up = self.up_modules[1](s) #[16, 512, 34, 34]
#         m_up = self.up_modules[0](m) #[16, 256, 68, 68]
#         m_down = self.down_modules[1](m) #[16, 1024, 17, 17]
#         l_down = self.down_modules[0](l) #[16, 512, 34, 34]

#         s_next = torch.cat([s,m_down],1) #[16, 1024, 17, 17]
#         m_next = torch.cat([m,l_down,s_up],1) #[16, 512, 34, 34]
#         l_next = torch.cat([l,m_up],1) #[16, 256, 68, 68]

#         l_next = self.top_proj(l_next)
#         m_next = self.mid_proj(m_next)
#         s_next = self.bot_proj(s_next)

#         return [l_next,m_next,s_next]

class LFPN(nn.Module):
    def __init__(self,v_planes=[512,512,512], hiden_planes=512, scaled=True):
        super().__init__()
        self.up_modules=nn.ModuleList(
            [nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            ),
            nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-1], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3)
            )]
        )

        self.down_modules=nn.ModuleList(
            [nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-3], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            ),
            nn.Sequential(
                nn.AvgPool2d(2, 2) if scaled else nn.Sequential(),
                darknet_conv(v_planes[-2], hiden_planes//2, ksize=1),
                darknet_conv(hiden_planes//2, hiden_planes//2, 3),
            )]
        )


        self.top_proj=darknet_conv(v_planes[0]+hiden_planes//2,hiden_planes,1) #l
        self.mid_proj=darknet_conv(v_planes[1]+hiden_planes,hiden_planes,1) #m
        self.bot_proj=darknet_conv(v_planes[2]+hiden_planes//2,hiden_planes,1) #s

    def forward(self, x):
        # l torch.Size([16, 512, 68, 68])    'down [16, 256, 34, 34]'
        # m torch.Size([16, 512, 34, 34])    'down [16, 256, 17, 17] up[16, 256, 68, 68]'
        # s torch.Size([16, 512, 17, 17])   'up [16, 256, 34, 34]'
        l,m,s=x
        # print(l.shape,m.shape,s.shape)

        s_up = self.up_modules[1](s) #[16, 256, 34, 34]
        m_up = self.up_modules[0](m) #[16, 256, 68, 68]
        m_down = self.down_modules[1](m) #[16, 256, 17, 17]
        l_down = self.down_modules[0](l) #[16, 256, 34, 34]

        # s_next = (s+m_down)/2 #[16, 1024, 17, 17]
        # m_next = (m+l_down+s_up)/3 #[16, 512, 34, 34]
        # l_next = (l+m_up)/2 #[16, 256, 68, 68]
        s_next = torch.concat([s,m_down],1) #[16, 512 + 256, 17, 17]
        m_next = torch.concat([m,l_down,s_up],1) #[16, 1024, 34, 34]
        l_next = torch.concat([l,m_up],1) #[16, 512 + 256, 68, 68]

        l_next = self.top_proj(l_next)
        m_next = self.mid_proj(m_next)
        s_next = self.bot_proj(s_next)

        return [l_next,m_next,s_next]