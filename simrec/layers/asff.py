import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入的是从FPN得到的特征
# 输出是要给head的特征，与之前不变，注意单个ASFF只能输出一个特征，level=0对应最底层的特征，这里是512*20*20,尺度大小 level_0 < level_1 < level_2

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [1024, 512, 256]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = self._make_cbl(512, self.inter_dim, 3, 2)
            self.stride_level_2 = self._make_cbl(256, self.inter_dim, 3, 2)
            self.expand = self._make_cbl(self.inter_dim, 1024, 3, 1)  # 输出是要给head的特征，与之前不变512-512
        elif level==1:
            self.compress_level_0 = self._make_cbl(1024, self.inter_dim, 1, 1)
            self.stride_level_2 = self._make_cbl(256, self.inter_dim, 3, 2)
            self.expand = self._make_cbl(self.inter_dim, 512, 3, 1)  # 输出是要给head的特征，与之前不变256-256
        elif level==2:
            self.compress_level_0 = self._make_cbl(1024, self.inter_dim, 1, 1)
            self.compress_level_1 = self._make_cbl(512, self.inter_dim, 1, 1)
            self.expand = self._make_cbl(self.inter_dim, 256, 3, 1)  # 输出是要给head的特征，与之前不变128-128

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = self._make_cbl(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = self._make_cbl(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = self._make_cbl(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def _make_cbl(self, _in, _out, ks, stride):
        return BaseConv(_in, _out, ks, stride, act="lrelu")

    def forward(self, x_level_0, x_level_1, x_level_2):   # 输入3个维度(512*20*20,256*40*40,128*80*80),输出也是
        if self.level==0:
            level_0_resized = x_level_0  # (512*20*20)
            level_1_resized = self.stride_level_1(x_level_1)  # (256*40*40->512*20*20)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)  # (128*80*80->128*40*40)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)  # (128*40*40->512*20*20)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)  # (512*20*20->256*20*20)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')  # (256*20*20->256*40*40)
            level_1_resized =x_level_1  # (256*40*40)
            level_2_resized =self.stride_level_2(x_level_2)  # (128*80*80->256*40*40)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)  # (512*20*20->128*20*20)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')  # (128*20*20->128*80*80)
            level_1_compressed = self.compress_level_1(x_level_1)  # (256*40*40->128*40*40)
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')  # (128*40*40->128*80*80)
            level_2_resized =x_level_2  # (128*80*80)

        level_0_weight_v = self.weight_level_0(level_0_resized)  # 
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class UpSample(nn.Module):

    def __init__(self, scale_factor=2, mode="bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.vision.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class SiLU(nn.Module):
    """export-friendly version of M.SiLU()"""

    @staticmethod
    def forward(x):
        return x * F.sigmoid(x)


def get_activation(name="silu"):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv =nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn =nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
