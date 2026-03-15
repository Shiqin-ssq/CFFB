import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics.nn.modules.conv import Conv, RepConv, DSConv
from ultralytics.nn.modules.block import *
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.Addmodules.wtconv2d import *
from ultralytics.nn.Addmodules.hcfnet import LocalGlobalAttention
# from .metaformer import *


__all__ = ['C3k2_WTConv', 'DySample', 'CFFB', 'DSC3k2', 'PSPPF']
##WTConv
class Bottleneck_WTConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = WTConv2d(c1, c2)
        self.cv2 = WTConv2d(c2, c2)

class C3k_WTConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WTConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_WTConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_WTConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WTConv(self.c, self.c, shortcut, g) for _ in range(n))

##DySample
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

# CFFB
class CFFB(nn.Module):
    # Hierarchical Attention Fusion Block
    def __init__(self, inc, ouc, group=False):
        super(CFFB, self).__init__()
        ch_1, ch_2 = inc
        hidc = ouc // 2

        self.lgb1_local = LocalGlobalAttention(hidc, 2)
        self.lgb1_global = LocalGlobalAttention(hidc, 4)
        self.lgb2_local = LocalGlobalAttention(hidc, 2)
        self.lgb2_global = LocalGlobalAttention(hidc, 4)

        self.W_x1 = Conv(ch_1, hidc, 1, act=False)
        self.W_x2 = Conv(ch_2, hidc, 1, act=False)
        self.W = DSConv(hidc, ouc, 3)
        self.WT = WTConv2d(ouc, ouc, kernel_size=5, wt_levels=2, wt_type='db2')

        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        self.DS_conv1 = DSConv(ouc, ouc, 3)
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        x1, x2 = inputs
        W_x1 = self.W_x1(x1)
        W_x2 = self.W_x2(x2)
        bp = self.W(W_x1 + W_x2)
        bpwt = self.WT(bp)

        x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        x3 = self.conv_squeeze(torch.cat([x1, x2, bpwt], 1))
        x4 = torch.cat([self.DS_conv1(x3)], dim = 1)
        return self.conv_final(x4)

# class DSBottleneck(nn.Module):
#     def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5, d2=1):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = DSConv(c1, c_, k1, s=1, p=None, d=1)
#         self.cv2 = DSConv(c_, c2, k2, s=1, p=None, d=d2)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         y = self.cv2(self.cv1(x))
#         return x + y if self.add else y
#
# class DSC3k(C3):
#     def __init__(
#         self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k1=3, k2=5, d2=1
#     ):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#
#         self.m = nn.Sequential(
#             *(
#                 DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2
#                 )
#                 for _ in range(n)
#             )
#         )
#
# class DSC3k2(C2f):
#     def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         if dsc3k:
#             self.m = nn.ModuleList(
#                 DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0,   k1=k1, k2=k2, d2=d2)
#                 for _ in range(n)
#             )
#         else:
#             self.m = nn.ModuleList(
#                 DSBottleneck( self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
#                 for _ in range(n)
#             )

# PDS-Bottleneck
class DSBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, k=3, d=1):
        super().__init__()
        c_ = int(c2 * e)
        self.pw = Conv(c1, c_, 1, s=1, p=0, act=True)
        self.cv2 = DSConv(c_, c2, k, s=1, p=None, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.pw(x))
        return x + y if self.add else y

class DSC3k(C3):
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, d=1
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(
            *(
                DSBottleneck(c_, c_, shortcut=shortcut, e=1.0, k=k, d=d
                )
                for _ in range(n)
            )
        )

class DSC3k2(C2f):
    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k=3, d=1):
        print(f"DSC3k2 init: c1={c1}, c2={c2}, n={n}, dsc3k={dsc3k}, e={e}")
        super().__init__(c1, c2, n, shortcut, g, e)
        if dsc3k:
            self.m = nn.ModuleList(
                DSC3k(self.c, self.c, n=2, shortcut=shortcut, g=g, e=1.0, k=k, d=d)
                for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                DSBottleneck( self.c, self.c, shortcut=shortcut, e=1.0, k=k, d=d)
                for _ in range(n)
            )

###########parallel SPPF
class PSPPF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.pool5 = nn.MaxPool2d(5, 1, 2)
        self.pool9 = nn.MaxPool2d(9, 1, 4)
        self.pool13 = nn.MaxPool2d(13, 1, 6)
        self.conv3x3 = nn.Conv2d(c_, c_, 3, 1, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_*5, c_*5 // 16, 1),
            nn.ReLU(),
            nn.Conv2d(c_*5 // 16, c_*5, 1),
            nn.Sigmoid()
        )
        self.cv2 = Conv(c_*5, c2, 1, 1)

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.pool5(y)
        y2 = self.pool9(y)
        y3 = self.pool13(y)
        y4 = self.conv3x3(y)
        y_cat = torch.cat([y, y1, y2, y3, y4], 1)
        attn = self.se(y_cat)
        y_cat = y_cat * attn
        return self.cv2(y_cat)

# if __name__ == "__main__":
#
#     model = DSC3k2(c1=128, c2=256, n=2, dsc3k=True)
#     x = torch.randn(2,128, 640,640)
#     output1 = model(x)
#     print(f"With DSBottleneck: {output1.shape}")