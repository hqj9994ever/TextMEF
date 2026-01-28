import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers
import numpy as np

from .cga import ChannelAttention, PixelAttention, SpatialAttention
from model.mamba_simple import Mamba

from utils.utils import to_3d, to_4d, weights_init


def define_G(in_channels):
    netG = FusionNet(base_filter=in_channels)
    netG.apply(weights_init)
    return netG


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
        


class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # BNC->BCHW
        return x
    


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
    

class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim,bimamba_type=None)
        self.norm = LayerNorm(dim,'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x,residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)
    

class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.underencoder = Mamba(dim,bimamba_type=None)
        self.overencoder = Mamba(dim,bimamba_type=None)
        self.norm1 = LayerNorm(dim,'with_bias')
        self.norm2 = LayerNorm(dim,'with_bias')
    def forward(self, under,over,under_residual,over_residual):
        # under (B,N,C)
        #over (B,N,C)
        under_residual = under+under_residual
        over_residual = over+over_residual
        under = self.norm1(under_residual)
        over = self.norm2(over_residual)
        B,N,C = under.shape
        under_first_half = under[:, :, :C//2]
        over_first_half = over[:, :, :C//2]
        under_swap= torch.cat([over_first_half,under[:,:,C//2:]],dim=2)
        over_swap= torch.cat([under_first_half,over[:,:,C//2:]],dim=2)
        under_swap = self.underencoder(under_swap)
        over_swap = self.overencoder(over_swap)
        return under_swap,over_swap,under_residual,over_residual
    

class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim,bimamba_type="v2")
        self.norm = LayerNorm(dim,'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
    def forward(self,under,under_resi,x_size):
        under_resi = under+under_resi
        under = self.norm(under_resi)
        global_f = self.cross_mamba(self.norm(under))
        B,HW,C = global_f.shape
        under = global_f.transpose(1, 2).view(B, C, x_size[0], x_size[1])
        under =  (self.dwconv(under)+under).flatten(2).transpose(1, 2)
        return under,under_resi
    

class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z * res + x


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            ChannelAttention(n_feat, 3))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class AttenionNet(torch.nn.Module):
    def __init__(self, dim):
        super(AttenionNet, self).__init__()

        self.embed_dim=dim
        self.stride=1
        self.patch_size=1

        self.fe1 = torch.nn.Conv2d(64, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.mamba_1 = SingleMambaBlock(self.embed_dim)
        self.mamba_2 = SingleMambaBlock(self.embed_dim)
        self.max_to_token = PatchEmbed(in_chans=64,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.avg_to_token = PatchEmbed(in_chans=64,embed_dim=self.embed_dim,patch_size=self.patch_size,stride=self.stride)
        self.patchunembe = PatchUnEmbed(basefilter=64)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.sAtt_4 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_5 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)

        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_L2 = torch.nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.sAtt_L3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, alignedframe):

        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))

        # spatial attention
        att = self.lrelu(self.sAtt_1(att))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        b,c,h,w = att_max.shape
        att_max_token = self.max_to_token(att_max)
        att_avg_token = self.avg_to_token(att_avg)
        att_max_token, _ = self.mamba_1([att_max_token,0])
        att_avg_token, _ = self.mamba_2([att_avg_token,0])
        att_max = self.patchunembe(att_max_token,(h,w))
        att_avg = self.patchunembe(att_avg_token,(h,w))
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L,size=[att.size(2), att.size(3)],
                              mode='bilinear', align_corners=False)


        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att,size=[alignedframe.size(2), alignedframe.size(3)],
                            mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att = torch.sigmoid(att)

        return att
    

class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

  
class FusionNet(nn.Module):
    def __init__(self, base_filter=64, args=None):
        super(FusionNet, self).__init__()
        self.base_filter = base_filter
        self.stride=1
        self.patch_size=1
        self.over_encoder = nn.Sequential(nn.Conv2d(3, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter))
        self.under_encoder = nn.Sequential(nn.Conv2d(3, base_filter, 3, 1, 1), HinResBlock(base_filter,base_filter))

        self.embed_dim = base_filter * self.stride * self.patch_size
        self.attention_1 = AttenionNet(self.embed_dim)
        self.attention_2 = AttenionNet(self.embed_dim)
        self.under_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.over_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim,patch_size=self.patch_size, stride=self.stride)
        self.f_to_token = PatchEmbed(in_chans=base_filter, embed_dim=self.embed_dim, patch_size=self.patch_size, stride=self.stride)
        self.deepfusion1= CrossMamba(self.embed_dim)
        self.deepfusion2 = CrossMamba(self.embed_dim)
        self.deepfusion3 = CGAFusion(self.embed_dim)

        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        self.patchunembe = PatchUnEmbed(base_filter)
        self.refine = Refine(base_filter, 3)

    def forward(self, under, over):

        under_f = self.under_encoder(under)
        _, _, H, W = under_f.shape
        over_f = self.over_encoder(over)

        residual_under_f = 0
        residual_over_f = 0
        under_f = self.attention_1(under_f) * under_f
        over_f = self.attention_2(over_f) * over_f
        
        under_f = self.under_to_token(under_f)
        over_f = self.over_to_token(over_f)
        under_f, over_f, residual_under_f, residual_over_f = self.swap_mamba1(under_f, over_f, residual_under_f, residual_over_f)
        under_f, over_f, residual_under_f, residual_over_f = self.swap_mamba2(under_f, over_f, residual_under_f, residual_over_f)
        under_f, residual_under_f = self.deepfusion1(under_f, residual_under_f, (H, W))
        over_f, residual_over_f = self.deepfusion2(over_f, residual_over_f, (H, W))
        under_f = self.patchunembe(under_f, (H, W))
        over_f = self.patchunembe(over_f, (H, W))
        f = self.deepfusion3(under_f,over_f)
        output = self.refine(f) + under + over

        return output

