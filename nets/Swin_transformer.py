# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def _no_grad_trunc_normal_(tensor, mean, std, a, b):
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()

            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            tensor.clamp_(min=a, max=b)
            return tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式
#--------------------------------------#
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))

#-------------------------------------------------------#
#   对输入进来的图片进行高和宽的压缩
#   并且进行通道的扩张。
#-------------------------------------------------------#
class PatchEmbed(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # [224, 224]
        self.img_size           = img_size
        # [4, 4]
        self.patch_size         = [patch_size, patch_size]
        # [56, 56]
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]

        # 3136
        self.num_patches        = self.patches_resolution[0] * self.patches_resolution[1]
        # 3
        self.in_chans           = in_chans
        # 96
        self.embed_dim          = embed_dim

        #-------------------------------------------------------#
        #   bs, 224, 224, 3 -> bs, 56, 56, 96
        #-------------------------------------------------------#
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]} * {self.img_size[1]})."
        #-------------------------------------------------------#
        #   bs, 224, 224, 3 -> bs, 56, 56, 96 -> bs, 3136, 96
        #-------------------------------------------------------#
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

def window_partition(x, window_size):
    B, H, W, C  = x.shape
    #------------------------------------------------------------------#
    #   bs, 56, 56, 96 -> bs, 8, 7, 8, 7, 96 -> bs * 64, 7, 7, 96
    #------------------------------------------------------------------#
    x           = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows     = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    #------------------------------------------------------------------#
    #   bs * 64, 7, 7, 96 -> bs, 8, 8, 7, 7, 96 -> bs, 56, 56, 96
    #------------------------------------------------------------------#
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim            = dim
        self.window_size    = window_size  # Wh, Ww
        self.num_heads      = num_heads
        head_dim            = dim // num_heads
        self.scale          = qk_scale or head_dim ** -0.5

        #--------------------------------------------------------------------------#
        #   相对坐标矩阵，用于表示每个窗口内，其它点相对于自己的坐标
        #   由于相对坐标取值范围为-6 ~ +6。中间共13个值，因此需要13 * 13
        #   13 * 13, num_heads
        #--------------------------------------------------------------------------#
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        ) 
        
        #--------------------------------------------------------------------------#
        #   该部分用于获取7x7的矩阵内部，其它特征点相对于自身相对坐标
        #--------------------------------------------------------------------------#
        coords_h    = torch.arange(self.window_size[0])
        coords_w    = torch.arange(self.window_size[1])
        coords      = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0]    += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1]    += self.window_size[1] - 1
        relative_coords[:, :, 0]    *= 2 * self.window_size[1] - 1
        relative_position_index     = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        #--------------------------------------------------------------------------#
        #   乘积获得q、k、v，用于计算多头注意力机制
        #--------------------------------------------------------------------------#
        self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C    = x.shape
        #--------------------------------------------------------------------------#
        #   bs * 64, 49, 96 -> bs * 64, 49, 96 * 3 -> 
        #   bs * 64, 49, 3, num_heads, 32 -> 3, bs * 64, num_head, 49, 32    
        #--------------------------------------------------------------------------#
        qkv         = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #--------------------------------------------------------------------------#
        #   bs * 64, num_head, 49, 32   
        #--------------------------------------------------------------------------#
        q, k, v     = qkv[0], qkv[1], qkv[2] 

        #--------------------------------------------------------------------------#
        #   bs * 64, num_head, 49, 49
        #--------------------------------------------------------------------------#
        q       = q * self.scale
        attn    = (q @ k.transpose(-2, -1))

        #--------------------------------------------------------------------------#
        #   这一步是根据已经求得的注意力，加上相对坐标的偏执量
        #   形成最后的注意力
        #--------------------------------------------------------------------------#
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        #--------------------------------------------------------------------------#
        #   加上mask，保证分区。
        #   bs * 64, num_head, 49, 49
        #--------------------------------------------------------------------------#
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        #---------------------------------------------------------------------------------------#
        #   bs * 64, num_head, 49, 49 @ bs * 64, num_head, 49, 32 -> bs * 64, num_head, 49, 32
        #    
        #   bs * 64, num_head, 49, 32 -> bs * 64, 49, 96
        #---------------------------------------------------------------------------------------#
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


#-------------------------------------------------------#
#   两次全连接
#-------------------------------------------------------#
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#-------------------------------------------------------#
#   每个阶段重复的基础模块
#   在这其中会使用WindowAttention进行特征提取
#-------------------------------------------------------#
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim                = dim
        self.input_resolution   = input_resolution
        self.num_heads          = num_heads
        self.window_size        = window_size
        self.shift_size         = shift_size

        self.mlp_ratio          = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1  = norm_layer(dim)
        self.attn   = WindowAttention(
            dim, 
            window_size = [self.window_size, self.window_size], 
            num_heads   = num_heads,
            qkv_bias    = qkv_bias, 
            qk_scale    = qk_scale, 
            attn_drop   = attn_drop, 
            proj_drop   = drop
        )

        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2      = norm_layer(dim)
        mlp_hidden_dim  = int(dim * mlp_ratio)
        self.mlp        = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            #----------------------------------------------------------------#
            #   由于进行特征提取时，会对输入的特征层进行的平移
            #   如：
            #   [                                   [
            #       [1, 2, 3],                          [5, 6, 4],   
            #       [4, 5, 6],          -->             [8, 9, 7],
            #       [7, 8, 9],                          [1, 2, 3],
            #   ]                                   ]
            #   这一步的作用就是使得平移后的区域块只计算自己部分的注意力机制
            #----------------------------------------------------------------#
            H, W = self.input_resolution
            _H, _W  =  _make_divisible(H, self.window_size), _make_divisible(W, self.window_size),
            img_mask = torch.zeros((1, _H, _W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask       = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask       = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask  = attn_mask.cpu().numpy()
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        #-----------------------------------------------#
        #   bs, 3136, 96 -> bs, 56, 56, 96
        #-----------------------------------------------#
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        _H, _W  =  _make_divisible(H, self.window_size), _make_divisible(W, self.window_size),
        x       = x.permute(0, 3, 1, 2)
        x       = F.interpolate(x, [_H, _W], mode='bicubic', align_corners=False).permute(0, 2, 3, 1)

        #-----------------------------------------------#
        #   进行特征层的平移
        #-----------------------------------------------#
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        #------------------------------------------------------------------------------------------#
        #   bs, 56, 56, 96 -> bs * 64, 7, 7, 96 -> bs * 64, 49, 96
        #------------------------------------------------------------------------------------------#
        x_windows = window_partition(shifted_x, self.window_size)  # num_windows * B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        #-----------------------------------------------#
        #   bs * 64, 49, 97 -> bs * 64, 49, 97
        #-----------------------------------------------#
        if type(self.attn_mask) != type(None):
            attn_mask = torch.tensor(self.attn_mask).cuda() if x.is_cuda else torch.tensor(self.attn_mask)
        else:
            attn_mask = None
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        #-----------------------------------------------#
        #   bs * 64, 49, 97 -> bs, 56, 56, 96
        #-----------------------------------------------#
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, _H, _W)  # B H' W' C

        #-----------------------------------------------#
        #   将特征层平移回来
        #-----------------------------------------------#
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, [H, W], mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
        #-----------------------------------------------#
        #   bs, 3136, 96
        #-----------------------------------------------#
        x = x.view(B, H * W, C)
        #-----------------------------------------------#
        #   FFN
        #   bs, 3136, 96
        #-----------------------------------------------#
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

#-------------------------------------------------------#
#   对输入进来的特征层进行高和宽的压缩
#   进行跨特征点的特征提取，提取完成后进行堆叠。
#-------------------------------------------------------#
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution   = input_resolution
        self.dim                = dim

        self.norm               = norm_layer(4 * dim)
        self.reduction          = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        #-------------------------------------------------------#
        #   bs, 3136, 96 -> bs, 56, 56, 96
        #-------------------------------------------------------#
        x = x.view(B, H, W, C)

        #-------------------------------------------------------#
        #   x0 ~ x3   bs, 56, 56, 96 -> bs, 28, 28, 96
        #-------------------------------------------------------#
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        #-------------------------------------------------------#
        #   4 X bs, 28, 28, 96 -> bs, 28, 28, 384
        #-------------------------------------------------------#
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        #-------------------------------------------------------#
        #   bs, 28, 28, 384 -> bs, 784, 384
        #-------------------------------------------------------#
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        #-------------------------------------------------------#
        #   bs, 784, 384 -> bs, 784, 192
        #-------------------------------------------------------#
        x = self.norm(x)
        x = self.reduction(x)
        return x


#-------------------------------------------------------#
#   Swin-Transformer的基础模块。
#   使用窗口多头注意力机制进行特征提取。
#   使用PatchMerging进行高和宽的压缩。
#-------------------------------------------------------#
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        #-------------------------------------------------------#
        #   四个阶段对应不同的dim
        #   [96, 192, 384, 768]
        #-------------------------------------------------------#
        self.dim                = dim
        #-------------------------------------------------------#
        #   四个阶段对应不同的输入分辨率
        #   [[56, 56], [28, 28], [14, 14], [7, 7]]
        #-------------------------------------------------------#
        self.input_resolution   = input_resolution
        #-------------------------------------------------------#
        #   四个阶段对应不同的多头注意力机制重复次数  
        #   [2, 2, 6, 2]
        #-------------------------------------------------------#
        self.depth              = depth
        self.use_checkpoint     = use_checkpoint

        #-------------------------------------------------------#
        #   根据depth的次数利用窗口多头注意力机制进行特征提取。
        #-------------------------------------------------------#
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim         = dim, 
                    input_resolution = input_resolution,
                    num_heads   = num_heads, 
                    window_size = window_size,
                    shift_size  = 0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio   = mlp_ratio,
                    qkv_bias    = qkv_bias, 
                    qk_scale    = qk_scale,
                    drop        = drop, 
                    attn_drop   = attn_drop,
                    drop_path   = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer  = norm_layer
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            #-------------------------------------------------------#
            #   判断是否要进行下采样，即：高宽压缩
            #-------------------------------------------------------#
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x_ = checkpoint.checkpoint(blk, x)
            else:
                x_ = blk(x)
        if self.downsample is not None:
            x = self.downsample(x_)
        else:
            x = x_
        return x_, x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=[640, 640], patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes    = num_classes
        self.num_layers     = len(depths)
        self.embed_dim      = embed_dim
        self.ape            = ape
        self.patch_norm     = patch_norm
        self.num_features   = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio      = mlp_ratio
        
        #--------------------------------------------------#
        #   bs, 224, 224, 3 -> bs, 3136, 96
        #--------------------------------------------------#
        self.patch_embed = PatchEmbed(
            img_size    = img_size, 
            patch_size  = patch_size,
            in_chans    = in_chans, 
            embed_dim   = embed_dim,
            norm_layer  = norm_layer if self.patch_norm else None
        )

        #--------------------------------------------------#
        #   PatchEmbed之后的图像序列长度        3136
        #   PatchEmbed之后的图像对应的分辨率    [56, 56]
        #--------------------------------------------------#
        num_patches             = self.patch_embed.num_patches
        patches_resolution      = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        #--------------------------------------------------#
        #   stochastic depth
        #--------------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        #---------------------------------------------------------------#
        #   构建swin-transform的每个阶段
        #   bs, 3136, 96 -> bs, 784, 192 -> bs, 196, 384 -> bs, 49, 768
        #---------------------------------------------------------------#
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim                 = int(embed_dim * 2 ** i_layer),
                input_resolution    = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth               = depths[i_layer],
                num_heads           = num_heads[i_layer],
                window_size         = window_size,
                mlp_ratio           = self.mlp_ratio,
                qkv_bias            = qkv_bias, 
                qk_scale            = qk_scale,
                drop                = drop_rate, 
                attn_drop           = attn_drop_rate,
                drop_path           = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer          = norm_layer,
                downsample          = PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint      = use_checkpoint
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        inverval_outs = []
        for i, layer in enumerate(self.layers):
            x_, x = layer(x)
            if i != 0:
                inverval_outs.append(x_)
        
        outs = []
        for i, layer in enumerate(inverval_outs):
            H, W    = (self.patches_resolution[0] // (2 ** (i + 1)), self.patches_resolution[1] // (2 ** (i + 1)))
            B, L, C = layer.shape
            layer   = layer.view([B, H, W, C]).permute([0, 3, 1, 2])
            outs.append(layer)

        return outs
    
def Swin_transformer_Tiny(pretrained = False, input_shape = [640, 640], **kwargs):
    model = SwinTransformer(input_shape, depths=[2, 2, 6, 2], **kwargs)
    if pretrained:
        url = "https://github.com/bubbliiiing/yolov5-pytorch/releases/download/v1.0/swin_tiny_patch4_window7.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        model.load_state_dict(checkpoint, strict=False)
        print("Load weights from ", url.split('/')[-1])
        
    return model