import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import math
from einops import rearrange

sys.path.append('..')

def conv_1x1_bn(in_channels, out_channels, norm=True):
    """
        1x1 Convolution Block
    """
    if norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
            )
    else:
        return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv_3x3_bn(in_channels, out_channels, stride=1):
    """
        3x3 Convolution Block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )


class FFN(nn.Module):
    """
        Feedforward (MLP) Block
    """
    def __init__(self, dim, hidden_dim, ffn_dropout=0.):
        super(FFN, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.block(x)


class MHSA(nn.Module):
    """
        Multi-Head Self-Attention: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0.):
        super(MHSA, self).__init__()
        assert embed_dim % num_heads == 0
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=True)

        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.out_proj(out)


class TransformerEncoder(nn.Module):
    """
        Transformer Enocder
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0., attn_dropout=0., ffn_dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MHSA(embed_dim, num_heads, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            FFN(embed_dim, mlp_dim, ffn_dropout),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.pre_norm_mha(x)
        x = x + self.pre_norm_ffn(x)
        return x


class InvertedResidual(nn.Module):
    """
        Inverted Residual Block (MobileNetv2)
    """
    def __init__(self, in_channels, out_channels, stride=1, exp_ratio=4):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        
        self.stride = stride
        hidden_dim = int(in_channels * exp_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if exp_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    """
        MobileViT Block
    """
    def __init__(self, channels, embed_dim, depth, num_heads, mlp_dim, patch_size=(2,2), dropout=0.1):
        super(MobileViTBlock, self).__init__()
        self.ph, self.pw = patch_size
        self.conv_3x3_in = conv_3x3_bn(channels, channels)
        self.conv_1x1_in = conv_1x1_bn(channels, embed_dim, norm=False)
        transformer = nn.ModuleList([])
        for i in range(depth):
            transformer.append(TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout))
        transformer.append(nn.LayerNorm(embed_dim))
        self.transformer = nn.Sequential(*transformer)

        self.conv_1x1_out = conv_1x1_bn(embed_dim, channels, norm=True)
        self.conv_3x3_out = conv_3x3_bn(2 * channels, channels)
    
    def forward(self, x):
        _, _, h, w = x.shape
        # make sure to height and width are divisible by patch size
        new_h = int(math.ceil(h / self.ph) * self.ph)
        new_w = int(math.ceil(w / self.pw) * self.pw)
        if new_h != h and new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y = x.clone()
        # Local representations
        x = self.conv_3x3_in(x)
        x = self.conv_1x1_in(x)
        # Global representations
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=new_h//self.ph, w=new_w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv_1x1_out(x)
        x = torch.cat((x, y), 1)
        x = self.conv_3x3_out(x)
        return x


class MobileViT(nn.Module):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, 
                 feat_cnt=2, 
                 image_size=(9,9), 
                 embed_dim=32,
                 num_heads=4,
                 depth=3,
                 mlp_ratio=2,
                 channels=[32,64,128], 
                 exp_ratio=4, 
                 patch_size=(3, 3),
                 num_classes=82,
                 ):

        super(MobileViT, self).__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.conv_in = conv_3x3_bn(feat_cnt, channels[0], stride=1)

        vit = nn.ModuleList([])
        vit.append(InvertedResidual(channels[0], channels[1], 1, exp_ratio)) 
        vit.append(MobileViTBlock(channels[1],
                                    embed_dim, 
                                    depth, 
                                    num_heads,
                                    int(embed_dim * mlp_ratio), 
                                    patch_size
                                    ))
        self.vit = nn.Sequential(*vit)

        self.conv_out = conv_1x1_bn(channels[1], channels[2])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.v_fc = nn.Linear(channels[2], 1)

        self.pi_conv = conv_1x1_bn(channels[2], 2)
        self.pi_fc = nn.Linear(ih * iw * 2, num_classes)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.vit(x)
        x = self.conv_out(x)
        
        v = self.pool(x).view(-1, x.shape[1])
        v = self.v_fc(v)

        pi = self.pi_conv(x)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt, self.board_x, self.board_y = game.getFeatureSize()
        self.action_size = game.getActionSize()
        self.mobile_vit = MobileViT(feat_cnt = self.feat_cnt, image_size = (self.board_x, self.board_y), num_classes = self.action_size)

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)

        return self.mobile_vit(s)