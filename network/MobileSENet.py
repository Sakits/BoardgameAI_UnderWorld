import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')

def conv_1x1_bn(in_channels, out_channels, norm=True, inplace=False):
    """
        1x1 Convolution Block
    """
    if norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=inplace)
            )
    else:
        return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)


def conv_3x3_bn(in_channels, out_channels, stride=1, inplace=False):
    """
        3x3 Convolution Block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=inplace)
    )

def conv_5x5_bn(in_channels, out_channels, stride=1, inplace=False):
    """
        5x5 Convolution Block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=inplace)
    )

def conv_7x7_bn(in_channels, out_channels, stride=1, inplace=False):
    """
        7x7 Convolution Block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 7, stride, 3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=inplace)
    )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileBlock(nn.Module):
    def __init__(self, channels):
        super(MobileBlock, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            # pw-linear
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

        self.se = SELayer(channels, 16)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x = self.se(x)

        x += residual
        x = self.silu(x)

        return x

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt, self.board_x, self.board_y = game.getFeatureSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv = conv_5x5_bn(self.feat_cnt, args.num_channels, inplace=True)

        self.res_layers = []
        for _ in range(args.depth):
            self.res_layers.append(MobileBlock(args.num_channels))
        self.resnet = nn.Sequential(*self.res_layers)

        self.v_conv = conv_1x1_bn(args.num_channels, 1)
        self.v_fc1 = nn.Linear(self.board_x * self.board_y,
                               self.board_x * self.board_y // 2)
        self.v_fc2 = nn.Linear(self.board_x * self.board_y // 2, 1)

        self.pi_conv = conv_1x1_bn(args.num_channels, 2)
        self.pi_fc = nn.Linear(self.board_x * self.board_y * 2, self.action_size)

    def forward(self, s):
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   # batch_size x feat_cnt x board_x x board_y
        s = self.conv(s)                                            # batch_size x num_channels x board_x x board_y
        s = self.resnet(s)                                          # batch_size x num_channels x board_x x board_y

        v = self.v_conv(s)
        v = torch.flatten(v, 1)
        v = self.v_fc1(v)
        v = self.v_fc2(v)

        pi = self.pi_conv(s)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)