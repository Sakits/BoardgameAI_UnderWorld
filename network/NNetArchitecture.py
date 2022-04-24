import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
sys.path.append('..')

def conv1x1(in_channels, out_channels, stride=1, bias = False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1, bias = False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1, bias = False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        stride = 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, groups = in_channels, bias = False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)

    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt, self.board_x, self.board_y = game.getFeatureSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = conv5x5(self.feat_cnt, args.num_channels)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers1 = []
        for _ in range(args.depth):
            self.res_layers1.append(ResidualBlock(
                args.num_channels, args.num_channels))
        self.resnet1 = nn.Sequential(*self.res_layers1)

        self.v_conv = conv1x1(args.num_channels, 1)
        self.v_bn1, self.v_bn2 = nn.BatchNorm2d(args.num_channels), nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(self.board_x*self.board_y,
                               self.board_x*self.board_y//2)
        self.v_fc2 = nn.Linear(self.board_x*self.board_y//2, 1)

        self.pi_conv = conv1x1(args.num_channels, 2)
        self.pi_bn1, self.pi_bn2 = nn.BatchNorm2d(args.num_channels), nn.BatchNorm2d(2)
        self.pi_fc1 = nn.Linear(self.board_x*self.board_y*2, self.action_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = self.conv1(s)
        # batch_size x num_channels x board_x x board_y
        s = self.resnet1(s)


        v = self.v_bn1(s)
        v = torch.tanh(v)
        v = self.v_conv(v)

        v = self.v_bn2(v)
        v = torch.tanh(v)
        v = torch.flatten(v, 1)
        v = self.v_fc1(v)

        v = torch.tanh(v)
        v = self.v_fc2(v)


        pi = self.pi_bn1(s)
        pi = self.relu(pi)
        pi = self.pi_conv(pi)

        pi = self.pi_bn2(pi)
        pi = self.relu(pi)
        pi = torch.flatten(pi, 1)
        pi = self.pi_fc1(pi)

        return F.log_softmax(pi, dim=1), torch.tanh(v)