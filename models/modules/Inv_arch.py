import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_util import initialize_weights_xavier
from torch.nn import init
import cv2
from basicsr.archs.arch_util import flow_warp
from models.modules.Subnet_constructor import subnet
import numpy as np

def thops_mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64, model='MIMO-VRN'):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
        # but this is how we trained the model in the first place and what we reported in the paper
        if model == 'LSTM-VRN':
            self.relu = nn.ReLU(inplace=True)
        elif model == 'MIMO-VRN':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, subnet_constructor_v2, channel_num_ho, channel_num_hi, clamp=1.):
        super(InvBlock, self).__init__()
        self.split_len1 = channel_num_ho  # channel_split_num
        self.split_len2 = channel_num_hi  # channel_num - channel_split_num
        self.clamp = clamp

        self.F = subnet_constructor_v2(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        if not rev:
            t2 = self.F(x2)
            y1 = x1 + t2
            s1, t1 = self.H(y1), self.G(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.H(x1), self.G(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.F(y2)
            y1 = (x1 - t2)

        return y1, y2  # torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class InvNN(nn.Module):
    def __init__(self, channel_in_ho=3, channel_in_hi=3, subnet_constructor=None, subnet_constructor_v2=None, block_num=[], down_num=2):
        super(InvNN, self).__init__()
        operations = []
#         current_channel = channel_in
        current_channel_ho = channel_in_ho
        current_channel_hi = channel_in_hi
        for i in range(1):
            for j in range(block_num[i]):
                b = InvBlock(subnet_constructor, subnet_constructor_v2, current_channel_ho, current_channel_hi)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, x_h, rev=False, cal_jacobian=False):
        # 		out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)
        else:
            for op in reversed(self.operations):
                x, x_h = op.forward(x, x_h, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(x, rev)

        if cal_jacobian:
            return x, x_h, jacobian
        else:
            return x, x_h

class PredictiveModuleMIMO(nn.Module):
    def __init__(self, channel_in, nf, block_num_rbm=8):
        super(PredictiveModuleMIMO, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, nf, 3, 1, 1, bias=True)
        residual_block = []
        for i in range(block_num_rbm):
            residual_block.append(ResidualBlockNoBN(nf))
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        x = self.conv_in(x)
        res = self.residual_block(x)

        return res

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def gauss_noise_mul(shape):
    noise = torch.randn(shape).cuda()

    return noise

class VSN(nn.Module):
    def __init__(self, opt, subnet_constructor=None, subnet_constructor_v2=None, down_num=2):
        # down_num이 뭔지는 모름
        # in_nc : 
        super(VSN, self).__init__()
        opt_net = opt['network_G']
        self.channel_in = opt_net['in_hi_nc'] 
        self.channel_in_hi = opt_net['in_hi_nc'] 
        self.channel_in_ho = opt_net['in_ho_nc']

        self.block_num = opt_net['block_num']
        self.block_num_rbm = opt_net['block_num_rbm']
        self.nf = self.channel_in_hi  
        self.irn = InvNN(self.channel_in_ho, self.channel_in_hi, subnet_constructor, subnet_constructor_v2, self.block_num, down_num)
        self.pm = PredictiveModuleMIMO(self.channel_in_ho, self.nf, block_num_rbm=self.block_num_rbm)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_h=None, rev=False, hs=[], direction='f', gt_mask=None):
        if not rev:
            out_y, out_y_h = self.irn(x, x_h, rev)
            return out_y, out_y_h
        else:
            if x_h is None:
                out_z = self.pm(x).unsqueeze(1)
                out_z_new = out_z.view(-1, self.channel_in, x.shape[-2], x.shape[-1]) 
                out_x, out_x_h = self.irn(x, out_z_new, rev)

                return out_x, out_x_h, out_z_new

            else:
                out_z = x_h
                out_x, out_x_h = self.irn(x, out_z, rev)

                return out_x, pred_message, out_z
