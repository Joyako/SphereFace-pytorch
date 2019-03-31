# -*- coding:utf-8 -*-

import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.autograd import Variable


__all__ = ['SphereFace4', 'SphereFace10', 'SphereFace20', 'SphereFace36', 'SphereFace64']

cfg = {
    'A': [0, 0, 0, 0],
    'B': [0, 1, 2, 0],
    'C': [1, 2, 4, 1],
    'D': [2, 4, 8, 2],
    'E': [3, 8, 16, 3]
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AngleLayer(nn.Module):
    """Convert the fully connected layer of output to """
    def __init__(self, in_planes, out_planes, m=4):
        super(AngleLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.cos_val = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x,
        ]

    def forward(self, input):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos(θ)
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1) / w_modulus.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.cos_val[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        # k * pi / m <= theta <= (k + 1) * pi / m
        k = (self.m * theta / 3.14159265).floor()
        minus_one = k * 0.0 - 1
        # Phi(yi, i) = (-1)**k * cos(myi,i) - 2 * k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k

        cos_x = cos_theta * x_modulus.view(-1, 1)
        phi_x = phi_theta * x_modulus.view(-1, 1)

        return cos_x, phi_x


class SphereFace(nn.Module):
    """
    Implement paper which is 'SphereFace: Deep Hypersphere Embedding for Face Recognition'.

    Reference:
        https://arxiv.org/abs/1704.08063
    """

    def __init__(self, block, layers, num_classes=10):
        """

        :param block: residual units.
        :param layers: number of repetitions per residual unit.
        :param num_classes:
        """
        super(SphereFace, self).__init__()
        self.conv1 = conv3x3(1, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(64, 128, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(128, 256, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv3x3(256, 512, 2)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.fc5 = nn.Linear(512 * 2 * 2, 512)
        self.fc6 = nn.Linear(512, 2)
        self.fc7 = AngleLayer(2, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks != 0:
            layers = []
            for _ in range(0, blocks):
                downsample = nn.Sequential(
                    conv1x1(planes, planes, stride),
                    nn.BatchNorm2d(planes),
                )

                layers.append(block(planes, planes, stride, downsample))

            return nn.Sequential(*layers)
        else:
            return None

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.layer1 is not None:
            x = self.layer1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.layer2 is not None:
            x = self.layer2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        if self.layer3 is not None:
            x = self.layer3(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        if self.layer4 is not None:
            x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.fc6(x)
        y = self.fc7(x)

        return x, y


def SphereFace4(**kwargs):
    """
    Constructs a SphereFace-4 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['A'], **kwargs)

    return model


def SphereFace10(**kwargs):
    """
    Constructs a SphereFace-10 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['B'], **kwargs)

    return model


def SphereFace20(**kwargs):
    """
    Constructs a SphereFace-20 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['C'], **kwargs)

    return model


def SphereFace36(**kwargs):
    """
    Constructs a SphereFace-36 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['D'], **kwargs)

    return model


def SphereFace64(**kwargs):
    """
    Constructs a SphereFace-64 model.
    :return:
    """
    model = SphereFace(BasicBlock, cfg['E'], **kwargs)

    return model

