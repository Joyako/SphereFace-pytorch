# -*- coding:utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['AngularSoftmaxWithLoss']


class AngularSoftmaxWithLoss(nn.Module):
    """"""
    def __init__(self, gamma=0):
        super(AngularSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.iter += 1
        target = target.view(-1, 1)

        index = input[0].data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = Variable(index.byte())

        # Tricks
        # output(θyi) = (lambda * cos(θyi) + (-1) ** k * cos(m * θyi) - 2 * k)) / (1 + lambda)
        #             = cos(θyi) - cos(θyi) / (1 + lambda) + Phi(θyi) / (1 + lambda)
        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = F.log_softmax(output)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss
