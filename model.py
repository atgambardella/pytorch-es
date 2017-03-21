# Taken from https://github.com/ikostrikov/pytorch-a4c
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    """
    Not actually using this but let's keep it here in case that changes
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space, small_net=False):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        num_outputs = action_space.n
        self.small_net = small_net
        if self.small_net:
            self.linear1 = nn.Linear(num_inputs, 64)
            self.linear2 = nn.Linear(64, 64)
            self.actor_linear = nn.Linear(64, num_outputs)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.lstm = nn.LSTMCell(32*3*3, 256)
            self.actor_linear = nn.Linear(256, num_outputs)
        self.train()

    def forward(self, inputs):
        if self.small_net:
            x = F.elu(self.linear1(inputs))
            x = F.elu(self.linear2(x))
            return self.actor_linear(x)
        else:
            inputs, (hx, cx) = inputs
            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = x.view(-1, 32*3*3)
            hx, cx = self.lstm(x, (hx, cx))
            x = hx
            return self.actor_linear(x), (hx, cx)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]
