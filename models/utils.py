from __future__ import division
import math

import torch
import torch.nn as nn


def conv(in_planes, out_planes, stride=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )


def deconv(in_planes, out_planes, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )


def predict_depth(in_planes, with_confidence):
    return nn.Conv2d(in_planes, 2 if with_confidence else 1, kernel_size=3, stride=1, padding=1, bias=True)


def post_process_depth(depth, activation_function=None, clamp=False):
    if activation_function is not None:
        depth = activation_function(depth)

    if clamp:
        depth = depth.clamp(10, 80)

    return depth[:,0]


def adaptative_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)


def init_modules(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2/n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
