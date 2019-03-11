# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

class MaskResNet6(nn.Module):

    def __init__(self, nb_ref_imgs=4, output_exp=True):
        super(MaskResNet6, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7, stride=2)
        self.conv2 = make_layer(conv_planes[0], BasicBlock, conv_planes[1], blocks=2, stride=2)
        self.conv3 = make_layer(conv_planes[1], BasicBlock, conv_planes[2], blocks=2, stride=2)
        self.conv4 = make_layer(conv_planes[2], BasicBlock, conv_planes[3], blocks=2, stride=2)
        self.conv5 = make_layer(conv_planes[3], BasicBlock, conv_planes[4], blocks=2, stride=2)
        self.conv6 = make_layer(conv_planes[4], BasicBlock, conv_planes[5], blocks=2, stride=2)

        if self.output_exp:
            upconv_planes = [256, 256, 128, 64, 32, 16]
            self.deconv6 = upconv(conv_planes[5], upconv_planes[0])
            self.deconv5 = upconv(upconv_planes[0]+conv_planes[4], upconv_planes[1])
            self.deconv4 = upconv(upconv_planes[1]+conv_planes[3], upconv_planes[2])
            self.deconv3 = upconv(upconv_planes[2]+conv_planes[2], upconv_planes[3])
            self.deconv2 = upconv(upconv_planes[3]+conv_planes[1], upconv_planes[4])
            self.deconv1 = upconv(upconv_planes[4]+conv_planes[0], upconv_planes[5])

            self.pred_mask6 = nn.Conv2d(upconv_planes[0], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask5 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask4 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask3 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask2 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.pred_mask1 = nn.Conv2d(upconv_planes[5], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_mask_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for module in [self.pred_mask1, self.pred_mask2, self.pred_mask3, self.pred_mask4, self.pred_mask5, self.pred_mask6]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()



    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        if self.output_exp:
            out_upconv6 = self.deconv6(out_conv6  )#[:, :, 0:out_conv5.size(2), 0:out_conv5.size(3)]
            out_upconv5 = self.deconv5(torch.cat((out_upconv6, out_conv5), 1))#[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.deconv4(torch.cat((out_upconv5, out_conv4), 1))#[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.deconv3(torch.cat((out_upconv4, out_conv3), 1))#[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.deconv2(torch.cat((out_upconv3, out_conv2), 1))#[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.deconv1(torch.cat((out_upconv2, out_conv1), 1))#[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask6 = nn.functional.sigmoid(self.pred_mask6(out_upconv6))
            exp_mask5 = nn.functional.sigmoid(self.pred_mask5(out_upconv5))
            exp_mask4 = nn.functional.sigmoid(self.pred_mask4(out_upconv4))
            exp_mask3 = nn.functional.sigmoid(self.pred_mask3(out_upconv3))
            exp_mask2 = nn.functional.sigmoid(self.pred_mask2(out_upconv2))
            exp_mask1 = nn.functional.sigmoid(self.pred_mask1(out_upconv1))
        else:
            exp_mask6 = None
            exp_mask5 = None
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return exp_mask1, exp_mask2, exp_mask3, exp_mask4, exp_mask5, exp_mask6
        else:
            return exp_mask1
