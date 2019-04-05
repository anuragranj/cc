# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from spatial_correlation_sampler import spatial_correlation_sample

def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=9,
                                          stride=1)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return out_corr

def conv_feat_block(nIn, nOut):
    return nn.Sequential(
        nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(nOut, nOut, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2)
    )

def conv_dec_block(nIn):
    return nn.Sequential(
        nn.Conv2d(nIn, 128, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
    )


class Model(nn.Module):
    def __init__(self, nlevels):
        super(Model, self).__init__()

        self.nlevels = nlevels
        idx = [list(range(n, -1, -9)) for n in range(80,71,-1)]
        idx = list(np.array(idx).flatten())
        self.idx_fwd = Variable(torch.LongTensor(np.array(idx)).cuda(), requires_grad=False)
        self.idx_bwd = Variable(torch.LongTensor(np.array(list(reversed(idx)))).cuda(), requires_grad=False)
        self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear')
        self.softmax2d = nn.Softmax2d()

        self.conv1a = conv_feat_block(3,16)
        self.conv1b = conv_feat_block(3,16)
        self.conv1c = conv_feat_block(3,16)

        self.conv2a = conv_feat_block(16,32)
        self.conv2b = conv_feat_block(16,32)
        self.conv2c = conv_feat_block(16,32)

        self.conv3a = conv_feat_block(32,64)
        self.conv3b = conv_feat_block(32,64)
        self.conv3c = conv_feat_block(32,64)

        self.conv4a = conv_feat_block(64,96)
        self.conv4b = conv_feat_block(64,96)
        self.conv4c = conv_feat_block(64,96)

        self.conv5a = conv_feat_block(96,128)
        self.conv5b = conv_feat_block(96,128)
        self.conv5c = conv_feat_block(96,128)

        self.conv6a = conv_feat_block(128,192)
        self.conv6b = conv_feat_block(128,192)
        self.conv6c = conv_feat_block(128,192)

        self.corr = correlate # Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)

        self.decoder_fwd6 = conv_dec_block(162)
        self.decoder_bwd6 = conv_dec_block(162)
        self.decoder_fwd5 = conv_dec_block(292)
        self.decoder_bwd5 = conv_dec_block(292)
        self.decoder_fwd4 = conv_dec_block(260)
        self.decoder_bwd4 = conv_dec_block(260)
        self.decoder_fwd3 = conv_dec_block(228)
        self.decoder_bwd3 = conv_dec_block(228)
        self.decoder_fwd2 = conv_dec_block(196)
        self.decoder_bwd2 = conv_dec_block(196)

        self.decoder_occ6 = conv_dec_block(354)
        self.decoder_occ5 = conv_dec_block(292)
        self.decoder_occ4 = conv_dec_block(260)
        self.decoder_occ3 = conv_dec_block(228)
        self.decoder_occ2 = conv_dec_block(196)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def normalize(self, ims):
        imt = []
        for im in ims:
            im = im * 0.5
            im = im + 0.5
            im[:,0,:,:] = im[:,0,:,:] - 0.485  # Red
            im[:,1,:,:] = im[:,1,:,:] - 0.456 # Green
            im[:,2,:,:] = im[:,2,:,:] - 0.406 # Blue

            im[:,0,:,:] = im[:,0,:,:] / 0.229  # Red
            im[:,1,:,:] = im[:,1,:,:] / 0.224 # Green
            im[:,2,:,:] = im[:,2,:,:] / 0.225 # Blue

            imt.append(im)
        return imt

    def forward(self, im_tar, im_refs):
        '''
            inputS:
                im_tar: Middle Frame, I_0
                im_refs: Adjecent Frames in the order, [I-, I+]

            outputs:
                At self.nlevels different scales:
                flow_fwd: optical flow from I_0 to I+
                flow_bwd: optical flow from I_0 to I+
                occ : occlusions
        '''
        # im = Variable(torch.zeros(1,9,512,512).cuda())
        # ima = im[:, :3, :, :] + 0.2     # I_0
        # imb = im[:, 3:6, :, :] + 0.3    # I_+
        # imc = im[:, 6:, :, :] + 0.1     # I_-
        im_norm = self.normalize([im_tar] + im_refs)

        feat1a = self.conv1a(im_norm[0])
        feat2a = self.conv2a(feat1a)
        feat3a = self.conv3a(feat2a)
        feat4a = self.conv4a(feat3a)
        feat5a = self.conv5a(feat4a)
        feat6a = self.conv6a(feat5a)

        feat1b = self.conv1b(im_norm[2])
        feat2b = self.conv2b(feat1b)
        feat3b = self.conv3b(feat2b)
        feat4b = self.conv4b(feat3b)
        feat5b = self.conv5b(feat4b)
        feat6b = self.conv6b(feat5b)

        feat1c = self.conv1c(im_norm[1])
        feat2c = self.conv2c(feat1c)
        feat3c = self.conv3c(feat2c)
        feat4c = self.conv4c(feat3c)
        feat5c = self.conv5c(feat4c)
        feat6c = self.conv6c(feat5c)

        corr6_fwd = self.corr(feat6a, feat6b)
        corr6_fwd = corr6_fwd.index_select(1,self.idx_fwd)
        corr6_bwd = self.corr(feat6a, feat6c)
        corr6_bwd = corr6_bwd.index_select(1,self.idx_bwd)
        corr6 = torch.cat((corr6_fwd, corr6_bwd), 1)

        flow6_fwd = self.decoder_fwd6(corr6)
        flow6_fwd_up = self.upsample(flow6_fwd)
        flow6_bwd = self.decoder_bwd6(corr6)
        flow6_bwd_up = self.upsample(flow6_bwd)
        feat5b_warped = self.warp(feat5b, 0.625*flow6_fwd_up)
        feat5c_warped = self.warp(feat5c, -0.625*flow6_fwd_up)

        occ6_feat = torch.cat((corr6, feat6a), 1)
        occ6 = self.softmax2d(self.decoder_occ6(occ6_feat))

        corr5_fwd = self.corr(feat5a, feat5b_warped)
        corr5_fwd = corr5_fwd.index_select(1,self.idx_fwd)
        corr5_bwd = self.corr(feat5a, feat5c_warped)
        corr5_bwd = corr5_bwd.index_select(1,self.idx_bwd)
        corr5 = torch.cat((corr5_fwd, corr5_bwd), 1)

        upfeat5_fwd = torch.cat((corr5, feat5a, flow6_fwd_up), 1)
        flow5_fwd = self.decoder_fwd5(upfeat5_fwd)
        flow5_fwd_up = self.upsample(flow5_fwd)
        upfeat5_bwd = torch.cat((corr5, feat5a, flow6_bwd_up),1)
        flow5_bwd = self.decoder_bwd5(upfeat5_bwd)
        flow5_bwd_up = self.upsample(flow5_bwd)
        feat4b_warped = self.warp(feat4b, 1.25*flow5_fwd_up)
        feat4c_warped = self.warp(feat4c, -1.25*flow5_fwd_up)

        occ5 = self.softmax2d(self.decoder_occ5(upfeat5_fwd))

        corr4_fwd = self.corr(feat4a, feat4b_warped)
        corr4_fwd = corr4_fwd.index_select(1,self.idx_fwd)
        corr4_bwd = self.corr(feat4a, feat4c_warped)
        corr4_bwd = corr4_bwd.index_select(1,self.idx_bwd)
        corr4 = torch.cat((corr4_fwd, corr4_bwd), 1)

        upfeat4_fwd = torch.cat((corr4, feat4a, flow5_fwd_up), 1)
        flow4_fwd = self.decoder_fwd4(upfeat4_fwd)
        flow4_fwd_up = self.upsample(flow4_fwd)
        upfeat4_bwd = torch.cat((corr4, feat4a, flow5_bwd_up),1)
        flow4_bwd = self.decoder_bwd4(upfeat4_bwd)
        flow4_bwd_up = self.upsample(flow4_bwd)
        feat3b_warped = self.warp(feat3b, 2.5*flow4_fwd_up)
        feat3c_warped = self.warp(feat3c, -2.5*flow4_fwd_up)

        occ4 = self.softmax2d(self.decoder_occ4(upfeat4_fwd))

        corr3_fwd = self.corr(feat3a, feat3b_warped)
        corr3_fwd = corr3_fwd.index_select(1,self.idx_fwd)
        corr3_bwd = self.corr(feat3a, feat3c_warped)
        corr3_bwd = corr3_bwd.index_select(1,self.idx_bwd)
        corr3 = torch.cat((corr3_fwd, corr3_bwd), 1)

        upfeat3_fwd = torch.cat((corr3, feat3a, flow4_fwd_up), 1)
        flow3_fwd = self.decoder_fwd3(upfeat3_fwd)
        flow3_fwd_up = self.upsample(flow3_fwd)
        upfeat3_bwd = torch.cat((corr3, feat3a, flow4_bwd_up),1)
        flow3_bwd = self.decoder_bwd3(upfeat3_bwd)
        flow3_bwd_up = self.upsample(flow3_bwd)
        feat2b_warped = self.warp(feat2b, 5.0*flow3_fwd_up)
        feat2c_warped = self.warp(feat2c, -5.0*flow3_fwd_up)

        occ3 = self.softmax2d(self.decoder_occ3(upfeat3_fwd))

        corr2_fwd = self.corr(feat2a, feat2b_warped)
        corr2_fwd = corr2_fwd.index_select(1,self.idx_fwd)
        corr2_bwd = self.corr(feat2a, feat2c_warped)
        corr2_bwd = corr2_bwd.index_select(1,self.idx_bwd)
        corr2 = torch.cat((corr2_fwd, corr2_bwd), 1)

        upfeat2_fwd = torch.cat((corr2, feat2a, flow3_fwd_up), 1)
        flow2_fwd = self.decoder_fwd2(upfeat2_fwd)
        flow2_fwd_up = self.upsample(flow2_fwd)
        upfeat2_bwd = torch.cat((corr2, feat2a, flow3_bwd_up),1)
        flow2_bwd = self.decoder_bwd2(upfeat2_bwd)
        flow2_bwd_up = self.upsample(flow2_bwd)

        occ2 = self.softmax2d(self.decoder_occ2(upfeat2_fwd))

        flow2_fwd_fullres = 20*self.upsample(flow2_fwd_up)
        flow3_fwd_fullres = 10*self.upsample(flow3_fwd_up)
        flow4_fwd_fullres = 5*self.upsample(flow4_fwd_up)
        flow5_fwd_fullres = 2.5*self.upsample(flow5_fwd_up)
        flow6_fwd_fullres = 1.25*self.upsample(flow6_fwd_up)

        flow2_bwd_fullres = -20*self.upsample(flow2_bwd_up)
        flow3_bwd_fullres = -10*self.upsample(flow3_bwd_up)
        flow4_bwd_fullres = -5*self.upsample(flow4_bwd_up)
        flow5_bwd_fullres = -2.5*self.upsample(flow5_bwd_up)
        flow6_bwd_fullres = -1.25*self.upsample(flow6_bwd_up)

        occ2_fullres = F.upsample(occ2, scale_factor=4)
        occ3_fullres = F.upsample(occ3, scale_factor=4)
        occ4_fullres = F.upsample(occ4, scale_factor=4)
        occ5_fullres = F.upsample(occ5, scale_factor=4)
        occ6_fullres = F.upsample(occ6, scale_factor=4)

        if self.training:
            flow_fwd = [flow2_fwd_fullres, flow3_fwd_fullres, flow4_fwd_fullres, flow5_fwd_fullres, flow6_fwd_fullres]
            flow_bwd = [flow2_bwd_fullres, flow3_bwd_fullres, flow4_bwd_fullres, flow5_bwd_fullres, flow6_bwd_fullres]
            occ = [occ2_fullres, occ3_fullres, occ4_fullres, occ5_fullres, occ6_fullres]

            if self.nlevels==6:
                flow_fwd.append(0.625*flow6_fwd_up)
                flow_bwd.append(-0.625*flow6_bwd_up)
                occ.append(F.upsample(occ6, scale_factor=2))

            return flow_fwd, flow_bwd, occ
        else:
            return flow2_fwd_fullres, flow2_bwd_fullres, occ2_fullres

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border')
        mask = torch.autograd.Variable(torch.ones(x.size()), requires_grad=False).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask.data<0.9999] = 0
        mask[mask.data>0] = 1

        return output#*mask
