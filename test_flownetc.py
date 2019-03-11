# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import custom_transforms
from datasets.validation_flow import ValidationFlowFlowNetC
import torch
from torch.autograd import Variable
import models
from logger import AverageMeter
from torchvision.transforms import ToPILImage
from tensorboardX import SummaryWriter
import os
from flowutils.flowlib import flow_to_image
from utils import tensor2array


parser = argparse.ArgumentParser(description='Test FlowNetC',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--flownet', dest='flownet', type=str, default='FlowNetC5', choices=['FlowNetS', 'FlowNetS5', 'FlowNetS6', 'SpyNet', 'FlowNetC5'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 4|5|6')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', choices=['kitti2015', 'kitti2012'],
                    help='path to pre-trained Flow net model')


def compute_epe(gt, pred, op='sub'):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = torch.nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)
    if op=='sub':
        epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    if op=='div':
        epe = ((u_gt / u_pred) + (v_gt / v_pred))

    return epe

def main():
    global args
    args = parser.parse_args()
    save_path = 'checkpoints/test_flownetc'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    summary_writer = SummaryWriter(save_path)
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[1.0, 1.0, 1.0])
    flow_loader_h, flow_loader_w = 384, 1280
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])
    if args.dataset == "kitti2015":
        val_flow_set = ValidationFlowFlowNetC(root='/is/ps2/aranjan/AllFlowData/kitti/kitti2015',
                                sequence_length=5, transform=valid_flow_transform)
    elif args.dataset == "kitti2012":
        val_flow_set = ValidationFlowKitti2012(root='/is/ps2/aranjan/AllFlowData/kitti/kitti2012',
                                sequence_length=5, transform=valid_flow_transform)

    val_flow_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    flow_net = getattr(models, args.flownet)(pretrained=True).cuda()

    flow_net.eval()
    error_names = ['epe']
    errors = AverageMeter(i=len(error_names))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, flownet_c_flow, obj_map) in enumerate(val_flow_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        if args.dataset=="kitti2015":
            ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
            ref_img_var = ref_imgs_var[2]
        elif args.dataset=="kitti2012":
            ref_img_var = Variable(ref_imgs.cuda(), volatile=True)

        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)
        flownet_c_flow = Variable(flownet_c_flow.cuda(), volatile=True)

        # compute output
        flow_fwd = flow_net(tgt_img_var, ref_img_var)
        epe = compute_epe(gt=flownet_c_flow, pred=flow_fwd)
        scale_factor = compute_epe(gt=flownet_c_flow, pred=flow_fwd, op='div')
        #import ipdb
        #ipdb.set_trace()
        summary_writer.add_image('Frame 1', tensor2array(tgt_img_var.data[0].cpu()) , i)
        summary_writer.add_image('Frame 2', tensor2array(ref_img_var.data[0].cpu()) , i)
        summary_writer.add_image('Flow Output', flow_to_image(tensor2array(flow_fwd.data[0].cpu())) , i)
        summary_writer.add_image('UnFlow Output', flow_to_image(tensor2array(flownet_c_flow.data[0][:2].cpu())) , i)
        summary_writer.add_image('gtFlow Output', flow_to_image(tensor2array(flow_gt_var.data[0][:2].cpu())) , i)
        summary_writer.add_image('EPE Image w UnFlow', tensor2array(epe.data.cpu()) , i)
        summary_writer.add_scalar('EPE mean w UnFlow', epe.mean().data.cpu(), i)
        summary_writer.add_scalar('EPE max w UnFlow', epe.max().data.cpu(), i)
        summary_writer.add_scalar('Scale Factor max w UnFlow', scale_factor.max().data.cpu(), i)
        summary_writer.add_scalar('Scale Factor mean w UnFlow', scale_factor.mean().data.cpu(), i)
        summary_writer.add_scalar('Flow value max', flow_fwd.max().data.cpu(), i)
        print(i, "EPE: ", epe.mean().item())

        #print(i, epe)
        #errors.update(epe)

    print('Done')
    #print("Averge EPE",errors.avg )



if __name__ == '__main__':
    main()
