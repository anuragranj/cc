# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import os
from tqdm import tqdm
import numpy as np
from path import Path
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

import custom_transforms
from inverse_warp import pose2flow
from datasets.validation_flow import ValidationFlow, ValidationFlowKitti2012
import models
from logger import AverageMeter
from PIL import Image
from torchvision.transforms import ToPILImage
from flowutils.flowlib import flow_to_image
from utils import tensor2array
from loss_functions import compute_all_epes, consensus_exp_masks, logical_or


parser = argparse.ArgumentParser(description='Evaluate optical flow on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/ps/project/datasets/AllFlowData/kitti/kitti2015',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispResNet6', 'DispNetS6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future', 'FlowNetC6'],
                    help='flow network architecture.')

parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.01, help='THRESH')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained posenet model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH', help='path to pre-trained flownet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH', help='path to pre-trained masknet model')

parser.add_argument('--nlevels', dest='nlevels', type=int, default=6, help='number of levels in multiscale. Options: 4|5')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', help='path to pre-trained Flow net model')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')


def main():
    global args
    args = parser.parse_args()
    args.pretrained_disp = Path(args.pretrained_disp)
    args.pretrained_pose = Path(args.pretrained_pose)
    args.pretrained_mask = Path(args.pretrained_mask)
    args.pretrained_flow = Path(args.pretrained_flow)

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
        args.output_dir.makedirs_p()

        image_dir = args.output_dir/'images'
        gt_dir = args.output_dir/'gt'
        mask_dir = args.output_dir/'mask'
        viz_dir = args.output_dir/'viz'

        image_dir.makedirs_p()
        gt_dir.makedirs_p()
        mask_dir.makedirs_p()
        viz_dir.makedirs_p()

        output_writer = SummaryWriter(args.output_dir)

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    flow_loader_h, flow_loader_w = 256, 832
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])
    if args.dataset == "kitti2015":
        val_flow_set = ValidationFlow(root=args.kitti_dir,
                                sequence_length=5, transform=valid_flow_transform)

    val_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    disp_net = getattr(models, args.dispnet)().cuda()
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=4).cuda()
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=4).cuda()
    flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()

    dispnet_weights = torch.load(args.pretrained_disp)
    posenet_weights = torch.load(args.pretrained_pose)
    masknet_weights = torch.load(args.pretrained_mask)
    flownet_weights = torch.load(args.pretrained_flow)
    disp_net.load_state_dict(dispnet_weights['state_dict'])
    pose_net.load_state_dict(posenet_weights['state_dict'])
    flow_net.load_state_dict(flownet_weights['state_dict'])
    mask_net.load_state_dict(masknet_weights['state_dict'])

    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    error_names = ['epe_total', 'epe_sp', 'epe_mv', 'Fl', 'epe_total_gt_mask', 'epe_sp_gt_mask', 'epe_mv_gt_mask', 'Fl_gt_mask']
    errors = AverageMeter(i=len(error_names))
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt) in enumerate(tqdm(val_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)
        obj_map_gt_var = Variable(obj_map_gt.cuda(), volatile=True)

        disp = disp_net(tgt_img_var)
        depth = 1/disp
        pose = pose_net(tgt_img_var, ref_imgs_var)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)

        if args.flownet=='Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        else:
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])

        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)
        flow_cam_bwd = pose2flow(depth.squeeze(1), pose[:,1], intrinsics_var, intrinsics_inv_var)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5
        rigidity_mask_census_soft = (flow_cam - flow_fwd).abs()#.normalize()
        rigidity_mask_census_u = rigidity_mask_census_soft[:,0] < args.THRESH
        rigidity_mask_census_v = rigidity_mask_census_soft[:,1] < args.THRESH
        rigidity_mask_census = (rigidity_mask_census_u).type_as(flow_fwd) * (rigidity_mask_census_v).type_as(flow_fwd)

        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        flow_fwd_non_rigid = (rigidity_mask_combined<=args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        flow_fwd_rigid = (rigidity_mask_combined>args.THRESH).type_as(flow_cam).expand_as(flow_cam) * flow_cam
        total_flow = flow_fwd_rigid + flow_fwd_non_rigid

        rigidity_mask = rigidity_mask.type_as(flow_fwd)
        _epe_errors = compute_all_epes(flow_gt_var, flow_cam, flow_fwd, rigidity_mask_combined) + compute_all_epes(flow_gt_var, flow_cam, flow_fwd, (1-obj_map_gt_var_expanded) )
        errors.update(_epe_errors)

        tgt_img_np = tgt_img[0].numpy()
        rigidity_mask_combined_np = rigidity_mask_combined.cpu().data[0].numpy()
        gt_mask_np = obj_map_gt[0].numpy()


        if args.output_dir is not None:
            np.save(image_dir/str(i).zfill(3), tgt_img_np )
            np.save(gt_dir/str(i).zfill(3), gt_mask_np)
            np.save(mask_dir/str(i).zfill(3), rigidity_mask_combined_np)



        if (args.output_dir is not None) and i%10==0:
            ind = int(i//10)
            output_writer.add_image('val Dispnet Output Normalized', tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone'), ind)
            output_writer.add_image('val Input', tensor2array(tgt_img[0].cpu()), i)
            output_writer.add_image('val Total Flow Output', flow_to_image(tensor2array(total_flow.data[0].cpu())), ind)
            output_writer.add_image('val Rigid Flow Output', flow_to_image(tensor2array(flow_fwd_rigid.data[0].cpu())), ind)
            output_writer.add_image('val Non-rigid Flow Output', flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu())), ind)
            output_writer.add_image('val Rigidity Mask', tensor2array(rigidity_mask.data[0].cpu(), max_value=1, colormap='bone'), ind)
            output_writer.add_image('val Rigidity Mask Census', tensor2array(rigidity_mask_census.data[0].cpu(), max_value=1, colormap='bone'), ind)
            output_writer.add_image('val Rigidity Mask Combined', tensor2array(rigidity_mask_combined.data[0].cpu(), max_value=1, colormap='bone'), ind)

            tgt_img_viz = tensor2array(tgt_img[0].cpu())
            depth_viz = tensor2array(disp.data[0].cpu(), max_value=None, colormap='bone')
            mask_viz = tensor2array(rigidity_mask_census_soft.data[0].prod(dim=0).cpu(), max_value=1, colormap='bone')
            rigid_flow_viz = flow_to_image(tensor2array(flow_cam.data[0].cpu()))
            non_rigid_flow_viz = flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu()))
            total_flow_viz = flow_to_image(tensor2array(total_flow.data[0].cpu()))
            row1_viz = np.hstack((tgt_img_viz, depth_viz, mask_viz))
            row2_viz = np.hstack((rigid_flow_viz, non_rigid_flow_viz, total_flow_viz))

            row1_viz_im = Image.fromarray((255*row1_viz).astype('uint8'))
            row2_viz_im = Image.fromarray((row2_viz).astype('uint8'))

            row1_viz_im.save(viz_dir/str(i).zfill(3)+'01.png')
            row2_viz_im.save(viz_dir/str(i).zfill(3)+'02.png')

    print("Results")
    print("\t {:>10}, {:>10}, {:>10}, {:>6}, {:>10}, {:>10}, {:>10}, {:>10} ".format(*error_names))
    print("Errors \t {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors.avg))

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / F_mag) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    f_err = n_err.sum()/valid_gt.sum();
    if type(f_err) == Variable: f_err = f_err.data
    return f_err[0]


if __name__ == '__main__':
    main()
