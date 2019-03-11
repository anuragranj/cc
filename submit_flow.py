import argparse
import os
from tqdm import tqdm
import numpy as np
from path import Path
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn as nn

import custom_transforms
from inverse_warp import pose2flow
from datasets.validation_flow import KITTI2015Test
import models
from logger import AverageMeter
from PIL import Image
from torchvision.transforms import ToPILImage
from flowutils.flowlib import flow_to_image
from utils import tensor2array
from loss_functions import compute_all_epes
from flowutils import flow_io


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/ps/project/datasets/AllFlowData/kitti/kitti2015',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispResNet6', 'DispNetS5', 'DispNetS6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6', 'PoseExpNet5', 'PoseExpNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6', 'PoseExpNet5', 'PoseExpNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['PWCNet','FlowNetS', 'Back2Future', 'FlowNetC5','FlowNetC6', 'SpyNet'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')

parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')
parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.01, help='THRESH')
parser.add_argument('--mu', dest='mu', type=float, default=1.0, help='mu')
parser.add_argument('--pretrained-path', dest='pretrained_path', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6, help='number of levels in multiscale. Options: 4|5')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', help='path to pre-trained Flow net model')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')


def main():
    global args
    args = parser.parse_args()
    args.pretrained_path = Path(args.pretrained_path)

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
        args.output_dir.makedirs_p()

        image_dir = args.output_dir/'images'
        mask_dir = args.output_dir/'mask'
        viz_dir = args.output_dir/'viz'
        testing_dir = args.output_dir/'testing'
        testing_dir_flo = args.output_dir/'testing_flo'

        image_dir.makedirs_p()
        mask_dir.makedirs_p()
        viz_dir.makedirs_p()
        testing_dir.makedirs_p()
        testing_dir_flo.makedirs_p()

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    flow_loader_h, flow_loader_w = 256, 832
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])

    val_flow_set = KITTI2015Test(root=args.kitti_dir,
                            sequence_length=5, transform=valid_flow_transform)

    if args.DEBUG:
        print("DEBUG MODE: Using Training Set")
        val_flow_set = KITTI2015Test(root=args.kitti_dir,
                        sequence_length=5, transform=valid_flow_transform, phase='training')

    val_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    disp_net = getattr(models, args.dispnet)().cuda()
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=4).cuda()
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=4).cuda()
    flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()

    dispnet_weights = torch.load(args.pretrained_path/'dispnet_model_best.pth.tar')
    posenet_weights = torch.load(args.pretrained_path/'posenet_model_best.pth.tar')
    masknet_weights = torch.load(args.pretrained_path/'masknet_model_best.pth.tar')
    flownet_weights = torch.load(args.pretrained_path/'flownet_model_best.pth.tar')
    disp_net.load_state_dict(dispnet_weights['state_dict'])
    pose_net.load_state_dict(posenet_weights['state_dict'])
    flow_net.load_state_dict(flownet_weights['state_dict'])
    mask_net.load_state_dict(masknet_weights['state_dict'])

    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_img_original) in enumerate(tqdm(val_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        disp = disp_net(tgt_img_var)
        depth = 1/disp
        pose = pose_net(tgt_img_var, ref_imgs_var)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)
        if args.flownet=='Back2Future':
            flow_fwd, _, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        else:
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])
        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5

        rigidity_mask_census_soft = (flow_cam - flow_fwd).abs()#.normalize()
        rigidity_mask_census_u = rigidity_mask_census_soft[:,0] < args.THRESH
        rigidity_mask_census_v = rigidity_mask_census_soft[:,1] < args.THRESH
        rigidity_mask_census = (rigidity_mask_census_u).type_as(flow_fwd) * (rigidity_mask_census_v).type_as(flow_fwd)
        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        _, _, h_pred, w_pred = flow_cam.size()
        _, _, h_gt, w_gt = tgt_img_original.size()
        rigidity_pred_mask = nn.functional.upsample(rigidity_mask_combined, size=(h_pred, w_pred), mode='bilinear')

        non_rigid_pred = (rigidity_pred_mask<=args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        rigid_pred = (rigidity_pred_mask>args.THRESH).type_as(flow_cam).expand_as(flow_cam) * flow_cam
        total_pred = non_rigid_pred + rigid_pred

        pred_fullres = nn.functional.upsample(total_pred, size=(h_gt, w_gt), mode='bilinear')
        pred_fullres[:,0,:,:] = pred_fullres[:,0,:,:] * (w_gt/w_pred)
        pred_fullres[:,1,:,:] = pred_fullres[:,1,:,:] * (h_gt/h_pred)

        flow_fwd_fullres = nn.functional.upsample(flow_fwd, size=(h_gt, w_gt), mode='bilinear')
        flow_fwd_fullres[:,0,:,:] = flow_fwd_fullres[:,0,:,:] * (w_gt/w_pred)
        flow_fwd_fullres[:,1,:,:] = flow_fwd_fullres[:,1,:,:] * (h_gt/h_pred)

        flow_cam_fullres = nn.functional.upsample(flow_cam, size=(h_gt, w_gt), mode='bilinear')
        flow_cam_fullres[:,0,:,:] = flow_cam_fullres[:,0,:,:] * (w_gt/w_pred)
        flow_cam_fullres[:,1,:,:] = flow_cam_fullres[:,1,:,:] * (h_gt/h_pred)

        tgt_img_np = tgt_img[0].numpy()
        rigidity_mask_combined_np = rigidity_mask_combined.cpu().data[0].numpy()

        if args.output_dir is not None:
            np.save(image_dir/str(i).zfill(3), tgt_img_np )
            np.save(mask_dir/str(i).zfill(3), rigidity_mask_combined_np)
            pred_u = pred_fullres[0][0].data.cpu().numpy()
            pred_v = pred_fullres[0][1].data.cpu().numpy()
            flow_io.flow_write_png(testing_dir/str(i).zfill(6)+'_10.png' ,u=pred_u, v=pred_v)
            flow_io.flow_write(testing_dir_flo/str(i).zfill(6)+'_10.flo' ,pred_u, pred_v)



        if (args.output_dir is not None):
            ind = int(i)
            tgt_img_viz = tensor2array(tgt_img[0].cpu())
            depth_viz = tensor2array(disp.data[0].cpu(), max_value=None, colormap='magma')
            mask_viz = tensor2array(rigidity_mask_combined.data[0].cpu(), max_value=1, colormap='magma')
            row2_viz = flow_to_image(np.hstack((tensor2array(flow_cam_fullres.data[0].cpu()),
                                   tensor2array(flow_fwd_fullres.data[0].cpu()),
                                   tensor2array(pred_fullres.data[0].cpu()) )) )

            row1_viz = np.hstack((tgt_img_viz, depth_viz, mask_viz))

            row1_viz_im = Image.fromarray((255*row1_viz.transpose(1,2,0)).astype('uint8'))
            row2_viz_im = Image.fromarray((255*row2_viz.transpose(1,2,0)).astype('uint8'))

            row1_viz_im.save(viz_dir/str(i).zfill(3)+'01.png')
            row2_viz_im.save(viz_dir/str(i).zfill(3)+'02.png')

    print("Done!")
    # print("\t {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} ".format(*error_names))
    # print("Errors \t {:10.4f}, {:10.4f} {:10.4f}, {:10.4f} {:10.4f}, {:10.4f}".format(*errors.avg))


if __name__ == '__main__':
    main()
