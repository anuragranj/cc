# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
from loss_functions import compute_epe
import custom_transforms
from datasets.validation_flow import ValidationFlow, ValidationFlowKitti2012
import torch
from torch.autograd import Variable
import models
from logger import AverageMeter
from loss_functions import compute_all_epes
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['Back2Future'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=5,
                    help='number of levels in multiscale. Options: 4|5|6')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', choices=['kitti2015', 'kitti2012'],
                    help='path to pre-trained Flow net model')


def main():
    global args
    args = parser.parse_args()
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    flow_loader_h, flow_loader_w = 256, 832
    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])
    if args.dataset == "kitti2015":
        val_flow_set = ValidationFlow(root='/home/anuragr/datasets/kitti/kitti2015',
                                sequence_length=5, transform=valid_flow_transform)
    elif args.dataset == "kitti2012":
        val_flow_set = ValidationFlowKitti2012(root='/is/ps2/aranjan/AllFlowData/kitti/kitti2012',
                                sequence_length=5, transform=valid_flow_transform)

    val_flow_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True)

    flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()

    if args.pretrained_flow:
        print("=> using pre-trained weights from {}".format(args.pretrained_flow))
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights['state_dict'])#, strict=False)

    flow_net = flow_net.cuda()
    flow_net.eval()
    error_names = ['epe_total', 'epe_non_rigid', 'epe_rigid', 'outliers']
    errors = AverageMeter(i=len(error_names))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map) in enumerate(tqdm(val_flow_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        if args.dataset=="kitti2015":
            ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
            ref_img_var = ref_imgs_var[1:3]
        elif args.dataset=="kitti2012":
            ref_img_var = Variable(ref_imgs.cuda(), volatile=True)

        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)
        # compute output
        flow_fwd, flow_bwd, occ = flow_net(tgt_img_var, ref_img_var)
        #epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
        obj_map_gt_var = Variable(obj_map.cuda(), volatile=True)
        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        epe = compute_all_epes(flow_gt_var, flow_fwd, flow_fwd,  (1-obj_map_gt_var_expanded) )
        #print(i, epe)
        errors.update(epe)

    print("Averge EPE",errors.avg )



if __name__ == '__main__':
    main()
