# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import glob
import torch
import cv2
from torch.autograd import Variable
from PIL import Image
from scipy import interpolate, io
from scipy.misc import imresize, imread
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from utils import tensor2array
import models
from loss_functions import spatial_normalize

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dispnet", dest='dispnet', type=str, default='DispResNet6', help='dispnet architecture')
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=256, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=70, type=float)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

class test_framework(object):
    def __init__(self, root, min_depth=1e-3, max_depth=70):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.img_files = sorted(glob.glob(root/'Test134/*.jpg'))
        self.depth_files = sorted(glob.glob(root/'Gridlaserdata/*.mat'))

        # This test file is corrupted in the original dataset
        self.img_files.pop(61)
        self.depth_files.pop(61)

        self.ratio = 2
        self.h_ratio = 1 / (1.33333 * self.ratio)
        self.color_new_height = 1704 // 2
        self.depth_new_height = 21

    def __getitem__(self, i):
        img = Image.open(self.img_files[i])
        try:
            imgarr = np.array(img)
            tgt_img = imgarr.astype(np.float32)
        except:
            imgarr = np.array(img)
            tgt_img = imgarr.astype(np.float32)

        tgt_img = tgt_img[ (2272 - self.color_new_height)//2:(2272 + self.color_new_height)//2,:]

        depth_map = io.loadmat(self.depth_files[i])
        depth_gt = depth_map["Position3DGrid"][:,:,3]
        depth_gt_cropped = depth_gt[(55 - 21)//2:(55 + 21)//2]
        return {'tgt': tgt_img,
                'path':self.img_files[i],
                'gt_depth': depth_gt_cropped,
                'mask': np.logical_and(depth_gt_cropped > self.min_depth, depth_gt_cropped < self.max_depth)
                }

    def __len__(self):
        return len(self.img_files)

def main():
    args = parser.parse_args()

    disp_net = getattr(models, args.dispnet)().cuda()
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
        (but consistent with original paper)')
    seq_length = 0

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.min_depth, args.max_depth)
    errors = np.zeros((2, 7, len(framework)), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        viz_dir = output_dir/'viz'
        output_dir.makedirs_p()
        viz_dir.makedirs_p()

    for j, sample in enumerate(tqdm(framework)):
        tgt_img = sample['tgt']

        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            tgt_img = imresize(tgt_img, (args.img_height, args.img_width)).astype(np.float32)

        tgt_img = np.transpose(tgt_img, (2, 0, 1))
        tgt_img = torch.from_numpy(tgt_img).unsqueeze(0)
        tgt_img = ((tgt_img/255 - 0.5)/0.5).cuda()
        tgt_img_var = Variable(tgt_img, volatile=True)

        pred_disp = disp_net(tgt_img_var)
        pred_disp = pred_disp.data.cpu().numpy()[0,0]
        gt_depth = sample['gt_depth']

        if args.output_dir is not None:
            if j == 0:
                predictions = np.zeros((len(framework), *pred_disp.shape))
            predictions[j] = 1/pred_disp
            gt_viz = interp_gt_disp(gt_depth)
            gt_viz = torch.FloatTensor(gt_viz)
            gt_viz[gt_viz == 0] = 1000
            gt_viz = (1/gt_viz).clamp(0,10)

            tgt_img_viz = tensor2array(tgt_img[0].cpu())
            depth_viz = tensor2array(torch.FloatTensor(pred_disp), max_value=None, colormap='hot')
            gt_viz = tensor2array(gt_viz, max_value=None, colormap='hot')
            tgt_img_viz_im = Image.fromarray((255*tgt_img_viz).astype('uint8'))
            tgt_img_viz_im = tgt_img_viz_im.resize(size=(args.img_width, args.img_height), resample=3)
            tgt_img_viz_im.save(viz_dir/str(j).zfill(4)+'img.png')
            depth_viz_im = Image.fromarray((255*depth_viz).astype('uint8'))
            depth_viz_im = depth_viz_im.resize(size=(args.img_width, args.img_height), resample=3)
            depth_viz_im.save(viz_dir/str(j).zfill(4)+'depth.png')
            gt_viz_im = Image.fromarray((255*gt_viz).astype('uint8'))
            gt_viz_im = gt_viz_im.resize(size=(args.img_width, args.img_height), resample=3)
            gt_viz_im.save(viz_dir/str(j).zfill(4)+'gt.png')

            all_viz_im = Image.fromarray( np.hstack([np.array(tgt_img_viz_im), np.array(gt_viz_im), np.array(depth_viz_im)]) )
            all_viz_im.save(viz_dir/str(j).zfill(4)+'all.png')


        pred_depth = 1/pred_disp
        pred_depth_zoomed = zoom(pred_depth, (gt_depth.shape[0]/pred_depth.shape[0],gt_depth.shape[1]/pred_depth.shape[1])).clip(args.min_depth, args.max_depth)
        if sample['mask'] is not None:
            pred_depth_zoomed = pred_depth_zoomed[sample['mask']]
            gt_depth = gt_depth[sample['mask']]

        scale_factor = np.median(gt_depth)/np.median(pred_depth_zoomed)
        pred_depth_zoomed = scale_factor*pred_depth_zoomed
        pred_depth_zoomed[pred_depth_zoomed>args.max_depth] = args.max_depth
        errors[1,:,j] = compute_errors(gt_depth, pred_depth_zoomed)

    mean_errors = errors.mean(2)
    error_names = ['abs_rel','sq_rel','rms','log_rms','a1','a2','a3']

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[1]))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions)

def interp_gt_disp(mat, mask_val=0):
    mat[mat==mask_val] = np.nan
    x = np.arange(0, mat.shape[1])
    y = np.arange(0, mat.shape[0])
    mat = np.ma.masked_invalid(mat)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~mat.mask]
    y1 = yy[~mat.mask]
    newarr = mat[~mat.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear', fill_value=mask_val)
    return GD1

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


if __name__ == '__main__':
    main()
