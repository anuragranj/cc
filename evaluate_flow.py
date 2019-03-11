# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import os
from tqdm import tqdm
import numpy as np
from path import Path
from flowutils import flow_io
from logger import AverageMeter
epsilon = 1e-8
parser = argparse.ArgumentParser(description='Benchmark optical flow predictions',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')
parser.add_argument('--gt-dir', dest='gt_dir', type=str, default=None, help='path to gt directory')
parser.add_argument('-N', dest='N', type=int, default=200, help='number of samples')


def main():
    global args
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.gt_dir = Path(args.gt_dir)

    error_names = ['epe_total', 'outliers']
    errors = AverageMeter(i=len(error_names))

    for i in tqdm(range(args.N)):
        gt_flow_path = args.gt_dir.joinpath(str(i).zfill(6)+'_10.png')
        output_flow_path = args.output_dir.joinpath(str(i).zfill(6)+'_10.png')
        u_gt,v_gt,valid_gt = flow_io.flow_read_png(gt_flow_path)
        u_pred,v_pred,valid_pred = flow_io.flow_read_png(output_flow_path)

        _errors = compute_err(u_gt, v_gt, valid_gt, u_pred, v_pred, valid_pred)
        errors.update(_errors)


    print("Results")
    print("\t {:>10}, {:>10} ".format(*error_names))
    print("Errors \t {:10.4f}, {:10.4f}".format(*errors.avg))

def compute_err(u_gt, v_gt, valid_gt, u_pred, v_pred, valid_pred, tau=[3,0.05]):
    epe = np.sqrt(np.power((u_gt - u_pred), 2) + np.power((v_gt - v_pred), 2))
    epe = epe * valid_gt
    aepe = epe.sum() / valid_gt.sum()
    F_mag = np.sqrt(np.power(u_gt, 2)+ np.power(v_gt, 2))
    E_0 = (epe > tau[0])#.type_as(epe)
    E_1 = ((epe / (F_mag+epsilon) ) > tau[1])#.type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    f_err = n_err.sum()/valid_gt.sum()
    return [aepe, f_err]


if __name__ == '__main__':
    main()
