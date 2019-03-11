# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp, flow_warp, pose2flow
from ssim import ssim
epsilon = 1e-8

def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp

def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x

def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x

def photometric_flow_loss(tgt_img, ref_imgs, flows, explainability_mask, lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(explainability_mask, occ_masks, flows):
        assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]

            ref_img_warped = flow_warp(ref_img, current_flow)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)

            if occ_masks is not None:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            #weight /= 2.83
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss

    if type(flows[0]) not in [tuple, list]:
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        # occ_mask_at_scale = None
        loss += one_scale(explainability_mask[i], occ_mask_at_scale, flow_at_scale)

    return loss


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(depth, explainability_mask, occ_masks):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            assert((oob_normalization_const == oob_normalization_const).item() == 1)

            if explainability_mask is not None:
                diff = diff * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(ssim_loss)
            else:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            assert((reconstruction_loss == reconstruction_loss).item() == 1)
            #weight /= 2.83
        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
        loss += one_scale(d, mask, occ_masks)
    return loss



def depth_occlusion_masks(depth, pose, intrinsics, intrinsics_inv):
    flow_cam = [pose2flow(depth.squeeze(), pose[:,i], intrinsics, intrinsics_inv) for i in range(pose.size(1))]
    masks1, masks2 = occlusion_masks(flow_cam[1], flow_cam[2])
    masks0, masks3 = occlusion_masks(flow_cam[0], flow_cam[3])
    masks = torch.stack((masks0, masks1, masks2, masks3), dim=1)
    return masks

def gaussian_explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        loss += torch.exp(-torch.mean((mask_scaled-0.5).pow(2))/0.15)
    return loss


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)

def consensus_exp_masks(cam_flows_fwd, cam_flows_bwd, flows_fwd, flows_bwd, tgt_img, ref_img_fwd, ref_img_bwd, wssim, wrig, ws=0.1):
    def one_scale(cam_flow_fwd, cam_flow_bwd, flow_fwd, flow_bwd, tgt_img, ref_img_fwd, ref_img_bwd, ws):
        b, _, h, w = cam_flow_fwd.size()
        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_img_fwd, (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_img_bwd, (h, w))

        cam_warped_im_fwd = flow_warp(ref_img_scaled_fwd, cam_flow_fwd)
        cam_warped_im_bwd = flow_warp(ref_img_scaled_bwd, cam_flow_bwd)

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_cam_fwd = 1 - (cam_warped_im_fwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_fwd)
        valid_pixels_cam_bwd = 1 - (cam_warped_im_bwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_bwd)
        valid_pixels_cam = logical_or(valid_pixels_cam_fwd, valid_pixels_cam_bwd)  # if one of them is valid, then valid

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        valid_pixels_flow = logical_or(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        cam_err_fwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_fwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_fwd)).mean(1, keepdim=True))
        cam_err_bwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_bwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_bwd)).mean(1, keepdim=True))
        cam_err = torch.min(cam_err_fwd, cam_err_bwd) * valid_pixels_cam

        flow_err = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_fwd).mean(1, keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_fwd)).mean(1, keepdim=True)
        # flow_err_bwd = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_bwd).mean(1, keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_bwd)).mean(1, keepdim=True)
        # flow_err = torch.min(flow_err_fwd, flow_err_bwd)

        exp_target = (wrig*cam_err <= (flow_err+epsilon)).type_as(cam_err)

        return exp_target

    exp_masks_target = []
    for i in range(len(cam_flows_fwd)):
        exp_masks_target.append(one_scale(cam_flows_fwd[i], cam_flows_bwd[i], flows_fwd[i], flows_bwd[i], tgt_img, ref_img_fwd, ref_img_bwd, ws))
        ws = ws / 2.3

    return exp_masks_target

def compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd, THRESH):
    joint_masks = []
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        rigidity_mask_fwd_one_scale = (rigidity_mask_fwd[i] > THRESH).type_as(exp_mask_one_scale)
        rigidity_mask_bwd_one_scale = (rigidity_mask_bwd[i] > THRESH).type_as(exp_mask_one_scale)
        exp_mask_one_scale_joint = 1 - (1-exp_mask_one_scale[:,1])*(1-exp_mask_one_scale[:,2]).unsqueeze(1) > 0.5
        joint_mask_one_scale_fwd = logical_or(rigidity_mask_fwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_bwd = logical_or(rigidity_mask_bwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_fwd = Variable(joint_mask_one_scale_fwd.data, requires_grad=False)
        joint_mask_one_scale_bwd = Variable(joint_mask_one_scale_bwd.data, requires_grad=False)
        joint_mask_one_scale = torch.cat((joint_mask_one_scale_bwd, joint_mask_one_scale_bwd,
                        joint_mask_one_scale_fwd, joint_mask_one_scale_fwd), dim=1)
        joint_masks.append(joint_mask_one_scale)

    return joint_masks

def consensus_depth_flow_mask(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce):
    # Loop over each scale
    assert(len(explainability_mask)==len(census_mask_bwd))
    assert(len(explainability_mask)==len(census_mask_fwd))
    loss = 0.
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        census_mask_fwd_one_scale = (census_mask_fwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)
        census_mask_bwd_one_scale = (census_mask_bwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)

        #Using the pixelwise consensus term
        exp_fwd_target_one_scale = exp_masks_fwd_target[i]
        exp_bwd_target_one_scale = exp_masks_bwd_target[i]
        census_mask_fwd_one_scale = logical_or(census_mask_fwd_one_scale, exp_fwd_target_one_scale)
        census_mask_bwd_one_scale = logical_or(census_mask_bwd_one_scale, exp_bwd_target_one_scale)

        # OR gate for constraining only rigid pixels
        # exp_mask_fwd_one_scale = (exp_mask_one_scale[:,2].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # exp_mask_bwd_one_scale = (exp_mask_one_scale[:,1].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # census_mask_fwd_one_scale = 1- (1-census_mask_fwd_one_scale)*(1-exp_mask_fwd_one_scale)
        # census_mask_bwd_one_scale = 1- (1-census_mask_bwd_one_scale)*(1-exp_mask_bwd_one_scale)

        census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
        census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)

        rigidity_mask_combined = torch.cat((census_mask_bwd_one_scale, census_mask_bwd_one_scale,
                        census_mask_fwd_one_scale, census_mask_fwd_one_scale), dim=1)
        loss += weighted_binary_cross_entropy(exp_mask_one_scale, rigidity_mask_combined.type_as(exp_mask_one_scale), [wbce, 1-wbce])

    return loss

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output + epsilon)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))

def edge_aware_smoothness_per_pixel(img, pred):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y
    import ipdb; ipdb.set_trace()
    return smoothness_x + smoothness_y


def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)

    return loss



def smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # 2sqrt(2)
    return loss

def occlusion_masks(flow_bw, flow_fw):
    mag_sq = flow_fw.pow(2).sum(dim=1) + flow_bw.pow(2).sum(dim=1)
    #flow_bw_warped = flow_warp(flow_bw, flow_fw)
    #flow_fw_warped = flow_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw
    flow_diff_bw = flow_bw + flow_fw
    occ_thresh =  0.08 * mag_sq + 1.0
    occ_fw = flow_diff_fw.sum(dim=1) > occ_thresh
    occ_bw = flow_diff_bw.sum(dim=1) > occ_thresh
    return occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)
#    return torch.stack((occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)), dim=1)

def flow_diff(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    diff = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    return diff


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.upsample(pred, size=(h_gt, w_gt), mode='bilinear')
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()

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
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon);
    if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()

def compute_all_epes(gt, rigid_pred, non_rigid_pred, rigidity_mask, THRESH=0.5):
    _, _, h_pred, w_pred = rigid_pred.size()
    _, _, h_gt, w_gt = gt.size()
    rigidity_pred_mask = nn.functional.upsample(rigidity_mask, size=(h_pred, w_pred), mode='bilinear')
    rigidity_gt_mask = nn.functional.upsample(rigidity_mask, size=(h_gt, w_gt), mode='bilinear')

    non_rigid_pred = (rigidity_pred_mask<=THRESH).type_as(non_rigid_pred).expand_as(non_rigid_pred) * non_rigid_pred
    rigid_pred = (rigidity_pred_mask>THRESH).type_as(rigid_pred).expand_as(rigid_pred) * rigid_pred
    total_pred = non_rigid_pred + rigid_pred

    gt_non_rigid = (rigidity_gt_mask<=THRESH).type_as(gt).expand_as(gt) * gt
    gt_rigid = (rigidity_gt_mask>THRESH).type_as(gt).expand_as(gt) * gt

    all_epe = compute_epe(gt, total_pred)
    rigid_epe = compute_epe(gt_rigid, rigid_pred)
    non_rigid_epe = compute_epe(gt_non_rigid, non_rigid_pred)
    outliers = outlier_err(gt, total_pred)

    return [all_epe, rigid_epe, non_rigid_epe, outliers]


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
