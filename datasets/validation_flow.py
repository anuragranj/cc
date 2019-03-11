# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from PIL import Image
from path import Path
from flowutils import flow_io
import torch
import os
from skimage import transform as sktransform

def crawl_folders(folders_list):
        imgs = []
        depth = []
        for folder in folders_list:
            current_imgs = sorted(folder.files('*.jpg'))
            current_depth = []
            for img in current_imgs:
                d = img.dirname()/(img.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                depth.append(d)
            imgs.extend(current_imgs)
            depth.extend(current_depth)
        return imgs, depth


def load_as_float(path):
    return imread(path).astype(np.float32)

def get_intrinsics(calib_file, cid='02'):
    #print(zoom_x, zoom_y)
    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
    return P_rect[:,:3]


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                    data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                    pass
    return data

class KITTI2015Test(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length, transform=None, N=200, phase='testing'):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.N = N
        self.transform = transform
        self.phase = phase
        seq_ids = list(range(-int(sequence_length/2), int(sequence_length/2)+1))
        seq_ids.remove(0)
        self.seq_ids = [x+10 for x in seq_ids]

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_paths = [self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_'+str(k).zfill(2)+'.png') for k in self.seq_ids]
        cam_calib_path = self.root.joinpath('data_scene_flow_calib', self.phase, 'calib_cam_to_cam', str(index).zfill(6)+'.txt')

        tgt_img_original = load_as_float(tgt_img_path)
        tgt_img = load_as_float(tgt_img_path)
        ref_imgs = [load_as_float(ref_img) for ref_img in ref_img_paths]
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')
        tgt_img_original = torch.FloatTensor(tgt_img_original.transpose(2,0,1))

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), tgt_img_original

    def __len__(self):
        return self.N

class ValidationFlow(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length, transform=None, N=200, phase='training', occ='flow_occ'):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.N = N
        self.transform = transform
        self.phase = phase
        seq_ids = list(range(-int(sequence_length/2), int(sequence_length/2)+1))
        seq_ids.remove(0)
        self.seq_ids = [x+10 for x in seq_ids]
        self.occ = occ

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_paths = [self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_'+str(k).zfill(2)+'.png') for k in self.seq_ids]
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, self.occ, str(index).zfill(6)+'_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib', self.phase, 'calib_cam_to_cam', str(index).zfill(6)+'.txt')
        obj_map_path = self.root.joinpath('data_scene_flow', self.phase, 'obj_map', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_imgs = [load_as_float(ref_img) for ref_img in ref_img_paths]
        if os.path.isfile(obj_map_path):
            obj_map = load_as_float(obj_map_path)
        else:
            obj_map = np.ones((tgt_img.shape[0], tgt_img.shape[1]))
        u,v,valid = flow_io.flow_read_png(gt_flow_path)
        gtFlow = np.dstack((u,v,valid))
        #gtFlow = scale_flow(np.dstack((u,v,valid)), h=self.flow_h, w=self.flow_w)
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), gtFlow, obj_map

    def __len__(self):
        return self.N

class ValidationMask(data.Dataset):
    """
        Kitti 2015 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length, transform=None, N=200, phase='training'):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.N = N
        self.transform = transform
        self.phase = phase
        seq_ids = list(range(-int(sequence_length/2), int(sequence_length/2)+1))
        seq_ids.remove(0)
        self.seq_ids = [x+10 for x in seq_ids]

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_10.png')
        ref_img_paths = [self.root.joinpath('data_scene_flow_multiview', self.phase, 'image_2',str(index).zfill(6)+'_'+str(k).zfill(2)+'.png') for k in self.seq_ids]
        gt_flow_path = self.root.joinpath('data_scene_flow', self.phase, 'flow_occ', str(index).zfill(6)+'_10.png')
        cam_calib_path = self.root.joinpath('data_scene_flow_calib', self.phase, 'calib_cam_to_cam', str(index).zfill(6)+'.txt')
        obj_map_path = self.root.joinpath('data_scene_flow', self.phase, 'obj_map', str(index).zfill(6)+'_10.png')
        semantic_map_path = self.root.joinpath('semantic_labels', self.phase, 'semantic', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_imgs = [load_as_float(ref_img) for ref_img in ref_img_paths]
        obj_map = torch.LongTensor(np.array(Image.open(obj_map_path)))
        semantic_map = torch.LongTensor(np.array(Image.open(semantic_map_path)))
        u,v,valid = flow_io.flow_read_png(gt_flow_path)
        gtFlow = np.dstack((u,v,valid))
        #gtFlow = scale_flow(np.dstack((u,v,valid)), h=self.flow_h, w=self.flow_w)
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))
        intrinsics = get_intrinsics(cam_calib_path).astype('float32')

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(intrinsics)
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), gtFlow, obj_map, semantic_map

    def __len__(self):
        return self.N

class ValidationFlowKitti2012(data.Dataset):
    """
        Kitti 2012 flow loader
        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, sequence_length=5, transform=None, N=194, flow_w=1024, flow_h=384, phase='training'):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.N = N
        self.transform = transform
        self.phase = phase
        self.flow_h = flow_h
        self.flow_w = flow_w

    def __getitem__(self, index):
        tgt_img_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_10.png')
        ref_img_path =  self.root.joinpath('data_stereo_flow', self.phase, 'colored_0',str(index).zfill(6)+'_11.png')
        gt_flow_path = self.root.joinpath('data_stereo_flow', self.phase, 'flow_occ', str(index).zfill(6)+'_10.png')

        tgt_img = load_as_float(tgt_img_path)
        ref_img = load_as_float(ref_img_path)

        u,v,valid = flow_io.flow_read_png(gt_flow_path)
        #gtFlow = scale_flow(np.dstack((u,v,valid)), h=self.flow_h, w=self.flow_w)
        gtFlow = np.dstack((u,v,valid))
        gtFlow = torch.FloatTensor(gtFlow.transpose(2,0,1))

        intrinsics = np.eye(3)
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + [ref_img], np.copy(intrinsics))
            tgt_img = imgs[0]
            ref_img = imgs[1]
        else:
            intrinsics = np.copy(intrinsics)
        return tgt_img, ref_img, intrinsics, np.linalg.inv(intrinsics), gtFlow

    def __len__(self):
        return self.N
