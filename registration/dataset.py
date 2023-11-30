import h5py
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset
from vis_utils import plot_single_pcd
from train_utils import *
# from model_utils import knn, batch_choice, FPFH, Conv1DBlock, Conv2DBlock, SVDHead
from model_utils import *
import random

def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class MVP_RG(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('../completion/data/MVP_Train_CP.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        if (self.prefix == 'val'):
            self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            if self.prefix == "train":
                self.input_data = np.array(f['incomplete_pcds'][()])
                self.gt_data = np.array(f['complete_pcds'][()])
                self.len = self.input_data.shape[0]
            elif self.prefix == "val":
                self.match_level = np.array(f['match_level'][:].astype('int32'))

                match_id = []
                for i in range(len(f['match_id'].keys())):
                    ds_data = f['match_id'][str(i)][:]
                    match_id.append(ds_data)
                self.match_id = np.array(match_id, dtype=object)
                self.src = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                self.complete = np.array(f['complete'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))
                self.pose_src = np.array(f['pose_src'][:].astype('float32'))
                self.pose_tgt = np.array(f['pose_tgt'][:].astype('float32'))
                self.len = self.src.shape[0]
        f.close()

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if (self.prefix == 'val'):
            src = self.src[index]
            tgt = self.tgt[index]
            complete = self.complete[index]
        
        if self.prefix == "train":
            list_ = range((index // 26) *26, (index // 26) *26 +26)
            complete = self.gt_data[index // 26]
            random_list = random.sample(list_, 2)
            src = self.input_data[random_list[0]]
            tgt = self.input_data[random_list[1]]
            
            
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            
            complete_src = complete
            complete_tgt = complete
        elif self.prefix == "val":
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]
            pose_src = self.pose_src[index]
            pose_tgt = self.pose_tgt[index]
            complete_src = complete @ pose_src[:3, :3].T + pose_src[:3, 3]
            complete_tgt = complete @ pose_tgt[:3, :3].T + pose_tgt[:3, 3]

        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)
        complete_src = torch.from_numpy(complete_src)
        complete_tgt = torch.from_numpy(complete_tgt)

        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            
            if (self.prefix == 'train'):
                return src, tgt, transform, transform, transform, complete_src, complete_tgt
            else:
                match_level = match_level
                rot_level = rot_level
                return src, tgt, transform, transform, transform, complete_src, complete_tgt, match_level, rot_level
        else:
            return src, tgt, complete_src, complete_tgt