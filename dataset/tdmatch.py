import random, os, sys, pickle, open3d
import torch.utils.data as data
import numpy as np
from dataset.common import FarthestSampler, to_tsfm, get_correspondences, to_o3d_pcd
from scipy.spatial.distance import cdist
import torch
from scipy.spatial.transform import Rotation
from lib.benchmark_utils import to_tensor


class TDMatchDataset(data.Dataset):
    '''
       Load subsampled coordinates, relative rotation and translation
       Output(torch.Tensor):
           src_pcd: (N, 3)
           tgt_pcd: (M, 3)
           rot: (3, 3)
           trans: (3, 1)
       '''

    def __init__(self, infos, config, data_augmentation=True):
        super(TDMatchDataset, self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.data_augmentation = data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.overlap_radius
        self.points_lim = 30000
        self.rot_factor = 1.
        self.augment_noise = config.augment_noise

    def __getitem__(self, index):
        # get transformation
        rot = self.infos['rot'][index]
        trans = self.infos['trans'][index]
        # get point cloud
        src_path = os.path.join(self.base_dir, self.infos['src'][index])
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][index])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        # if we get too many points, we do some downsampling
        if (src_pcd.shape[0] > self.points_lim):
            idx = np.random.permutation(src_pcd.shape[0])[:self.points_lim]
            src_pcd = src_pcd[idx]

        if (tgt_pcd.shape[0] > self.points_lim):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.points_lim]
            tgt_pcd = tgt_pcd[idx]

        # add gaussian noise
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2. / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        if (trans.ndim == 1):
            trans = trans[:, None]

        # get correspondences
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.search_voxel_size)
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

    def __len__(self):
        return len(self.infos['rot'])
