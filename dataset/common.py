import numpy as np
import open3d
import random
import torch
import pickle


class FarthestSampler:
    '''
    A class for farthest point sampling (FPS)
    '''
    def __init__(self):
        pass

    def calc_distance(self, p0, points):
        '''
        Calculate one-to-all distances
        :param p0: single point
        :param points: point clouds
        :return: one-to-all distances
        '''
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, points, k):
        '''
        Downsample give point cloud (nx3) into smaller point cloud (kx3) via farthest point sampling
        :param points: input point cloud in shape (nx3)
        :param k: number of points after downsampling
        :return: farthest_points: point cloud in shape (kx3) after farthest point sampling
        '''
        farthest_points = np.zeros(shape=(k, 3))
        farthest_points[0] = points[np.random.randint(points.shape[0])]
        distances = self.calc_distance(farthest_points[0], points)
        for i in range(1, k):
            farthest_points[i] = points[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distance(farthest_points[i], points))
        return farthest_points


def make_rotation_matrix(augment_axis, augment_rotation):
    '''
    Generate a random rotaion matrix
    :param augment_axis: 0 or 1, if set to 1, returned matrix will describe a rotation around a random selected axis from {x, y, z}
    :param augment_rotation: float number scales the generated rotation
    :return: rotaion matrix
    '''
    angles = np.random.rand(3) * 2. * np.pi * augment_rotation
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = rx @ ry @ rz
    if augment_axis == 1:
        return random.choice([rx, ry, rz])
    return rx @ ry @ rz


def make_translation_vector(augment_translation):
    '''
    Generate a random translation vector
    :param augment_translation: float number scales the generated translation
    :return: translation vector
    '''
    T = np.random.rand(3) * augment_translation
    return T


def to_tsfm(rot, trans):
    '''
    Make (4, 4) transformation matrix from rotation matrix and translation vector
    :param rot: (3, 3) rotation matrix
    :param trans: (3, 1) translation matrix
    :return: (4, 4) transformation matrix
    '''
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    '''
    Give source & target point clouds as well as the relative transformation between them, calculate correspondences according to give threshold
    :param src_pcd: source point cloud
    :param tgt_pcd: target point cloud
    :param trans: relative transformation between source and target point clouds
    :param search_voxel_size: given threshold
    :param K: if k is not none, select top k nearest neighbors from candidate set after radius search
    :return: (m, 2) torch tensor, consisting of m correspondences
    '''
    src_pcd.transform(trans)
    pcd_tree = open3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def load_info(path):
    '''
    read a dictionary from a pickle file
    :param path: path to the pickle file
    :return: loaded info
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)
