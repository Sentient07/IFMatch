# Author: @Sentient07

import os
import os.path as osp
import trimesh
import numpy as np
import torch
from tqdm import tqdm


def safe_make_dirs(cur_dir):
    if not osp.isdir(cur_dir):
        os.makedirs(cur_dir)


def save_xyz(pts, file_name):
    s = trimesh.util.array_to_string(pts)
    with open(file_name, 'w') as f:
        f.write("%s\n" % s)


def scale_to_unit_sphere(points, return_scale=False):
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    if return_scale:
        return points, scale
    return points


def trimesh_from_vf(v, f):
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def torched(np_array, dtype=torch.float32, device='cuda'):
    return torch.from_numpy(np_array).to(dtype).to(device=device)
