# data_process.py
import argparse
import os.path as osp
from glob import glob
from pathlib import Path

import numpy as np
import trimesh
from mesh_to_sdf import get_surface_point_cloud as sample_mesh
from scipy.io import savemat
from tqdm import tqdm

from utils import safe_make_dirs, scale_to_unit_sphere, trimesh_from_vf


def save_npz_sdf_mat(npz_dir, save_dir):
    for npz in tqdm(glob(osp.join(npz_dir, '*.npz'))):
        sdf_pts = np.load(npz)
        pts = np.r_[sdf_pts['neg'][:, :3], sdf_pts['pos'][:, :3]]
        sdf = np.r_[sdf_pts['neg'][:, 3:], sdf_pts['pos'][:, 3:]]
        col_stacked = np.column_stack((pts, sdf))
        assert col_stacked.shape[-1] == 4
        cur_filename = Path(npz).stem + '.mat'
        savemat(osp.join(save_dir, cur_filename), {'p_sdf': col_stacked})


def save_xyz_sdf_mat(xyz_dir, save_dir, ext='xyz'):
    for xyz in tqdm(glob(osp.join(xyz_dir, '*.%s' % ext))):
        pts = np.array(trimesh.load(xyz, process=False).vertices)
        pts = scale_to_unit_sphere(pts)/1.03
        sdf = np.zeros((pts.shape[0]))
        col_stacked = np.column_stack((pts, sdf))
        assert col_stacked.shape[-1] == 4
        cur_filename = Path(xyz).stem + '.mat'
        savemat(osp.join(save_dir, cur_filename), {'p_sdf': col_stacked})


def save_surfpts_normals_mat(ply_dir, save_dir,
                             ext='ply', overwrite=False):
    all_meshes = [i for i in sorted(glob(osp.join(ply_dir, '*.'+ext)))]
    for mesh in tqdm(all_meshes):
        cur_filename = Path(mesh).stem + '.mat'
        save_name = osp.join(save_dir, cur_filename)
        if osp.isfile(save_name) and not overwrite:
            print("Skipping %s" % cur_filename)
            continue
        m = trimesh.load(mesh, process=False)
        scaled_vert = scale_to_unit_sphere(np.array(m.vertices))/1.03
        new_mesh = trimesh_from_vf(scaled_vert, faces=m.faces)
        sur_pc = sample_mesh(new_mesh, surface_point_method='scan',
                             bounding_radius=None, scan_count=100,
                             scan_resolution=400, sample_point_count=500000,
                             calculate_normals=True)
        col_stacked = np.column_stack((sur_pc.points, sur_pc.normals))[:500000]
        savemat(save_name, {'p': col_stacked})


def save_pc_normals_mat(ply_dir, save_dir,
                        ext='ply', with_normal=False):
    all_meshes = [i for i in sorted(glob(osp.join(ply_dir, '*.'+ext)))
                  if not i.split('/')[-1].endswith('Reconstruction.ply')]
    for mesh in tqdm(all_meshes):
        m = trimesh.load(mesh, process=False)
        if with_normal:
            if isinstance(m, trimesh.Trimesh):
                normals = m.vertex_normals
            else:
                normals = np.array(
                    m.metadata['ply_raw']['vertex']['data'].tolist())[:, 3:]
            old_vert = np.array(m.vertices)
            new_vert = scale_to_unit_sphere(old_vert)/1.03
        else:
            old_vert = np.array(m.vertices)
            new_vert = scale_to_unit_sphere(old_vert)/1.03
            normals = np.zeros_like(new_vert)
        col_stacked = np.column_stack((new_vert, normals))
        cur_filename = Path(mesh).stem + '.mat'
        savemat(osp.join(save_dir, cur_filename), {'p': col_stacked})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", dest="npz_dir",
                        help="Sampled SDF files from DeepSDF Pre-processing")
    parser.add_argument("--ply_dir", dest="data_dir",
                        help="Directory containing the geometry")
    parser.add_argument("--save_dir", dest="save_dir",
                        help="Place to save the pre-processed contents.")
    parser.add_argument("--noisy_data", action="store_true", dest="noisy_data",
                        help="If noisy data, normals aren't computed.")
    parser.add_argument("--extension", default="ply", dest="ext",
                        help="We only need it for noisy point clouds.")

    args = parser.parse_args()

    safe_make_dirs(args.save_dir)
    sdf_save_dir = osp.join(args.save_dir, 'free_space_pts')
    verts_save_dir = osp.join(args.save_dir, 'vertex_pts_n_normal')
    normal_save_dir = osp.join(args.save_dir, 'surface_pts_n_normal')

    # Convert the pre-processed SDF
    if args.noisy_data:
        save_xyz_sdf_mat(args.ply_dir, sdf_save_dir,
                         ext=args.ext)
    else:
        save_npz_sdf_mat(args.npz_dir, sdf_save_dir)

    # Extract vertex into .mat format
    save_pc_normals_mat(args.ply_dir, verts_save_dir,
                        ext=args.ext)

    if args.noisy_data:
        save_pc_normals_mat(args.ply_dir, normal_save_dir,
                            with_normal=False, ext=args.ext)
    else:
        save_pc_normals_mat(args.ply_dir, normal_save_dir)
