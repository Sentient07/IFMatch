
# get_faust_p2p.py

import json
import os
import os.path as osp
import sys
from itertools import permutations
from pathlib import Path

import configargparse
import numpy as np
import torch
import trimesh
import yaml
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model import IFMatchNet
from utils import *

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def fetch_eval_pairs(eval_file):
    eval_pairs = []
    for i in open(eval_file).readlines():
        eval_pairs.append((i.rstrip().split(',')[0], i.rstrip().split(',')[1]))
    return eval_pairs


def build_argparse():
    p = configargparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Config file')
    p.add_argument('--dense_template', action='store_true',
                   help='Use subdivided template')
    p.add_argument('--latent_opt', action='store_true',
                   help='Optimise latent vec using CD')
    p.add_argument('--lr', default=1e-3, type=float, help='LR optimisation')
    p.add_argument('--n_iter', default=1000, type=int, help='Iter optimisation')
    p.add_argument('--skip', action='store_true',
                   help='Skip already recon template')
    p.add_argument('--save_recon', action='store_true',
                   help='Saving reconstruction')
    p.add_argument('--save_indv_maps', action='store_true',
                   help='Saving indv maps as txt file')
    p.add_argument(
        '--eval_pairs_file', type=str, default=None,
        help='Eval pairs as txt file')
    return p.parse_args()


def recon_template(
        model, mesh_names, templ_dir, templ_m_scaled, meta_params, lr, n_iter,
        skip=False, do_latent_opt=True, save_recon=True):
    template_pts = {}
    for i in tqdm(mesh_names):
        if skip:
            templ_path = osp.join(templ_dir, '%s.ply' % i)
            assert osp.isfile(templ_path), "Template not found."
            template_pts[i] = np.array(trimesh.load(
                templ_path, process=False).vertices)
            continue

        src_pcl = loadmat(osp.join(meta_params['vertex_path'],
                                   '%s.mat' % i))['p'][..., :3]

        latent_vec = torch.load(osp.join(meta_params['logging_root'],
                                         meta_params['experiment_name'],
                                         'Codes', '%s.pth' % i)).cuda()
        coords = torched(np.array(templ_m_scaled.vertices)).unsqueeze(0)
        deformed_vert = model.get_shape_coords(coords, latent_vec)
        deformed_vert = deformed_vert[0].detach().cpu().numpy()
        if do_latent_opt:
            src_pcl_cuda = torched(src_pcl)
            latent_vec_opt = model.latent_opt(
                latent_vec, lr, coords, src_pcl_cuda, n_iter=n_iter)
            deformed_vert_opt = model.get_shape_coords(coords, latent_vec_opt)
            template_pts[i] = deformed_vert_opt[0].data.cpu().numpy()
            if save_recon:
                recon_m = trimesh_from_vf(template_pts[i], templ_m_scaled.faces)
                _ = recon_m.export(osp.join(templ_dir, '%s.ply' % i))

    return template_pts


def dump_p2p_maps(template_pts, test_combo, meta_params, root_dir,
                  map_save_dir=None, do_latent_opt=True):
    p2p_json = {}
    for ind, (i, j) in enumerate(tqdm(test_combo)):
        src_pcl = loadmat(osp.join(meta_params['vertex_path'], '%s.mat' % i))[
            'p'][..., :3]
        src_recon = template_pts[i]
        tar_pcl = loadmat(osp.join(meta_params['vertex_path'], '%s.mat' % j))[
            'p'][..., :3]
        tar_recon = template_pts[j]
        neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        neigh.fit(src_recon)
        nn_ind = neigh.kneighbors(src_pcl, return_distance=False)
        closest_points = tar_recon[nn_ind]
        closest_points = np.mean(closest_points, 1, keepdims=False)
        neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        neigh.fit(tar_pcl)
        idx_knn = neigh.kneighbors(closest_points, return_distance=False)
        cp = np.arange(len(tar_pcl))[np.squeeze(idx_knn)].tolist()
        p2p_json[str(i) + "_" + str(j)] = cp
        if map_save_dir:
            file_name = str(i) + "_" + str(j) + '.txt'
            with open(osp.join(map_save_dir, file_name), 'w') as fo:
                for item in cp:
                    # WARNING: Going to matlab convention
                    fo.write("%s\n" % (item+1))

        if do_latent_opt:
            save_path = osp.join(root_dir, '%s_p2p_lso.json' %
                                 meta_params['experiment_name'])
        else:
            save_path = osp.join(root_dir, '%s_p2p.json' %
                                 meta_params['experiment_name'])
    json.dump(p2p_json, open(save_path, 'w'))


if __name__ == '__main__':

    args = build_argparse()
    with open(osp.join(args.config), 'r') as stream:
        meta_params = yaml.safe_load(stream)

    model = IFMatchNet(**meta_params)
    model.load_state_dict(torch.load(meta_params['checkpoint_path']))
    model.cuda()

    root_dir = osp.join(
        meta_params['logging_root'],
        meta_params['experiment_name'])
    templ_dir = osp.join(root_dir, 'Template')
    safe_make_dirs(root_dir)
    safe_make_dirs(templ_dir)

    if args.save_indv_maps:
        map_save_dir = osp.join(root_dir, 'IndvMaps')
        safe_make_dirs(map_save_dir)
    else:
        map_save_dir = None

    if args.dense_template:
        templ_m_us = trimesh.load(
            './datasets/template_dense.ply', process=False)
    else:
        templ_m_us = trimesh.load('./datasets/template.ply', process=False)

    # Remember to scale to DeepSDF standards for a valid NNSearch
    templ_v_scaled = scale_to_unit_sphere(np.array(templ_m_us.vertices))/1.03
    templ_m_scaled = trimesh_from_vf(templ_v_scaled, templ_m_us.faces)

    # Create permutation pairs
    mesh_names = sorted([Path(i).stem
                         for i in os.listdir(osp.join(root_dir, 'Meshes'))])
    if args.eval_pairs_file:
        test_combo = fetch_eval_pairs(args.eval_pairs_file)
    else:
        test_combo = [i for i in permutations(mesh_names, 2)]

    template_pts = recon_template(
        model, mesh_names, templ_dir, templ_m_scaled, meta_params, args.lr,
        args.n_iter, skip=args.skip, do_latent_opt=True,
        save_recon=args.save_recon)
    dump_p2p_maps(template_pts, test_combo, meta_params, root_dir,
                  map_save_dir=map_save_dir, do_latent_opt=True)
