# Author: @Sentient07
# Adapted from code of DIF-Net and SIREN

import os
import numpy as np
import torch
import trimesh

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import *
from pathlib import Path
from trimesh.triangles import barycentric_to_points

class PointCloud_with_FreePoints(Dataset):
    def __init__(self, template_info, pointcloud_path, on_surface_points,
                 instance_idx=None, max_points=-1, mesh_dataset_dir=None,
                 is_train=True, ext='.ply', n_sdf_pts=280000):
        super().__init__()

        self.instance_idx = instance_idx
        self.is_train = is_train
        self.n_sdf_pts = n_sdf_pts
        self.on_surface_points = on_surface_points
        self.max_points = max_points
        self.template_info = template_info
        
        self.templ_surf_pc = self.template_info['templ_surf_pc']
        self.templ_sdf_pts_coord = self.template_info['templ_sdf_pts_coord']
        self.templ_sdf_pts_sdf = self.template_info['templ_sdf_pts_sdf']
        self.templ_trig_id = self.template_info['templ_trig_id']
        self.templ_baryc = self.template_info['templ_baryc']
        
        # Surface points and normal information
        point_cloud = loadmat(pointcloud_path)
        point_cloud = point_cloud['p']
        self.coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # SDF Points
        if 'surface_pts_n_normal' in pointcloud_path:
            free_points = loadmat(pointcloud_path.replace('surface_pts_n_normal','free_space_pts'))
        else:
            assert 'vertex_pts_n_normal' in pointcloud_path, "Invalid Path"
            free_points = loadmat(pointcloud_path.replace('vertex_pts_n_normal','free_space_pts'))

        self.free_points_psdf = free_points['p_sdf']
        self.free_points_coords = free_points['p_sdf'][:,:3]
        self.free_points_sdf = free_points['p_sdf'][:,3:]

        # surface points
        self.file_name = Path(pointcloud_path).stem
        if mesh_dataset_dir is None:
            assert is_train is False, "Need meshes to train"
            self.current_mesh = None
            self.gt_pts_sdfs = np.empty((1,4))
        else:
            self.current_mesh_old = trimesh.load(os.path.join(mesh_dataset_dir, self.file_name + ext), process=False)
            # Scale to DeepSDF standards
            scaled_vert = scale_to_unit_sphere(self.current_mesh_old.vertices)/1.03
            self.current_mesh = trimesh_from_vf(scaled_vert, self.current_mesh_old.faces)
            self.gt_pts_sdfs = self.prepare_sdf_pts()

    def __len__(self):
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.coords.shape[0] // self.on_surface_points

    def prepare_sdf_pts(self, n_surf_pts=70000):
        surface_pts, _ = trimesh.sample.sample_surface(self.current_mesh, n_surf_pts)
        surface_sdf = np.zeros((len(surface_pts)))
        surface_pts_sdf = np.c_[surface_pts, surface_sdf]
        combined_pts_sdf = np.r_[self.free_points_psdf[:self.n_sdf_pts, :], 
                                 surface_pts_sdf]
        rand_ind = np.random.choice(np.arange(len(combined_pts_sdf)),
                                    size=len(combined_pts_sdf), replace=False)
        return combined_pts_sdf[rand_ind]
    

    def get_proj_info_fast(self, trig_id, bary_coord, n_pts):
        cur_mesh = self.current_mesh
        corresp_pts = np.zeros((n_pts, 3))
        src_nearest_trigs = cur_mesh.vertices[cur_mesh.faces[trig_id]]
        corresp_pts[:len(bary_coord),:] = barycentric_to_points(src_nearest_trigs, bary_coord)
        return corresp_pts

    def __getitem__(self, _):
        point_cloud_size = self.coords.shape[0]
        free_point_size = self.free_points_coords.shape[0]

        off_surface_samples = 2*self.on_surface_points 
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples//2, 3))
        free_rand_idcs = np.random.choice(free_point_size, size=off_surface_samples//2)
        free_points_coords = self.free_points_coords[free_rand_idcs,:]

        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1
        sdf[self.on_surface_points+off_surface_samples//2:,:] = self.free_points_sdf[free_rand_idcs]

        coords = np.concatenate((on_surface_coords, off_surface_coords, free_points_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        # Do the same for templates
        
        temp_rand_idcs = np.random.choice(self.templ_surf_pc.shape[0],
                                          size=self.on_surface_points)
        template_on_surface_coords = self.templ_surf_pc[temp_rand_idcs, :]
        template_off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples//2, 3))
        temp_free_rand_idcs = np.random.choice(self.templ_sdf_pts_coord.shape[0],
                                               size=off_surface_samples//2)
        template_free_points_coords = self.templ_sdf_pts_coord[temp_free_rand_idcs,:]
        template_coords = np.concatenate((template_on_surface_coords, template_off_surface_coords, template_free_points_coords))
        # SDF for template
        template_sdfs = np.zeros((template_coords.shape[0], 1))
        template_sdfs[self.on_surface_points:, :] = -1
        template_sdfs[self.on_surface_points+off_surface_samples//2:,:] = self.templ_sdf_pts_sdf[temp_free_rand_idcs]

        # While training, pick pairs by barycentric correspondence
        if self.is_train:
            corresp_pts = self.get_proj_info_fast(self.current_mesh,
                                                  self.templ_trig_id[temp_rand_idcs],
                                                  self.templ_baryc[temp_rand_idcs],
                                                  total_samples)

            coords_tr = torch.from_numpy(coords).float()
            normals_tr = torch.from_numpy(normals).float()
            corresp_pts_tr =torch.from_numpy(corresp_pts).float()
            
            return {'coords_s': coords_tr.float(),
                    'coords_t': torch.from_numpy(template_coords).float(),
                    'sdf': torch.from_numpy(sdf).float(),
                    'normals': normals_tr.float(),
                    'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long(),
                    'corresp_pts' : corresp_pts_tr.float(),
                    'n_surface_pts' : torch.IntTensor([self.on_surface_points]),
                    'n_space_pts' : torch.IntTensor([off_surface_samples]),
                    'gt_pts_sdfs' : torch.from_numpy(self.gt_pts_sdfs).float(),
                    'template_sdfs' : torch.from_numpy(template_sdfs).float()}
        else:
            return {'coords_s': torch.from_numpy(coords).float(),
                    'coords_t': torch.from_numpy(template_coords).float(),
                    'sdf': torch.from_numpy(sdf).float(),
                    'normals': torch.from_numpy(normals).float(),
                    'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long(),
                    'n_surface_pts' : torch.IntTensor([self.on_surface_points]),
                    'n_space_pts' : torch.IntTensor([off_surface_samples]),
                    'gt_pts_sdfs' : torch.from_numpy(self.gt_pts_sdfs).float(),
                    'template_sdfs' : torch.from_numpy(template_sdfs).float()}


class PointCloudMulti(Dataset):
    def __init__(self, root_dir, on_surface_points,
                 max_points=-1, is_train=True, n_sdf_pts=280000,
                 **kwargs):
        
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.n_sdf_pts = n_sdf_pts
        
        self.template_name = kwargs.get('template_name', 'template')
        assertion_msg = "Place the template into '/datasets/"
        assert os.path.isfile('./datasets/%s.ply'%self.template_name), assertion_msg
        # Load Template stuff
        self.template_info = self._get_template_info()
        mesh_dir = kwargs.get('mesh_dir', None)
        ext=kwargs.get('mesh_ext', '.ply')
        assert (len(self.root_dir) != 0), "No objects!"

        self.all_instances = []        
        for idx, dir in enumerate(tqdm(self.root_dir)):
            self.all_instances.append(PointCloud_with_FreePoints(self.template_info,
                                                                 instance_idx=idx,
                                                                 pointcloud_path=dir,
                                                                 on_surface_points=on_surface_points,
                                                                 max_points=max_points,
                                                                 mesh_dataset_dir=mesh_dir,
                                                                 is_train=self.is_train,
                                                                 ext=ext,
                                                                 n_sdf_pts=n_sdf_pts))
        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def _get_template_info(self):
        templ_pth = './datasets/%s.ply'%self.template_name
        templ_surf_pth = './datasets/templates/surface_pts_n_normal/%s.mat'%self.template_name
        templ_sdf_pth = './datasets/templates/free_space_pts/%s.mat'%self.template_name
        assert osp.isfile(templ_surf_pth), "Surface points not found"
        assert osp.isfile(templ_sdf_pth), "SDF points not found"

        temp_mesh = trimesh.load(templ_pth, process=False)
        scaled_vert = scale_to_unit_sphere(temp_mesh.vertices)/1.03
        temp_mesh_sc = trimesh_from_vf(scaled_vert, temp_mesh.faces)
        templ_surf_mat = loadmat(templ_surf_pth)
        templ_surf_pc = templ_surf_mat['p'][:, :3]
        templ_sdf_pts = loadmat(templ_sdf_pth)['p_sdf']
        templ_sdf_pts_coord = templ_sdf_pts[:, :3]
        templ_sdf_pts_sdf = templ_sdf_pts[:, 3:]
        templ_trig_id = np.squeeze(templ_surf_mat['trig_id'])
        templ_baryc = templ_surf_mat['bary_coord']
        
        templ_info = {'templ_surf_pc': templ_surf_pc, 'templ_sdf_pts_coord': templ_sdf_pts_coord,
                      'templ_sdf_pts_sdf': templ_sdf_pts_sdf, 'templ_trig_id': templ_trig_id,
                      'templ_baryc': templ_baryc}
        return templ_info

    def get_instance_idx(self, idx):
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])
        if self.is_train:
            ground_truth = [{'sdf':obj['sdf'], 'normals': obj['normals'],
                             'corresp_pts' : obj['corresp_pts'],
                             'n_surface_pts' : obj['n_surface_pts'],
                             'n_space_pts' : obj['n_space_pts'],
                             'template_sdfs' : obj['template_sdfs'],
                             'gt_pts_sdfs' : obj['gt_pts_sdfs']} for obj in observations]
        else:
            ground_truth = [{'sdf':obj['sdf'], 'normals': obj['normals'],
                             'n_surface_pts' : obj['n_surface_pts'],
                             'n_space_pts' : obj['n_space_pts'],
                             'template_sdfs' : obj['template_sdfs'],
                             'gt_pts_sdfs' : obj['gt_pts_sdfs']} for obj in observations]

        return observations, ground_truth
