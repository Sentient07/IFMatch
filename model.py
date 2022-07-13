
# Author: @Sentient07
# Adapted from code of DIF-Net and SIREN
import numpy as np
import torch
from torch import nn
import modules
from meta_modules import HyperNetwork
from loss import *
from tqdm import tqdm
from pytorch3d.loss.chamfer import chamfer_distance


class IFMatchNet(nn.Module):
    def __init__(self, num_instances, latent_dim=128, act_type='sine',
                 hyper_hidden_layers=1, hyper_hidden_features=256,
                 hidden_num=128, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        # SDFNet
        self.sdf_net = modules.SingleBVPNet(
            coord_type='coords_s', type=act_type, mode='mlp',
            hidden_features=hidden_num, num_hidden_layers=3, in_features=3,
            out_features=1, use_dropout=True)

        # DeFieldNet
        self.deform_net = modules.SingleBVPNet(
            coord_type='coords_t', type=act_type, mode='mlp',
            hidden_features=hidden_num, num_hidden_layers=3, in_features=3,
            out_features=3, use_dropout=False)

        # Hyper-Nets
        self.hyper_net_d = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=hyper_hidden_layers,
            hyper_hidden_features=hyper_hidden_features,
            hypo_module=self.deform_net)

        self.hyper_net_s = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=hyper_hidden_layers,
            hyper_hidden_features=hyper_hidden_features,
            hypo_module=self.sdf_net)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input['instance_idx']
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self, instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    # for training

    def forward(self, model_input, gt, deform_sdf_loss=True, **kwargs):

        instance_idx = model_input['instance_idx']
        coords_t = model_input['coords_t']  # 3 dimensional input coordinates
        coords_s = model_input['coords_s']  # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net
        embedding = self.latent_codes(instance_idx)
        hypo_params_d = self.hyper_net_d(embedding)
        hypo_params_s = self.hyper_net_s(embedding)

        model_input_t = {'coords_t': coords_t}
        model_output = self.deform_net(model_input_t, params=hypo_params_d)

        deformation = model_output['model_out'][
            :, :, : 3]  # 3 dimensional deformation field
        new_coords = coords_t + deformation  # deform into template space

        x = model_output['model_in']
        n_batch = x.shape[0]
        u = deformation[:, :, 0]
        v = deformation[:, :, 1]
        w = deformation[:, :, 2]
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(
            u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_v = torch.autograd.grad(
            v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        grad_w = torch.autograd.grad(
            w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        # gradient of deformation wrt. input position
        grad_deform = torch.stack([grad_u, grad_v, grad_w], dim=2)
        jacob_mat = torch.cat((grad_u, grad_v, grad_w),
                              dim=-1).reshape(n_batch, -1, 3, 3).contiguous()
        model_input_s = {'coords_s': coords_s}

        model_output_sdf = self.sdf_net(model_input_s, params=hypo_params_s)

        sdf_final = model_output_sdf['model_out']  # SDF value in object space

        grad_sdf = torch.autograd.grad(
            sdf_final, [coords_s],
            grad_outputs=torch.ones_like(sdf_final),
            create_graph=True)[0]

        model_out = {
            'model_in': model_output['model_in'],
            'new_coords': new_coords, 'grad_deform': grad_deform,
            'model_out': sdf_final, 'latent_vec': embedding,
            'hypo_params_d': hypo_params_d, 'grad_sdf': grad_sdf,
            'hypo_params_s': hypo_params_s, 'jacob_mat': jacob_mat}

        losses = training_loss(model_out, gt, deform_sdf_loss=deform_sdf_loss)
        return losses

    def latent_opt(self, latent_vec, lr, templ_coord, src_coord,
                   n_iter=1000):
        embedding = latent_vec.clone().detach().cuda().float()
        embedding.requires_grad = True
        optim = torch.optim.Adam(lr=lr, params=[embedding])
        for param in self.hyper_net_d.parameters():
            param.requires_grad = False
        src_coord.requires_grad = False
        self.hyper_net_d.eval()
        new_coords = templ_coord

        for i in tqdm(range(n_iter)):
            optim.zero_grad()
            model_in = {'coords_t': templ_coord}
            hypo_params_d = self.hyper_net_d(embedding)
            model_output = self.deform_net(model_in, params=hypo_params_d)
            deformation = model_output['model_out'][:, :, :3]
            new_coords = templ_coord + deformation
            loss, _ = chamfer_distance(src_coord.unsqueeze(0), new_coords)
            loss.backward()
            optim.step()
        return embedding.detach()

    def embedding(
            self, embed, model_input, gt, deformsdf_loss=False,
            wo_normal_constraint=False):

        coords_s = model_input['coords_s']  # 3 dimensional input coordinates

        model_input_sdf = {'coords_s': coords_s}
        hypo_params_s = self.hyper_net_s(embed)

        model_output_sdf = self.sdf_net(model_input_sdf, params=hypo_params_s)

        sdf_final = model_output_sdf['model_out']  # SDF value in shape space
        model_out = {'model_in': coords_s,
                     'model_out': sdf_final, 'latent_vec': embed}

        grad_sdf = torch.autograd.grad(sdf_final, [coords_s], grad_outputs=torch.ones_like(
            sdf_final), create_graph=True)[0]  # normal direction in original shape space
        model_out['grad_sdf'] = grad_sdf

        if deformsdf_loss:
            hypo_params_d = self.hyper_net_d(embed)
            model_input_deform = {'coords_t': model_input['coords_t']}
            model_output_deform = self.deform_net(
                model_input_deform, params=hypo_params_d)
            deformation = model_output_deform['model_out'][:, :, :3]
            new_coords = model_input['coords_t'] + deformation
            model_out['new_coords'] = new_coords

        losses = inference_loss(model_out, gt,
                                deformsdf_loss=deformsdf_loss,
                                wo_normal_constraint=wo_normal_constraint)

        return losses

    def get_shape_coords(self, coords, embedding):
        with torch.no_grad():
            model_in = {'coords_t': coords}
            hypo_params_d = self.hyper_net_d(embedding)
            model_output = self.deform_net(model_in, params=hypo_params_d)
            deformation = model_output['model_out'][:, :, :3]
            new_coords = coords + deformation
            return new_coords
