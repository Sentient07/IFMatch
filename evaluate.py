# Author: @Sentient07

import os.path as osp
import sys
from pathlib import Path

import configargparse
import numpy as np
import torch
import yaml
from scipy.io import savemat
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from model import IFMatchNet
from utils import *

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def build_argparse():
    p = configargparse.ArgumentParser()
    p.add_argument('--config', required=True,
                   help='Evaluation configuration')
    p.add_argument('--wo_normal_constraint', action='store_true',
                   help='Evaluating point cloud wo normals?')
    p.add_argument('--deformsdf_loss', action='store_true',
                   help='Use deform SDF loss?')
    p.add_argument('--rot_invariance', action='store_true',
                   help='Use rot_invariance?')
    p.add_argument('--redo', action='store_true',
                   help='Use rot_invariance?')
    return p.parse_args()


def evaluate(model, train_dataloader, file_name, model_dir,
             wo_normal_constraint=False, deformsdf_loss=False,
             **kwargs):

    assert file_name is not None, "Must provide to save"

    epochs = kwargs['epochs']
    lr = kwargs['lr']
    # Initialise LVs
    embedding = model.latent_codes(
        torch.zeros(1).long().cuda()).clone().detach()
    embedding.requires_grad = True
    optim = torch.optim.Adam(lr=lr, params=[embedding])

    checkpoints_dir = osp.join(model_dir, 'Codes')
    safe_make_dirs(checkpoints_dir)

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        net_losses = []

        for _ in range(epochs):
            for _, (model_input, gt) in enumerate(train_dataloader):
                model_input = {key: value.cuda()
                               for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                losses = model.embedding(
                    embedding, model_input, gt,
                    wo_normal_constraint=wo_normal_constraint,
                    deformsdf_loss=deformsdf_loss)
                net_loss = 0.
                for _, loss in losses.items():
                    single_loss = loss.mean()
                    net_loss += single_loss

                net_losses.append(net_loss.item())
                optim.zero_grad()
                net_loss.backward()
                optim.step()
                pbar.update(1)

        embed_save = embedding.detach().squeeze()
        torch.save(embed_save, osp.join(checkpoints_dir, '%s.pth' % file_name))


if __name__ == '__main__':
    # load configs
    args = build_argparse()

    with open(osp.join(args.config), 'r') as stream:
        meta_params = yaml.safe_load(stream)

    if args.deformsdf_loss:
        assert meta_params['mesh_dir'], 'Cannot use Deform Loss w/o mesh'

    model = IFMatchNet(**meta_params)
    model.load_state_dict(torch.load(meta_params['checkpoint_path']))

    for param in model.hyper_net_s.parameters():
        param.requires_grad = False
    for param in model.hyper_net_d.parameters():
        param.requires_grad = False
    model.cuda()

    # create save path
    root_path = osp.join(
        meta_params['logging_root'],
        meta_params['experiment_name'])
    safe_make_dirs(root_path)

    with open(meta_params['eval_split'], 'r') as file:
        all_names = file.read().split('\n')

    # optimize latent code for each test subject
    for file in all_names:
        save_path = osp.join(root_path, file)

        if osp.isfile(
                osp.join(root_path, 'Codes', '%s.pth' % file)) and not args.redo:
            continue

        sdf_dataset = dataset.PointCloudMulti(
            root_dir=[osp.join(
                meta_params['point_cloud_path'],
                file + '.mat')],
            is_train=False, **meta_params)

        dataloader = DataLoader(
            sdf_dataset, shuffle=False, collate_fn=sdf_dataset.collate_fn,
            batch_size=1, pin_memory=True, num_workers=0, drop_last=True)

        evaluate(
            model=model, train_dataloader=dataloader, model_dir=root_path,
            file_name=file, wo_normal_constraint=args.wo_normal_constraint,
            deformsdf_loss=args.deformsdf_loss, **meta_params)
