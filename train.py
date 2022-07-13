# Author: @Sentient07

import io
import os
import os.path as osp
import sys
import time
import configargparse
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from model import IFMatchNet
from utils import *

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def build_argparse():
    p = configargparse.ArgumentParser()
    p.add_argument('--config', type=str, default='',
                   help='training configuration.')
    p.add_argument('--train_split', type=str, default='',
                   help='training subject names.')
    p.add_argument('--logging_root', type=str, default='./logs',
                   help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                   help='Name of subdirectory where checkpoints will be saved.')
    p.add_argument('--point_cloud_path', type=str, default='',
                   help='training data path.')

    # General training options
    p.add_argument('--batch_size', type=int, default=16,
                   help='training batch size.')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='learning rate. default=1e-4')
    p.add_argument('--epochs', type=int, default=35,
                   help='Number of epochs to train for.')

    p.add_argument('--act_type', type=str, default='sine',
                   help='Activation function')
    p.add_argument('--max_points', type=int, default=200000,
                   help='number of surface points for each epoch.')
    p.add_argument('--wo_deformsdf_loss', action='store_true',
                   help='Opposite OF Use deform SDF loss?')
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help='Load from?')
    p.add_argument('--start_epoch', type=int, default=0,
                   help='Starting epoch')
    return p.parse_args()


def train(model, train_dataloader, epochs, lr, save_interv, model_dir,
          start_epoch=0, deformsdf_loss=True, **kwargs):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    if not osp.isdir(model_dir):
        os.makedirs(model_dir)

    checkpoints_dir = osp.join(model_dir, 'checkpoints')
    safe_make_dirs(checkpoints_dir)

    with tqdm(total=len(train_dataloader) * (epochs-start_epoch)) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            model.train()
            if epoch % save_interv:
                torch.save(model.state_dict(), osp.join(
                    checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" %
                           (epoch, train_loss, time.time() - start_time))

            for _, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda()
                               for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                losses = model(model_input, gt,
                               deform_sdf_loss=deformsdf_loss, **kwargs)
                train_loss = 0.
                for _, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss
                train_losses.append(train_loss.item())
                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)
        torch.save(model.cpu().state_dict(), osp.join(
            checkpoints_dir, 'model_final.pth'))


if __name__ == '__main__':
    args = build_argparse()
    if args.config == '':
        meta_params = vars(args)
    else:
        with open(args.config, 'r') as stream:
            meta_params = yaml.safe_load(stream)

    with open(meta_params['train_split'], 'r') as file:
        all_names = file.read().split('\n')

    data_path = [osp.join(meta_params['point_cloud_path'],
                          f + '.mat') for f in all_names]
    sdf_dataset = dataset.PointCloudMulti(root_dir=data_path, **meta_params)
    dataloader = DataLoader(
        sdf_dataset, shuffle=True, collate_fn=sdf_dataset.collate_fn,
        batch_size=meta_params['batch_size'],
        pin_memory=True, num_workers=meta_params['batch_size'],
        drop_last=True)
    print('Total subjects: ', sdf_dataset.num_instances)
    meta_params['num_instances'] = sdf_dataset.num_instances
    # define DIF-Net
    model = IFMatchNet(**meta_params)
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print("[*] Loaded from %s" % args.checkpoint_path)
    model.cuda()

    # create save path
    root_path = osp.join(
        meta_params['logging_root'],
        meta_params['experiment_name'])
    safe_make_dirs(root_path)

    with io.open(osp.join(root_path, 'model.yml'), 'w', encoding='utf8') as outfile:
        yaml.dump(meta_params, outfile,
                  default_flow_style=False, allow_unicode=True)

    # main training loop
    print("[*] Starting to train from Epoch %d" % args.start_epoch)
    use_deform_sdf_loss = not args.wo_deformsdf_loss  # Because default=True
    train(model=model, train_dataloader=dataloader, model_dir=root_path,
          start_epoch=args.start_epoch,
          deformsdf_loss=use_deform_sdf_loss, **meta_params)
