# Author: @sentient07
# Code adapted from DIF-Net and SIREN

import torch
import torch.nn.functional as F


def linear_plate_kernel(x1, x2, epsilon=5.):
    # make pred_pts -> (B, P, 1, 3)
    cdist_mat = torch.sqrt(
        ((x1[:, :, :, None, :] - x2[:, :, None, :, :]) ** 2).sum(-1))
    k_linear = cdist_mat
    return k_linear


def get_sdf_interp(gt_pts_sdfs, pred_pts):
    from pytorch3d.ops import ball_query, knn_points
    gt_pts_sdfs = gt_pts_sdfs.float()
    gt_pts = gt_pts_sdfs[:, :, :3]
    gt_sdfs = gt_pts_sdfs[:, :, 3]
    knn = knn_points(pred_pts, gt_pts, K=8, return_nn=True)
    knn_idx = knn.idx
    knn_pts = knn.knn
    kernel_matrix = linear_plate_kernel(knn_pts, knn_pts)
    closest_sdfs = gt_sdfs[torch.arange(gt_pts_sdfs.shape[0]).unsqueeze(
        1), knn_idx.view(gt_pts_sdfs.shape[0], -1)].view(kernel_matrix.size()[:-1])
    X_mat = torch.linalg.solve(kernel_matrix, closest_sdfs)
    closest_sdfs_interpolated = torch.einsum(
        'ijlk,ijk->ijl', linear_plate_kernel(pred_pts.unsqueeze(-2), knn_pts), X_mat)
    return closest_sdfs_interpolated


def training_loss(model_output, gt, deform_sdf_loss=True):
    n_surface_pt = gt['n_surface_pts'].squeeze().unique().item()
    n_space_pt = gt['n_space_pts'].squeeze().unique().item()
    gt_sdf = gt['sdf']
    template_sdfs = gt['template_sdfs']
    gt_normals = gt['normals']
    corresp_pts = gt['corresp_pts']
    pred_sdf = model_output['model_out']
    pred_shape_pt = model_output['new_coords']
    embeddings = model_output['latent_vec']
    gradient_sdf = model_output['grad_sdf']
    gradient_deform = model_output['grad_deform']
    gt_pts_sdfs = gt['gt_pts_sdfs']
    jacob_mat = model_output['jacob_mat']

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(
        gt_sdf != -1, torch.clamp(pred_sdf, -0.5, 0.5) - torch.clamp(
            gt_sdf, -0.5, 0.5),
        torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf),
        torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(
        gt_sdf == 0, 1 - F.cosine_similarity(
            gradient_sdf, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient_sdf[..., : 1]))
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)
    # End of Sitzmann

    # On surface L2 Loss.
    surface_superv = torch.where(gt_sdf == 0, torch.abs(
        corresp_pts - pred_shape_pt)**2, torch.zeros_like(corresp_pts))

    grad_deform_constraint = gradient_deform.norm(dim=-1)

    vol_preserv_loss = torch.clamp(
        torch.abs(1-torch.linalg.det(jacob_mat)), -10., 10.)

    embeddings_constraint = torch.mean(embeddings ** 2)
    loss_dict = {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
                 'inter': inter_constraint.mean() * 5e2,
                 'normal_constraint': normal_constraint.mean() * 1e2,
                 'grad_constraint': grad_constraint.mean() * 5e1,
                 'embeddings_constraint': embeddings_constraint.mean() * 1e6,
                 'grad_deform_constraint': grad_deform_constraint.mean() * 5,
                 'on_surface_constraint': surface_superv.mean() * 5e3,
                 'vol_preserv_loss': vol_preserv_loss.mean()*1e-1}

    # Deform SDF Loss
    if deform_sdf_loss:
        deform_sdfs_gt = get_sdf_interp(
            gt_pts_sdfs, pred_shape_pt).view(
            gt_sdf.shape)
        deform_sdf_loss = torch.where(deform_sdfs_gt[:, :n_surface_pt] < 1e3,
                                      torch.clamp(deform_sdfs_gt[:, :n_surface_pt], -0.5, 0.5),
                                      torch.zeros_like(deform_sdfs_gt[:, :n_surface_pt]))
        clamped_gt_sdf = torch.clamp(
            deform_sdfs_gt[:, -n_space_pt//2:], -0.1, 0.1)
        clamped_def_sdf = torch.clamp(
            template_sdfs[:, -n_space_pt//2:], -0.1, 0.1)
        clamped_diff_sdf = clamped_gt_sdf - clamped_def_sdf
        deform_sdf_loss_space = torch.where(
            deform_sdfs_gt
            [:, -n_space_pt // 2:] < 1e3,
            clamped_diff_sdf, torch.zeros_like(
                clamped_diff_sdf))

        loss_dict['deform_sdf_loss'] = torch.abs(deform_sdf_loss).mean() * 2e2
        loss_dict['deform_sdf_loss_space'] = torch.abs(
            deform_sdf_loss_space).mean()*2e2

    return loss_dict


def inference_loss(model_output, gt, deformsdf_loss=False,
                   wo_normal_constraint=False):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']
    n_surface_pt = gt['n_surface_pts'].squeeze().unique().item()
    n_space_pt = gt['n_space_pts'].squeeze().unique().item()
    pred_sdf = model_output['model_out']
    gt_pts_sdfs = gt['gt_pts_sdfs']
    template_sdfs = gt['template_sdfs']
    embeddings = model_output['latent_vec']

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(
        gt_sdf != -1, torch.clamp(pred_sdf, -0.5, 0.5) - torch.clamp(
            gt_sdf, -0.5, 0.5),
        torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf),
                                   torch.exp(-1e2 * torch.abs(pred_sdf)))
    embeddings_constraint = torch.mean(embeddings ** 2)
    gradient_sdf = model_output['grad_sdf']
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)

    loss_dict = {
        'sdf': torch.abs(sdf_constraint).mean() * 3e3,
        'inter': inter_constraint.mean() * 5e2,
        'embeddings_constraint': embeddings_constraint.mean() * 1e6,
        'grad_constraint': grad_constraint.mean() * 5e1}
    if not wo_normal_constraint:
        normal_constraint = torch.where(
            gt_sdf == 0, 1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
            torch.zeros_like(gradient_sdf[..., : 1]))
        loss_dict['normal_constraint'] = normal_constraint.mean() * 1e2

    if deformsdf_loss:
        pred_shape_pt = model_output['new_coords']
        deform_sdfs_gt = get_sdf_interp(
            gt_pts_sdfs, pred_shape_pt).view(
            gt_sdf.shape)

        deform_sdf_loss = torch.clamp(
            deform_sdfs_gt[:, :n_surface_pt], -0.5, 0.5)
        deform_sdf_loss_space = torch.clamp(
            deform_sdfs_gt[:, -n_space_pt//2:], -0.1, 0.1) - torch.clamp(template_sdfs[:, -n_space_pt//2:], -0.1, 0.1)
        loss_dict['deform_sdf_loss'] = torch.abs(deform_sdf_loss).mean() * 2e2
        loss_dict['deform_sdf_loss_space'] = torch.abs(
            deform_sdf_loss_space).mean()*2e2
    # -----------------
    return loss_dict
