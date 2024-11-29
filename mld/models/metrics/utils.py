import numpy as np

import scipy.linalg
from scipy.ndimage import uniform_filter1d

import torch
from torch import linalg


# Motion Reconstruction

def calculate_mpjpe(gt_joints: torch.Tensor, pred_joints: torch.Tensor, align_root: bool = True) -> torch.Tensor:
    """
    gt_joints: num_poses x num_joints x 3
    pred_joints: num_poses x num_joints x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, \
        f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    if align_root:
        gt_joints = gt_joints - gt_joints[:, [0]]
        pred_joints = pred_joints - pred_joints[:, [0]]

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1)  # num_poses x num_joints
    mpjpe = mpjpe.mean(-1)  # num_poses

    return mpjpe


# Text-to-Motion

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dists: N1 x N2
    dists[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def euclidean_distance_matrix_np(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dists: N1 x N2
    dists[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat: torch.Tensor, top_k: int) -> torch.Tensor:
    size = mat.shape[0]
    gt_mat = (torch.unsqueeze(torch.arange(size), 1).to(mat.device).repeat_interleave(size, 1))
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat


def calculate_activation_statistics(activations: torch.Tensor) -> tuple:
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_np(activations: np.ndarray) -> tuple:
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance_np(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6) -> float:
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_diversity(activation: torch.Tensor, diversity_times: int) -> float:
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices],
                       axis=1)
    return dist.mean()


def calculate_diversity_np(activation: np.ndarray, diversity_times: int) -> float:
    assert len(activation.shape) == 2
    assert activation.shape[0] >= diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = scipy.linalg.norm(activation[first_indices] -
                             activation[second_indices],
                             axis=1)
    return dist.mean()


def calculate_multimodality_np(activation: np.ndarray, multimodality_times: int) -> float:
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent,
                                   multimodality_times,
                                   replace=False)
    second_dices = np.random.choice(num_per_sent,
                                    multimodality_times,
                                    replace=False)
    dist = scipy.linalg.norm(activation[:, first_dices] -
                             activation[:, second_dices],
                             axis=2)
    return dist.mean()


# Motion Control

def calculate_skating_ratio(motions: torch.Tensor, dataset_name: str) -> tuple:
    thresh_height = 0.05
    fps = 20.0
    thresh_vel = 0.50
    avg_window = 5  # frames

    # XZ plane, y up
    # 10 left, 11 right foot. (HumanML3D)
    # 15 left, 20 right foot. (KIT)
    # motions [bsz, fs, 22 or 21, 3]

    if dataset_name == 'humanml3d':
        foot_idx = [10, 11]
    elif dataset_name == 'kit':
        foot_idx = [15, 20]
    else:
        raise ValueError(f'Invalid Dataset: {dataset_name}')

    verts_feet = motions[:, :, foot_idx, :].detach().cpu().numpy()  # [bsz, fs, 2, 3]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, 1:, :, [0, 2]] -
                                          verts_feet[:, :-1, :, [0, 2]], axis=-1) * fps  # [bsz, fs-1, 2]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, :, 1]  # [bsz, fs, 2]
    # If feet touch ground in adjacent frames
    feet_contact = np.logical_and((verts_feet_height[:, :-1, :] < thresh_height),
                                  (verts_feet_height[:, 1:, :] < thresh_height))  # [bs, fs-1, 2]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    skating = np.logical_and(skating, (vel_avg > thresh_vel))

    # Both feet slide
    skating = np.logical_or(skating[:, :, 0], skating[:, :, 1])  # [bs, fs-1]
    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]

    return skating_ratio, skate_vel


def calculate_trajectory_error(dist_error: torch.Tensor, mean_err_traj: torch.Tensor,
                               mask: torch.Tensor, strict: bool = True) -> torch.Tensor:
    if strict:
        # Traj fails if any of the key frame fails
        traj_fail_02 = 1.0 - int((dist_error <= 0.2).all().item())
        traj_fail_05 = 1.0 - int((dist_error <= 0.5).all().item())
    else:
        # Traj fails if the mean error of all keyframes more than the threshold
        traj_fail_02 = int((mean_err_traj > 0.2).item())
        traj_fail_05 = int((mean_err_traj > 0.5).item())
    all_fail_02 = (dist_error > 0.2).sum() / mask.sum()
    all_fail_05 = (dist_error > 0.5).sum() / mask.sum()

    return torch.tensor([traj_fail_02, traj_fail_05, all_fail_02, all_fail_05, dist_error.sum() / mask.sum()])


def control_l2(motion: torch.Tensor, hint: torch.Tensor, hint_mask: torch.Tensor) -> torch.Tensor:
    loss = torch.norm((motion - hint) * hint_mask, p=2, dim=-1)
    return loss
