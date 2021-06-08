import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.stats
import scipy.integrate
import math
import os

import torch

import multiprocessing as mp
import subprocess
import hydra
import glob
import pandas
import yaml
from scipy.special import logsumexp

"""
Compute the KSD divergence using samples, adapted from the theano code
"""
# From https://github.com/YingzhenLi/SteinGrad/blob/master/hamiltonian/ksd.py
def KSD(z, Sqx, in_h_square=None):

    # compute the rbf kernel
    K, dimZ = z.shape
    sq_dist = pdist(z)
    pdist_square = squareform(sq_dist)**2
    if in_h_square is None:
        # use median
        median = np.median(pdist_square)
        h_square = 0.5 * median / np.log(K+1.0)
    else:
        h_square = in_h_square
    print("h_square", h_square)
    Kxy = np.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = np.dot(Sqx, z.T) - np.tile(np.sum(Sqx * z, 1, keepdims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = Sqxdy.T
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
    # M is a (K, K) tensor
    M = (np.dot(Sqx, Sqx.T) + Sqxdy + dxSqy + dxdy) * Kxy

    # the following for U-statistic
    M2 = M - np.diag(np.diag(M))
    return np.sum(M2) / (K * (K - 1))

def blockKSD(z, Sqx, num_blocks, h_square):
    K, dimZ = z.shape
    block_step = math.floor(K/num_blocks)
    culm_sum = 0
    for i in np.floor(np.linspace(0, K, num=num_blocks+1)[0:-1]).astype(int):
        for j in np.floor(np.linspace(0, K, num=num_blocks+1)[0:-1]).astype(int):
            zrow = z[i:i+block_step, :]
            zcol = z[j:j+block_step, :]
            Sqxrow = Sqx[i:i+block_step, :]
            Sqxcol = Sqx[j:j+block_step, :]
            pdist_square = cdist(zrow, zcol)**2
            Kxy = np.exp(- pdist_square / h_square / 2.0)
            Sqxdy = np.tile(np.sum(Sqxrow * zrow, 1, keepdims=True),
                (1, block_step)) - np.dot(Sqxrow, zcol.T)
            Sqxdy = Sqxdy / h_square
            dxSqy = (np.dot(Sqxcol, zrow.T) - \
                np.tile(np.sum(Sqxcol * zcol, 1, keepdims=True),
                (1, block_step))).T
            dxSqy = -dxSqy / h_square
            dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square

            M = (np.dot(Sqxrow, Sqxcol.T) + Sqxdy + dxSqy + dxdy) * Kxy

            if i == j:
                M = M - np.diag(np.diag(M))
            culm_sum += np.sum(M)
    return culm_sum / (K*(K-1))

def blockKSDparallel(z, Sqx, num_blocks, h_square, num_processes):
    K, dimZ = z.shape
    block_step = math.floor(K/num_blocks)
    data_chunks = []
    for i in np.floor(np.linspace(0, K, num=num_blocks+1)[0:-1]).astype(int):
        for j in np.floor(np.linspace(0, K, num=num_blocks+1)[0:-1]).astype(int):
            zrow = z[i:i+block_step, :]
            zcol = z[j:j+block_step, :]
            Sqxrow = Sqx[i:i+block_step, :]
            Sqxcol = Sqx[j:j+block_step, :]
            data_chunks.append(
                (zrow, zcol, Sqxrow, Sqxcol, h_square, block_step, dimZ, i, j))
    pool = mp.Pool(processes=num_processes)
    results = pool.map(blockKSDparallelCompute, data_chunks)
    culm_sum = np.sum(np.array(results))
    return culm_sum / (K*(K-1))

def blockKSDparallelCompute(data_chunk):
    zrow, zcol, Sqxrow, Sqxcol, h_square, block_step, dimZ, i, j = data_chunk
    pdist_square = cdist(zrow, zcol)**2
    Kxy = np.exp(- pdist_square / h_square / 2.0)
    Sqxdy = np.tile(np.sum(Sqxrow * zrow, 1, keepdims=True),
        (1, block_step)) - np.dot(Sqxrow, zcol.T)
    Sqxdy = Sqxdy / h_square
    dxSqy = (np.dot(Sqxcol, zrow.T) - \
        np.tile(np.sum(Sqxcol * zcol, 1, keepdims=True),
        (1, block_step))).T
    dxSqy = -dxSqy / h_square
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square

    M = (np.dot(Sqxrow, Sqxcol.T) + Sqxdy + dxSqy + dxdy) * Kxy

    if i == j:
        M = M - np.diag(np.diag(M))
    return np.sum(M)

            
def get_median_estimate(z, num_samples=1000):
    z_block = z[0:num_samples, :]
    sq_dist = pdist(z_block)
    pdist_square = squareform(sq_dist)**2
    return np.median(pdist_square)


def get_latest_checkpoint(dir_path, key=''):
    """
    Get path to latest checkpoint in directory
    :param dir_path: Path to directory to search for checkpoints
    :param key: Key which has to be in checkpoint name
    :return: Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f and ".pt" in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]


def estimate_kl(samples_a, samples_b, range_min, range_max):
    """
    Estimates the KL divergence between two sets of 1-D samples i.e KL(a||b)
    :param samples_a: First set of samples
    :param samples_b: Second set of samples
    :param range_min: The lower range of integration
    :param range_max: The upper range of integration
    """
    kde_a = scipy.stats.gaussian_kde(samples_a)
    kde_b = scipy.stats.gaussian_kde(samples_b)

    eps = 1e-5

    def kl(x):
        q = kde_a
        p = kde_b
        return q(x) * (np.log(q(x) + eps) - np.log(p(x) + eps))

    return scipy.integrate.quad(kl, range_min, range_max)[0]

def SKSD(x, Sqx, g):
    """
    Estimates the sliced KSD using pytorch functions and the squared
    exponential function
    :param x: samples (N x dim)
    :param Sqx: \nabla_x log p(x) evaluated at samples (N x dim)
    :param g: The slicing directions each row is one direction (dim x dim)
    """

    N = x.shape[0]

    # Project each sample in each of the g directions
    proj_x = torch.matmul(x, g.transpose(0,1)) # (N x dim)

    transpose_proj_x = torch.transpose(proj_x, 0, 1)
    exp_transpose_proj_x = torch.unsqueeze(transpose_proj_x, 2)
    exp_transpose_proj_x = exp_transpose_proj_x.contiguous()

    # Squared pairwise distances (dim x N x N)
    # The squared pairwise distances within each 1-D projection hence the number
    # of N x N matrices is dim
    # Need to set compute mode since the default compute mode can sometimes
    # mess up gradient calculations and give nan gradients.
    squared_pairwise_distances = torch.cdist(exp_transpose_proj_x, exp_transpose_proj_x,
        compute_mode='donot_use_mm_for_euclid_dist') ** 2

    # median squared distances (dim), one for each projection direction
    median_squared_distances = torch.median(
        torch.flatten(squared_pairwise_distances, start_dim=1, end_dim=2),
        dim=1)[0]

    # Kernel matrix (dim x N x N)
    K = torch.exp(- squared_pairwise_distances / \
        median_squared_distances.unsqueeze(1).unsqueeze(1))

    # Since the r directions are just the one-hot basis vectors, the matrix
    # s_p^r is just the same as Sqx
    term1 = Sqx.transpose(0,1).unsqueeze(2) * K * Sqx.transpose(0,1).unsqueeze(1)

    diag_g = g.diag()
    term2 = diag_g.unsqueeze(1).unsqueeze(2) * \
        Sqx.transpose(0,1).unsqueeze(1) * \
        (-2.0 / median_squared_distances.unsqueeze(1).unsqueeze(2)) * \
        (proj_x.transpose(0,1).unsqueeze(2) - proj_x.transpose(0,1).unsqueeze(1)) * \
        K

    term3 = diag_g.unsqueeze(1).unsqueeze(2) * \
        Sqx.transpose(0,1).unsqueeze(2) * \
        (2.0 / median_squared_distances.unsqueeze(1).unsqueeze(2)) * \
        (proj_x.transpose(0,1).unsqueeze(2) - proj_x.transpose(0,1).unsqueeze(1)) * \
        K

    term4 = diag_g.unsqueeze(1).unsqueeze(2) ** 2 * \
        K * \
        (
            (2.0 / median_squared_distances.unsqueeze(1).unsqueeze(2)) - \
            (4.0 / median_squared_distances.unsqueeze(1).unsqueeze(2) ** 2) * \
            (proj_x.transpose(0,1).unsqueeze(2) - proj_x.transpose(0,1).unsqueeze(1)) ** 2 \
        )

    h_prg = term1 + term2 + term3 + term4

    # Subtract off diagonals for U-statistic
    h_prg_minus_diag = h_prg - \
        torch.diag_embed(torch.diagonal(h_prg, dim1=-2, dim2=-1))

    sksd = (1.0 / (N * (N-1))) * torch.sum(h_prg_minus_diag)

    return sksd

def blockSKSD(x, Sqx, g, num_blocks, num_median=None, input_median=None):
    N, dim = x.shape
    block_step = math.floor(N/num_blocks)

    diag_g = g.diag()

    # Project each sample in each of the g directions
    proj_x = torch.matmul(x, g.transpose(0,1)) # (N x dim)

    transpose_proj_x = torch.transpose(proj_x, 0, 1)
    exp_transpose_proj_x = torch.unsqueeze(transpose_proj_x, 2) # (dim x N x 1)
    exp_transpose_proj_x = exp_transpose_proj_x.contiguous()

    if num_median is None:
        median_squared_distances = input_median
    else:
        # Median estimation:
        squared_pairwise_distances = torch.cdist(
            exp_transpose_proj_x[:, 0:num_median, :],
            exp_transpose_proj_x[:, 0:num_median, :]) ** 2
        median_squared_distances = torch.median(
            torch.flatten(squared_pairwise_distances, start_dim=1, end_dim=2),
            dim=1)[0]

    culm_sum = 0
    for i in range(dim):
        for j in np.floor(np.linspace(0, N, num=num_blocks+1)[0:-1]).astype(int):
            for k in np.floor(np.linspace(0, N, num=num_blocks+1)[0:-1]).astype(int):
                pass
                squared_pairwise_distances = torch.cdist(
                    exp_transpose_proj_x[i, j:j+block_step, :],
                    exp_transpose_proj_x[i, k:k+block_step, :]) ** 2 # (block_step x block_step)
                K = torch.exp(- squared_pairwise_distances / \
                    median_squared_distances[i])
                term1 = Sqx[j:j+block_step, i].unsqueeze(1) * \
                    K * \
                    Sqx[k:k+block_step, i].unsqueeze(0)
                term2 = diag_g[i] * Sqx[k:k+block_step, i].unsqueeze(0) * \
                    (-2.0 / median_squared_distances[i]) * \
                    (proj_x[j:j+block_step, i].unsqueeze(1) - \
                        proj_x[k:k+block_step, i].unsqueeze(0)) * \
                    K
                term3 = diag_g[i] * Sqx[j:j+block_step, i].unsqueeze(1) * \
                    (2.0 / median_squared_distances[i]) * \
                    (proj_x[j:j+block_step, i].unsqueeze(1) - \
                        proj_x[k:k+block_step, i].unsqueeze(0)) * \
                    K
                term4 = diag_g[i] ** 2 * K * \
                    (
                        (2.0 / median_squared_distances[i]) - \
                        (4.0 / median_squared_distances[i]**2) * \
                        (proj_x[j:j+block_step, i].unsqueeze(1) - \
                            proj_x[k:k+block_step, i].unsqueeze(0)) ** 2
                    )
                h = term1 + term2 + term3 + term4

                if j == k:
                    h = h - torch.diag(h.diag())

                culm_sum = culm_sum + torch.sum(h)

    sksd = (1.0 / (N * (N-1))) * culm_sum

    return sksd

def save_git_hash():
    sparse_hmc_git_hash = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=hydra.utils.get_original_cwd()).decode("utf-8")
    nf_git_hash = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=hydra.utils.get_original_cwd()+"/../normalizing-flows").decode("utf-8")

    if os.path.exists(hydra.utils.get_original_cwd() + "/../NUTS"):
        nuts_git_hash = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=hydra.utils.get_original_cwd()+"/../NUTS").decode("utf-8")
        with open('nuts_git_hash.txt', 'w') as f:
            f.write(nuts_git_hash)


    with open('sparse_hmc_git_hash.txt', 'w') as f:
        f.write(sparse_hmc_git_hash)
    with open('nf_git_hash.txt', 'w') as f:
        f.write(nf_git_hash)

def load_dataframe(experiment_path, hpam_names, metric_names):
    # For loading experiment data
    # experiment path e.g. ../outputs/2020-12-21_19-44-37_highdim_lr_sweep
    # metric_names e.g. ['w_samples, 'log_likelihoods']
    folders = glob.glob(experiment_path + r'/*/')
    if len(folders) == 0:
        print("No folders found!")
    df = pandas.DataFrame(columns=hpam_names + metric_names)
    for folder in folders:
        config_name = folder + '.hydra/config.yaml'
        with open(config_name) as f:
            config = yaml.load(f)
        hpam_vals = pandas.Series(
            [config[name] for name in hpam_names],
            index=hpam_names)
        metric_vals = pandas.Series(
            [np.load(folder + name + '.npy') for name in metric_names],
            index=metric_names)
        # print(hpam_vals)
        combined_series = hpam_vals.append(metric_vals)
        df = df.append(combined_series, ignore_index=True)
    return df

def calculate_log_likelihood(w_samples, X_star, y_star, sigma_0):
    m = w_samples.shape[0]

    const_factor = -np.log(m) - (X_star.shape[0]/2) * np.log(2 * np.pi * sigma_0**2)
    y_preds = w_samples @ X_star.T
    errors = (np.expand_dims(y_star, axis=0) - y_preds)**2
    sum_errors = np.sum(errors, axis=1)
    log_sum_errors = logsumexp((-1/(2*sigma_0**2)) * sum_errors)

    log_likelihood = const_factor + log_sum_errors
    return log_likelihood

