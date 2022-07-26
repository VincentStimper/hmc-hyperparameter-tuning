import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import normflows as nf
from tqdm import tqdm
import hydra
import subprocess

from core.target import HorseshoeTarget
from core.utils import calculate_log_likelihood
from core.models import HMC

@hydra.main(config_path='conf', config_name="evaluate_model")
def main(cfg):

    print("Evaluating model on {} data".format('validation' if cfg.is_validation else 'test'))
    d = cfg.w_dim
    sigma_0 = cfg.sigma_0
    tau = cfg.tau
    num_hmc_steps = cfg.num_hmc_steps
    num_leapfrog_steps = cfg.num_leapfrog_steps

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = hydra.utils.to_absolute_path(cfg.path_to_observed_data) + "/"
    print("Loading observed data from", data_path)
    X_train, y_train, \
        X_validation, y_validation, \
        X_test, y_test, true_w, true_lamb = \
                              np.load(data_path + 'X_train.npy'),\
                              np.load(data_path + 'y_train.npy'),\
                              np.load(data_path + 'X_validation.npy'),\
                              np.load(data_path + 'y_validation.npy'),\
                              np.load(data_path + 'X_test.npy'),\
                              np.load(data_path + 'y_test.npy'),\
                              np.load(data_path + 'w.npy'),\
                              np.load(data_path + 'lambda.npy')

    assert cfg.is_test != cfg.is_validation
    if cfg.is_validation:
        X_star = X_validation
        y_star = y_validation
    elif cfg.is_test:
        X_star = X_test
        y_star = y_test


    X_train, y_train, true_w, true_lamb = \
            torch.from_numpy(X_train).double(),\
            torch.from_numpy(y_train).double(),\
            torch.from_numpy(true_w).double(),\
            torch.from_numpy(true_lamb).double(),\
            
    target = HorseshoeTarget(X_train, y_train, tau, sigma_0)

    if cfg.test_initial_dist_model:
        print("Loading an initial dist model") 

        initial_dist = nf.core.NormalizingFlow(
            q0 = nf.distributions.DiagGaussian(
                shape=2*d
            ),
            flows=None,
            p=target
        ).double()
        initial_dist_path = hydra.utils.to_absolute_path(
            cfg.path_to_initial_dist_model)
        print("Loading initial dist from", initial_dist_path)
        initial_dist.load_state_dict(torch.load(initial_dist_path))

        print("Sampling initial distribution")
        samples, _ = initial_dist.sample(cfg.num_repeats * cfg.num_w_samples)

    else:
        print("Loading a HMC model")
        initial_dist = nf.core.NormalizingFlow(
            q0 = nf.distributions.DiagGaussian(
                shape=2*d
            ),
            flows=None,
            p=target
        ).double()

        hmc = HMC(target, 2*d, initial_dist,
            num_hmc_steps, num_leapfrog_steps,
            0, 1, 0, 1, cfg.hmc_model_includes_scale,
            torch.zeros(2*d)).double()

        hmc_path = hydra.utils.to_absolute_path(
            cfg.path_to_hmc_model)
        print("Loading HMC model from", hmc_path)
        hmc.load_state_dict(torch.load(hmc_path))

        print("Sampling hmc model")
        samples, _ = hmc.forward(cfg.num_repeats * cfg.num_w_samples)

    samples = samples.detach().numpy()
    samples = samples[:, 0:d]

    log_likelihood_vals = np.zeros((cfg.num_repeats))

    print("Calculating log likelihood values")
    for i in range(cfg.num_repeats):
        sub_samples = samples[i*cfg.num_w_samples:(i+1)*cfg.num_w_samples, :]
        log_likelihood_vals[i] = calculate_log_likelihood(
            sub_samples, X_star, y_star, sigma_0)

    np.save('log_likelihood_vals.npy', log_likelihood_vals)
    np.save('w_samples.npy', samples)

if __name__ == "__main__":
    main()