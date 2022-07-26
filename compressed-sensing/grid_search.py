#%%
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

import pickle


@hydra.main(config_path='conf', config_name="grid_search")
def main(cfg):

    print("Grid searching for HMC params")

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
    X_train, y_train, X_validation, \
        y_validation, true_w, true_lamb = np.load(data_path + 'X_train.npy'),\
                                          np.load(data_path + 'y_train.npy'),\
                                          np.load(data_path + 'X_validation.npy'),\
                                          np.load(data_path + 'y_validation.npy'),\
                                          np.load(data_path + 'w.npy'),\
                                          np.load(data_path + 'lambda.npy')

    X_train, y_train, true_w, true_lamb = torch.from_numpy(X_train).double(),\
                                          torch.from_numpy(y_train).double(),\
                                          torch.from_numpy(true_w).double(),\
                                          torch.from_numpy(true_lamb).double()

    X_star = X_validation
    y_star = y_validation

    target = HorseshoeTarget(X_train, y_train, tau, sigma_0)

    initial_dist = nf.core.NormalizingFlow(
        q0 = nf.distributions.DiagGaussian(
            shape=2*d
        ),
        flows=None,
        p=target
    ).double()
    initial_dist_path = hydra.utils.to_absolute_path(cfg.path_to_initial_dist_model)
    print("Loading initial dist from", initial_dist_path)
    initial_dist.load_state_dict(torch.load(initial_dist_path))

    log_stepsizes = np.linspace(cfg.log_stepsize_min, cfg.log_stepsize_max, cfg.fidelity)
    log_masses = np.linspace(cfg.log_mass_min, cfg.log_mass_max, cfg.fidelity)

    print("log stepsizes", log_stepsizes)
    print("log masses", log_masses)

    grid_vals = {}
    best_log_stepsize = 0
    best_log_mass = 0
    best_avg_ll = -1e200

    for log_stepsize in log_stepsizes:
        for log_mass in log_masses:

            hmc = HMC(target, 2*d, initial_dist,
                num_hmc_steps, num_leapfrog_steps,
                log_stepsize, log_stepsize,
                log_mass, log_mass).double()

            samples, _ = hmc.forward(cfg.num_ll_calc_repeats * cfg.num_ll_calc_samples)
            samples = samples.detach().numpy()
            samples = samples[:, 0:d]
            log_likelihood_vals = np.zeros((cfg.num_ll_calc_repeats))

            for i in range(cfg.num_ll_calc_repeats):
                sub_samples = samples[i*cfg.num_ll_calc_samples:(i+1)*cfg.num_ll_calc_samples, :]
                log_likelihood_vals[i] = calculate_log_likelihood(
                    sub_samples, X_star, y_star, sigma_0)

            avg_ll = np.mean(log_likelihood_vals)
            grid_vals[(log_stepsize, log_mass)] = avg_ll
            if avg_ll > best_avg_ll:
                best_avg_ll = avg_ll
                best_log_stepsize = log_stepsize
                best_log_mass = log_mass

    print("Best log stepsize and mass", best_log_stepsize, best_log_mass)
    print("With top avg ll", best_avg_ll)
    hmc = HMC(target, 2*d, initial_dist,
        num_hmc_steps, num_leapfrog_steps,
        best_log_stepsize, best_log_stepsize,
        best_log_mass, best_log_mass).double()
    torch.save(hmc.state_dict(), 'best_hmc_model.pt')

    f = open("grid_vals.pkl","wb")
    pickle.dump(grid_vals,f)
    f.close()


if __name__ == "__main__":
    main()

# %%
