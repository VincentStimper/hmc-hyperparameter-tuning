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
from core.utils import SKSD
from core.models import HMC


@hydra.main(config_path='conf', config_name="acc_prob")
def main(cfg):

    print("Training HMC with acceptance probability")

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
    X_train, y_train, true_w, true_lamb = np.load(data_path + 'X_train.npy'),\
                                          np.load(data_path + 'y_train.npy'),\
                                          np.load(data_path + 'w.npy'),\
                                          np.load(data_path + 'lambda.npy')

    X_train, y_train, true_w, true_lamb = torch.from_numpy(X_train).double(),\
                                            torch.from_numpy(y_train).double(),\
                                            torch.from_numpy(true_w).double(),\
                                            torch.from_numpy(true_lamb).double()

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


    hmc = HMC(target, 2*d, initial_dist,
        num_hmc_steps, num_leapfrog_steps,
        cfg.start_log_stepsize, cfg.start_log_stepsize,
        cfg.start_log_mass, cfg.start_log_mass).double()

    iters = cfg.num_iters

    log_step_sizes = np.zeros((iters,))
    acc_probs = np.zeros((iters,))

    torch.save(hmc.state_dict(), 'pre_training_hmc_model.pt')

    log_step_size = cfg.start_log_stepsize
    lr = cfg.learning_rate
    kappa = cfg.learning_rate_decay
    target_ap = cfg.target_acceptance_prob

    print("\n Training HMC params")
    for i in tqdm(range(iters)):

        samples, ap = hmc.forward(cfg.batch_size)
        ap = np.array(ap)
        mean_ap = np.mean(ap)

        a_i = lr * (i+1)**(-kappa)

        log_step_size = log_step_size - a_i * (target_ap - mean_ap)

        state_dict = hmc.state_dict()
        for k in range(num_hmc_steps):
            state_dict['flows.{}.log_step_size'.format(k)] = \
                log_step_size * torch.ones(2*d)
        hmc.load_state_dict(state_dict)

        log_step_sizes[i] = log_step_size
        acc_probs[i] = mean_ap

    torch.save(hmc.state_dict(), 'final_hmc_model.pt')
    np.save('log_step_sizes.npy', log_step_sizes)
    np.save('acc_probs.npy', acc_probs)

if __name__ == "__main__":
    main()

# %%
