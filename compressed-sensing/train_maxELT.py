#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import normflow as nf
from tqdm import tqdm
import hydra
import subprocess

from core.target import HorseshoeTarget
from core.utils import SKSD
from core.models import HMC


@hydra.main(config_path='conf', config_name="maxELT")
def main(cfg):

    print("Training HMC with maxELT")

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

    initial_dist_samples, _ = initial_dist.sample(cfg.num_samples_estimate_initial_mean)
    initial_dist_mean = torch.mean(initial_dist_samples, dim=0).detach()

    if cfg.log_stepsize_const is None:
        print("Initializing with a range of stepsizes")
        hmc = HMC(target, 2*d, initial_dist,
            num_hmc_steps, num_leapfrog_steps,
            cfg.log_stepsize_min, cfg.log_stepsize_max,
            cfg.log_mass_min, cfg.log_mass_max,
            cfg.train_scale, initial_dist_mean).double()
    else:
        print("Initializing with const stepsize and mass")
        hmc = HMC(target, 2*d, initial_dist,
            num_hmc_steps, num_leapfrog_steps,
            cfg.log_stepsize_const, cfg.log_stepsize_const,
            cfg.log_mass_const, cfg.log_mass_const,
            cfg.train_scale, initial_dist_mean).double()


    iters = cfg.num_iters
    hmc_optim = torch.optim.Adam(hmc.get_hmc_params(), lr=cfg.hmc_lr)
    if cfg.train_scale:
        print("Training with SKSD")
        scale_optim = torch.optim.Adam([hmc.scale], lr=cfg.hmc_scale_lr)
        g_optim = torch.optim.Adam([hmc.raw_g], lr=cfg.hmc_g_lr)
    else:
        print("Training without SKSD")

    losses = np.zeros((iters,))
    log_step_sizes = np.zeros((iters, num_hmc_steps, 2*d))
    log_masses = np.zeros((iters, num_hmc_steps, 2*d))
    acc_probs = np.zeros((iters,))

    torch.save(hmc.state_dict(), 'pre_training_hmc_model.pt')


    print("\n Training HMC params")
    for i in tqdm(range(iters)):
        hmc_optim.zero_grad()

        samples, ap = hmc.forward(cfg.batch_size)
        loss = - torch.mean(target.log_prob(samples))
        loss.backward(retain_graph=cfg.train_scale)
        hmc_optim.step()

        if cfg.train_scale:
            scale_optim.zero_grad()
            g_optim.zero_grad()
            gradlogp = hmc.flows[0].gradlogP(samples)
            sksd = SKSD(samples, gradlogp, hmc.get_g())
            sksd.backward()
            scale_optim.step()
            hmc.raw_g.grad = -hmc.raw_g.grad # Since we want to max sksd wrt g
            g_optim.step()

        losses[i] = loss.detach().numpy()
        log_step_sizes[i, :, :], log_masses[i, :, :] = hmc.get_np_params()
        acc_probs[i] = np.median(np.array(ap))

        if (i+1)%cfg.save_interval == 0:
            torch.save(hmc.state_dict(), 'hmc_model_{}.pt'.format(i+1))
            np.save('losses.npy', losses)
            np.save('log_step_sizes.npy', log_step_sizes)
            np.save('log_masses.npy', log_masses)
            np.save('acc_probs.npy', acc_probs)

    torch.save(hmc.state_dict(), 'final_hmc_model.pt')
    np.save('losses.npy', losses)
    np.save('log_step_sizes.npy', log_step_sizes)
    np.save('log_masses.npy', log_masses)
    np.save('acc_probs.npy', acc_probs)

if __name__ == "__main__":
    main()
