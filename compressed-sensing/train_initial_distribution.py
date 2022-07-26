import numpy as np
import torch
import torch.nn as nn
import sys
import normflows as nf
from tqdm import tqdm
import hydra
import subprocess

from core.target import HorseshoeTarget

@hydra.main(config_path='conf', config_name="initial_distribution")
def main(cfg):

    d = cfg.w_dim
    sigma_0 = cfg.sigma_0
    tau = cfg.tau

    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = hydra.utils.to_absolute_path(cfg.path_to_observed_data) + "/"
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

    #%%
    # Train initial distribution
    inner_iters = cfg.inner_iters
    anneal_steps = cfg.anneal_steps
    initial_dist_losses = np.zeros((anneal_steps * inner_iters))
    initial_dist_log_scales = np.zeros((anneal_steps * inner_iters, 2*d))
    initial_dist_means = np.zeros((anneal_steps * inner_iters, 2*d))
    learning_rates = [1 * 10 **(-x) for x in np.linspace(1, 3, anneal_steps)]
    for i in tqdm(range(1, anneal_steps+1)):
        target.gamma = 0.8 ** (anneal_steps - i)
        initial_dist_optimizer = torch.optim.Adam(initial_dist.parameters(),
            lr=learning_rates[i-1])

        for k in range(inner_iters):
            initial_dist_optimizer.zero_grad()

            if cfg.alpha == 0:
                loss = initial_dist.reverse_kld(num_samples=cfg.batch_size)
            else:
                loss = initial_dist.reverse_alpha_div(
                    num_samples=cfg.batch_size,
                    alpha=cfg.alpha)

            loss.backward()
            initial_dist_optimizer.step()

            initial_dist_losses[(i-1)*inner_iters + k] = loss.detach().numpy()
            initial_dist_log_scales[(i-1)*inner_iters + k, :] = initial_dist.q0.log_scale.detach().numpy()
            initial_dist_means[(i-1)*inner_iters + k, :] = initial_dist.q0.loc.detach().numpy()

        if i % cfg.save_interval == 0:
            np.save('losses.npy', initial_dist_losses)
            np.save('log_scales.npy', initial_dist_log_scales)
            np.save('means.npy', initial_dist_means)
            torch.save(initial_dist.state_dict(), 'initial_dist_{}.pt'.format(i))

    # Final saving
    np.save('losses.npy', initial_dist_losses)
    np.save('log_scales.npy', initial_dist_log_scales)
    np.save('means.npy', initial_dist_means)
    torch.save(initial_dist.state_dict(), 'initial_dist_final.pt')

if __name__ == "__main__":
    main()