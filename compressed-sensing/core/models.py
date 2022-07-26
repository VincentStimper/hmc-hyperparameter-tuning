import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../../normalizing-flows')
import normflows as nf

class CustomHamiltonianMonteCarlo(nf.flows.Flow):
    """
    Custom version of HMC flow layer with additional logging
    """
    def __init__(self, target, steps, log_step_size, log_mass):
        """
        Constructor
        :param target: The stationary distribution of this Markov transition. Should be logp
        :param steps: The number of leapfrog steps
        :param log_step_size: The log step size used in the leapfrog integrator. shape (dim)
        :param log_mass: The log_mass determining the variance of the momentum samples. shape (dim)
        """
        super().__init__()
        self.target = target
        self.steps = steps
        self.register_parameter('log_step_size', torch.nn.Parameter(log_step_size))
        self.register_parameter('log_mass', torch.nn.Parameter(log_mass))

    def forward(self, z):
        # Draw momentum
        p = torch.randn_like(z) * torch.exp(0.5 * self.log_mass)

        # leapfrog
        z_new = z.clone()
        p_new = p.clone()
        step_size = torch.exp(self.log_step_size)
        for i in range(self.steps):
            p_half = p_new - (step_size/2.0) * -self.gradlogP(z_new)
            z_new = z_new + step_size * (p_half/torch.exp(self.log_mass))
            p_new = p_half - (step_size/2.0) * -self.gradlogP(z_new)

        # Metropolis Hastings correction
        probabilities = torch.exp(
            self.target.log_prob(z_new) - self.target.log_prob(z) - \
            0.5 * torch.sum(p_new ** 2 / torch.exp(self.log_mass), 1) + \
            0.5 * torch.sum(p ** 2 / torch.exp(self.log_mass), 1))
        probabilities = torch.clamp(probabilities, 0, 1)
        uniforms = torch.rand_like(probabilities)
        mask = uniforms < probabilities
        z_out = torch.where(mask.unsqueeze(1), z_new, z)

        return z_out, self.target.log_prob(z) - self.target.log_prob(z_out), \
            torch.mean(probabilities)

    def inverse(self, z):
        return self.forward(z)

    def gradlogP(self, z):
        z_ = z.detach().requires_grad_()
        logp = self.target.log_prob(z_)
        return torch.autograd.grad(logp, z_,
            grad_outputs=torch.ones_like(logp))[0]



class HMC(nn.Module):
    def __init__(self, target, hmc_dim, initial_dist,
        num_hmc_steps, num_leapfrog_steps,
        log_stepsize_min, log_stepsize_max,
        log_mass_min, log_mass_max,
        apply_scaling=False, initial_dist_mean=None):
        super().__init__()
        raw_flows = []
        for n in range(num_hmc_steps):
            raw_flows.append(CustomHamiltonianMonteCarlo(
                target=target,
                steps=num_leapfrog_steps,
                log_step_size=torch.rand(hmc_dim) * (log_stepsize_max - log_stepsize_min) + log_stepsize_min,
                log_mass=torch.rand(hmc_dim) * (log_mass_max - log_mass_min) + log_mass_min
            ))
        self.flows = nn.ModuleList(raw_flows)
        self.initial_dist = initial_dist
        self.apply_scaling = apply_scaling
        if apply_scaling:
            self.register_buffer('initial_dist_mean', initial_dist_mean)
            self.scale = nn.Parameter(torch.tensor(1.0))
        self.raw_g = nn.Parameter(torch.eye(hmc_dim))

        self.hmc_dim = hmc_dim

    def forward(self, num_samples):
        z, _ = self.initial_dist.sample(num_samples)
        if self.apply_scaling:
            z = (z - self.initial_dist_mean) * self.scale + self.initial_dist_mean
        acc_probs = []
        for flow in self.flows:
            z_p, _, mean_prob = flow.forward(z)
            z = z_p
            if not torch.isinf(mean_prob):
                acc_probs.append(mean_prob.detach().numpy())
        return z, acc_probs

    def get_np_params(self):
        log_step_sizes = np.zeros((len(self.flows), self.hmc_dim))
        log_masses = np.zeros((len(self.flows), self.hmc_dim))
        for i, flow in enumerate(self.flows):
            log_step_sizes[i, :] = flow.log_step_size.detach().numpy()
            log_masses[i, :] = flow.log_mass.detach().numpy()
        return log_step_sizes, log_masses

    def get_hmc_params(self):
        hmc_parameters = []
        for flow in self.flows:
            for p in flow.parameters():
                hmc_parameters.append(p)
        return hmc_parameters

    def get_g(self):
        normalizer = torch.sqrt(torch.sum(self.raw_g**2, dim=1)).reshape(self.hmc_dim, 1)
        g = self.raw_g / normalizer

        return g