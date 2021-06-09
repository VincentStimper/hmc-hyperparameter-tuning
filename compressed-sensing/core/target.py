import numpy as np
import torch

class HorseshoeTarget():
    def __init__(self, X, y, tau, sigma_0, gamma=1):
        """
        X (n, d)
        y (n,)

        gamma is an annealing parameter where for the likelihood instead of
        using N(y|x^Tw, sigma_0**2) use N(y|x^Tw, sigma_0**2/gamma) and slowly
        increase gamma to 1 to make a series of easy targets
        """

        self.X = X
        self.y = y

        self.sigma_0 = sigma_0
        self.tau = tau

        self.gamma = gamma

        self.d = X.shape[1]

    def log_prob(self, z):
        """
        Returns log p(w, \lambda | y, X) up to a constant
        z = {w, \lambda}
        z dim should be (N, 2d) where N is the number of parallel chains
        Output is (N,)
        z should be arranged such that the first d columns are w and the last d
        columns are \lambda
        """
        w = z[:, 0:self.d]
        lamb = z[:, self.d:]

        var = (self.sigma_0**2) / self.gamma

        log_p_y = torch.sum(-0.5 * np.log(2 * np.pi * var) - \
            0.5 * ((self.y - w @ self.X.T)**2 / var), dim=1)

        log_p_z = torch.sum(- torch.log(1 + lamb**2) - \
            0.5 * torch.log(self.tau**2 * lamb**2) - \
            0.5 * (w**2 / (self.tau**2 * lamb**2)), dim=1)

        return log_p_y + log_p_z