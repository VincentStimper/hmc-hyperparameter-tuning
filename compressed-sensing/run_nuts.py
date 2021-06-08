from __future__ import absolute_import
from __future__ import print_function
import numpy as npy
import random as random
import matplotlib.pyplot as plt
import time
from nuts import nuts6, NutsSampler_fn_wrapper
import hydra

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad
from autograd.core import getval
from autograd.scipy.special import logsumexp

import arviz as az


@hydra.main(config_path='conf', config_name="nuts")
def main(cfg):

    class HorseshoeTarget():
        def __init__(self, X, y, tau, sigma_0, gamma=1):
            self.X = X
            self.y = y
        
            self.sigma_0 = sigma_0
            self.tau = tau
        
            self.gamma = gamma
        
            self.d = X.shape[1]
        
        def log_prob(self, z):
            w = z[0:self.d] 
            lamb = z[self.d:]
        
            var = (self.sigma_0**2) / self.gamma
        
            log_p_y = np.sum(-0.5 * np.log(2 * np.pi * var) - \
                0.5 * ((self.y - w @ self.X.T)**2 / var))
        
            log_p_z = np.sum(- np.log(1 + lamb**2) - \
                0.5 * np.log(self.tau**2 * lamb**2) - \
                0.5 * (w**2 / (self.tau**2 * lamb**2)))
        
            return log_p_y + log_p_z

    def logP(z):
        return target.log_prob(z)
        
    def dlogP(z):
        egrad = elementwise_grad(logP)
        return egrad(z)

    def nuts_target(W_lamb):
        logp = logP(W_lamb)
        dlogp = dlogP(W_lamb)
        return logp, dlogp 
        
    def calculate_log_likelihood(w_samples, X_star, y_star, sigma_0):
        m = w_samples.shape[0]

        const_factor = -np.log(m) - (X_star.shape[0]/2) * np.log(2 * np.pi * sigma_0**2)
        y_preds = w_samples @ X_star.T
        errors = (np.expand_dims(y_star, axis=0) - y_preds)**2
        sum_errors = np.sum(errors, axis=1)
        log_sum_errors = logsumexp((-1/(2*sigma_0**2)) * sum_errors)
        
        log_likelihood = const_factor + log_sum_errors
        return log_likelihood



    data_path = hydra.utils.to_absolute_path(cfg.path_to_observed_data) + '/'
    print("Loading observed data from", data_path)
    X_train = np.load(data_path + 'X_train.npy')
    X_test = np.load(data_path + 'X_test.npy')
    y_train = np.load(data_path + 'y_train.npy')
    y_test = np.load(data_path + 'y_test.npy')
    target = HorseshoeTarget(X_train, y_train, tau=cfg.tau, sigma_0=cfg.sigma_0)
    initial_dist_path = hydra.utils.to_absolute_path(cfg.path_to_initial_dist) + '/'
    init_mean = np.load(initial_dist_path + 'means.npy')[-1,:]
    init_log_scale = np.load(initial_dist_path + 'log_scales.npy')[-1,:]

    for i in range(20):
        init_sample = init_mean + npr.randn(128)* np.exp(init_log_scale)
        samples, _, epsilon = nuts6(nuts_target, cfg.num_samples, cfg.burn_in, init_sample, delta=cfg.delta)
        print('epsilon = {}'.format(epsilon))
        print('(nuts) test set log-lik: {}'.format(calculate_log_likelihood(samples[:,:samples.shape[1]//2],X_test,y_test,0.005)))
        np.savetxt('nuts_samples'+str(i+1)+'.csv', samples, delimiter=',')
        

    test_log_lik_list = []
    essmin_list = []
    essmean_list = []
    for i in range(20):
        samples=np.genfromtxt('nuts_samples'+str(i+1)+'.csv',delimiter=',')
        idata = az.convert_to_inference_data(np.expand_dims(samples[:,:64], 0))
        ess = az.ess(idata, method='bulk')
        essmin_list.append(np.min(ess.x))
        essmean_list.append(np.mean(ess.x))
        test_log_lik_list.append(calculate_log_likelihood(samples[:,:samples.shape[1]//2],X_test,y_test,0.005))

    res = np.array(test_log_lik_list)
    print('test log-lik: {}'.format(res))
    print('mean = {}'.format(np.mean(res)))
    print('std = {}'.format(np.std(res)))
    print('ESS (min): {}'.format(essmin_list))
    print('ESS (mean): {}'.format(essmean_list))

if __name__ == "__main__":
    main()
