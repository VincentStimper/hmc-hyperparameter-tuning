from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import getval
from autograd.scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform

from targets import *

import time

def KSD(z, Sqx,flag_U = False):
    
    # compute the rbf kernel
    K, dimZ = z.shape
    sq_dist = pdist(z)
    pdist_square = squareform(sq_dist)**2
    
    h_square = 1.0

    Kxy = np.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = np.dot(getval(Sqx), z.T) - np.tile(np.sum(getval(Sqx) * z, 1, keepdims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = Sqxdy.T
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
    
    # M is a (K, K) tensor
    M = (np.dot(getval(Sqx), getval(Sqx).T) + Sqxdy + dxSqy + dxdy) * Kxy

    # U-statistic
    if flag_U:
        M2 = M - np.diag(np.diag(M))
        return np.sum(M2) / (K * (K - 1))
    # V-statistic
    else:
        return np.sum(M) / (K * K )
    
def init_random_params(L):  
    eps = 0.01 + rs.rand(L,2) * 0.015
    log_eps = np.log(eps)
    log_v_r = np.zeros([ L, 2 ])
    mu0 = np.zeros((1,2))
    log_sigma0 = np.zeros((1,2))
    log_inflation = 0 * np.ones((1,2))
    return np.concatenate((log_eps, log_v_r, mu0, log_sigma0, log_inflation), 0)

def leapfrog(z, r, eps, log_v_r, dlogP):   
    for i in range(5):
        r_half = r - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
        z = z + eps * r_half / np.exp(log_v_r)
        r = r_half - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
    return z, r

def generate_samples_HMC(params, n = 100):  
    log_eps = params[ 0 : L, : ]
    log_v_r = params[ L : (2 * L), : ]
    
    mu0 = getval(params[-3,:])
    mu0 = np.ones((n,1)) * mu0
    log_sigma0 = getval(params[-2,:])
    sigma0 = np.exp(log_sigma0)
    
    log_inflation = getval(params[-1,:][0])
    inflation = np.exp(log_inflation)
    
    z = rs.randn(n, params.shape[ 1 ]) * (np.ones((n,1)) * (sigma0 * inflation)) + mu0 

    for j in range(L):
        r = rs.randn(n, params.shape[ 1 ]) * np.exp(0.5 * log_v_r[ j, : ])
        z_new, r_new = leapfrog(z, r, np.exp(log_eps[ j, : ]), log_v_r[ j, : ], dlogP)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) -0.5 * np.sum(r_new**2 /  np.exp(log_v_r[ j, : ]), 1) + \
            0.5 * np.sum(r**2 /  np.exp(log_v_r[ j, : ]), 1)))
        accepted = rs.rand(n) < p_acceptance
        accepted = np.transpose(np.tile(accepted, (params.shape[ 1 ], 1)))
        z = z_new * accepted + (1 - accepted) * z

    return z


def evaluate_objective(params): 
    N = 100
    samples_HMC = generate_samples_HMC(params, N)

    sigma0 = np.exp(params[-2,:])
    var0 = sigma0**2
    mu0 = params[-3,:]

    epsilon0 = rs.randn(N,params.shape[1])
    samples0 = np.ones((N,1)) * mu0 + epsilon0* (np.ones((N,1)) * sigma0) 
    
    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-2,0] + params[-2,1]
    return -np.mean(logP(samples_HMC)) - elbo0


def adam(evaluate_objective, params):
    print("    Step       |     objective      ")
    def print_perf(epoch, params):
        objective = evaluate_objective(params)
        print("{0:15}|{1:15}".format(epoch, -objective))
    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.05
    t = 0
    grad_objective = grad(evaluate_objective)
    epochs = 200
    
    #start = time.time()
    for epoch in range(epochs):
        #if epoch + 1 == 100:
        #    end = time.time()
        #    print("time: {}".format(end-start))
        t += 1
        print_perf(epoch, params)
        grad_params= grad_objective(params)  
        m1 = beta1 * m1 + (1 - beta1) * grad_params
        m2 = beta2 * m2 + (1 - beta2) * grad_params**2
        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta2**t)
        
        params = params - alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)   #alpha is step size of adam
        
    return params

if __name__ == "__main__":
    rs = npr.RandomState(0)
    L = 30
    logP = logP_gauss     # can specify other targets included in targets.py
    dlogP = dlogP_gauss
    
    params = init_random_params(L)
    params = adam(evaluate_objective, params)     
    exp_params = np.exp(params)
    exp_params[-3,:] = np.log(exp_params[-3,:])
    exp_params[-1,1] = None
    print("step_sizes, mu, sigma and inflation: {}".format(exp_params))
    
    z = generate_samples_HMC(params, 100000)
    
    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))   
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))
    
    z1=z[:,0]
    z2=z[:,1]
    plt.hist2d(z1, z2, bins=(300, 300))
    #plt.xlim(-4,4)
    #plt.ylim(-4,4)
    plt.show()