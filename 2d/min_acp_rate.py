from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.core import getval
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
    mu0 = np.zeros((1,2))
    log_sigma0 = np.zeros((1,2))
    return np.concatenate((mu0, log_sigma0), 0)

def leapfrog(z, r, eps, log_v_r, dlogP):   
    for i in range(5):
        r_half = r - eps / 2.0 * -getval(dlogP(z)) 
        z = z + eps * r_half / np.exp(log_v_r)
        r = r_half - eps / 2.0 * -getval(dlogP(z)) 
    return z, r

def generate_samples_HMC(eps0, params, n = 100):  

    
    mu0 = getval(params[-2,:])
    mu0 = np.ones((n,1)) * mu0
    log_sigma0 = getval(params[-1,:])
    sigma0 = np.exp(log_sigma0)
    
    eps = sigma0 * eps0
    
    z = rs.randn(n, params.shape[ 1 ]) * (np.ones((n,1)) * sigma0 ) + mu0 

    acp = np.zeros(n)
    for j in range(L):
        r = rs.randn(n, params.shape[ 1 ]) * np.exp(0.5 * 0)
        z_new, r_new = leapfrog(z, r, eps, 0.0, dlogP)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) -0.5 * np.sum(r_new**2 /  np.exp(0), 1) + \
            0.5 * np.sum(r**2 /  np.exp(0), 1)))
        accepted = rs.rand(n) < p_acceptance  # n-dimensional, ie. 100-dimensional
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[ 1 ], 1)))
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        acp += accepted
    min_acp = np.min(acp/L)

    return z,min_acp


def evaluate_objective(params): 
    N = 100
                                  
    sigma0 = np.exp(params[-1,:])
    mu0 = params[-2,:]

    randn0 = rs.randn(N,params.shape[1])
    samples0 = np.ones((N,1)) * mu0 + randn0* (np.ones((N,1)) * sigma0) 
    
    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-1,0] + params[-1,1]

    return -elbo0 

def adam(evaluate_objective, params, eps0):
    print("    Step       |     objective      ")
    def print_perf(epoch, params,eps0):
        objective = evaluate_objective(params)
        #print("{0:15}|{1:15}|{}".format(epoch, -objective, eps0))
        print("{}|{}|{}".format(epoch, -objective, eps0))
    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.02
    t = 0
    grad_objective = grad(evaluate_objective)
    epochs = 500
                                  
    
    #start = time.time()
    for epoch in range(epochs):
        #if epoch + 1 == 100:
        #    end = time.time()
        #    print("time: {}".format(end-start))
        t += 1
        print_perf(epoch, params,eps0)
                                  
        _, min_acp = generate_samples_HMC(eps0, params)   
                                  
        grad_params= grad_objective(params)  
        m1 = beta1 * m1 + (1 - beta1) * grad_params
        m2 = beta2 * m2 + (1 - beta2) * grad_params**2
        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta2**t)
        
        params = params - alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)   #alpha is step size of adam
        
        if min_acp<0.25:
            eps0 *= 0.995
        else:
            eps0 *= 1.005
        
    return params, eps0

if __name__ == "__main__":
    rs = npr.RandomState(0)
    L = 30
    logP = logP_gauss     # can specify other targets included in targets.py
    dlogP = dlogP_gauss
    eps0 = 1.0   # initialized to be 1.0 for correlated gaussian and laplace, 0.05 for dual moon, 0.1 for gaussian mixture and wave2, 0.01 for wave1 and 0.0002 for wave3 

    params = init_random_params(L)
    
    params,eps0 = adam(evaluate_objective, params, eps0)     
    exp_params = np.exp(params)
    exp_params[-2,:] = np.log(exp_params[-2,:])
    
    print("mu and sigma:{}".format(exp_params))
    print("eps0:{}".format(eps0))
    z, _ = generate_samples_HMC(eps0, params, 100000)

    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))   
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))
    
    z1=z[:,0]
    z2=z[:,1]
    plt.hist2d(z1, z2, bins=(300, 300))
    #plt.xlim(-4,4)
    #plt.ylim(-4,4)
    plt.show()
