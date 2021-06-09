from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
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
    mu0 = np.zeros((1,2))
    log_sigma0 = np.zeros((1,2))
    return np.concatenate((mu0, log_sigma0), 0)

def leapfrog(z, r,  log_v_r, dlogP, eps=0.01):   
    for i in range(5):
        r_half = r - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
        z = z + eps * r_half / np.exp(log_v_r)
        r = r_half - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
    return z, r

def generate_samples(eps0,params,n=100):
    
    mu0 = params[0,:]
    mu0 = np.ones((n,1)) * mu0
    log_sigma0 = params[1,:]
    sigma0 = np.exp(log_sigma0)
    eps=sigma0*eps0
    
    z = rs.randn(n, params.shape[ 1 ]) * (np.ones((n,1)) * sigma0 ) + mu0 
    z0 = z
    acp = np.zeros(n)
    for j in range(L):
        r = rs.randn(n, params.shape[ 1 ])
        z_new, r_new = leapfrog(z, r, 0, dlogP, eps)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) -0.5 * np.sum(r_new**2 /  np.exp(0), 1) + \
            0.5 * np.sum(r**2 /  np.exp(0), 1)))
        accepted = rs.rand(n) < p_acceptance 
        accepted_tile = np.transpose(np.tile(accepted, (params.shape[ 1 ], 1)))
        z = z_new * accepted_tile + (1 - accepted_tile) * z
        acp += accepted
    mean_acp = np.mean(acp/L)
    return z, z0,mean_acp

def generate_samples_z0(eps0,z0,params,n=10):
    z=z0 
    z=np.tile(z,(1,n)).reshape((-1,2))
    log_sigma0 = params[1,:]
    sigma0 = np.exp(log_sigma0)
    eps = sigma0*eps0
    for j in range(L):
        r = rs.randn(z.shape[0], z0.shape[ 1 ])
        z_new, r_new = leapfrog(z, r, 0, dlogP, eps)
        p_acceptance = np.minimum(1, np.exp(logP(z_new) - logP(z) -0.5 * np.sum(r_new**2 /  np.exp(0), 1) + \
            0.5 * np.sum(r**2 /  np.exp(0), 1)))
        accepted = rs.rand(z.shape[0]) < p_acceptance 
        accepted = np.transpose(np.tile(accepted, (params.shape[ 1 ], 1)))
        z = z_new * accepted + (1 - accepted) * z
    return z 


def log_q(z, params):
    k, zdim = z.shape
    mu0 = params[0,:]
    mu0 = np.ones((k,1)) * mu0
    log_sigma0 = params[1,:]
    sigma0 = np.exp(log_sigma0)
    var = sigma0**2
    var_inv = np.expand_dims(1/var,1)*np.eye(2)
    det = np.prod(var)
    return np.log(1/ np.sqrt((2*np.pi)**zdim * det)) + np.diag((-0.5* np.matmul(np.matmul((z-mu0), var_inv), (z-mu0).T)))
    
def f(z, params):
    return logP(z) - log_q(z, params)
    
    
def evaluate_objective(params,eps0): 
    N = 100
    samples, samples0,_ = generate_samples(eps0,params)
    samples_gz0 = generate_samples_z0(eps0,samples0,params)
    w = getval(f(samples_gz0, params))
    
    w=np.reshape(w,(100,-1))
    w=np.mean(w,1)
    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-1,0] + params[-1,1]

    loss = -elbo0 -np.mean(log_q(getval(samples),params)) + np.mean(w* log_q(getval(samples0),params))
    return loss


def adam(evaluate_objective, params, eps0):
    print("    Step       |     objective      |       eps0")
    def print_perf(epoch, params):
        objective = evaluate_objective(params,eps0)
        print("{0:15}|{1:15}|{2:15}".format(epoch, -objective, eps0))
    m1 = 0
    m2 = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.05
    t = 0
    
    epochs = 200

    for epoch in range(epochs):
        t += 1
        print_perf(epoch, params)
        
        _,_, mean_acp = generate_samples(eps0, params)
        evaluate_objective_i = lambda params: evaluate_objective(params,eps0=eps0)
        grad_objective = grad(evaluate_objective_i)
        grad_params= grad_objective(params)  
        
        m1 = beta1 * m1 + (1 - beta1) * grad_params
        m2 = beta2 * m2 + (1 - beta2) * grad_params**2
        m1_hat = m1 / (1 - beta1**t)
        m2_hat = m2 / (1 - beta2**t)
        
        params = params - alpha * m1_hat / (np.sqrt(m2_hat) + epsilon)  

        #print("mean_acp = {}".format(mean_acp))
        if mean_acp<0.65:
            eps0 *= 0.95
        else:
            eps0 *= 1.05
    return params,eps0


if __name__ == '__main__':
    rs = npr.RandomState(0)
    L = 30
    logP = logP_gauss     # can specify other targets included in targets.py
    dlogP = dlogP_gauss
    eps0 = 1.0
    
    params = init_random_params(L)
    params , eps0= adam(evaluate_objective, params,eps0)     
    exp_params = np.exp(params)
    exp_params[0,:] = np.log(exp_params[0,:])
    print("mu and sigma: {}".format(exp_params))
    print("eps0: {}".format(eps0))
    
    z,_,_ = generate_samples(eps0,params, 100000)
    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))
    
    z1=z[:,0]
    z2=z[:,1]
    plt.hist2d(z1, z2, bins=(300, 300))
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.show()