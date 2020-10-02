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


def np_median_heruistic_proj(sample1,sample2):
    '''
    Median Heuristic for projected samples
    '''
    # samples 1 is * x g x N x 1
    # samples 2 is * x g x N x 1

    G=np.sum(sample1*sample1,axis=-1) # * x num_g x N or r x g x N
    G_exp = np.expand_dims(G, axis=-2)  # * x num_g x 1 x N or * x r x g x 1 x N

    H=np.sum(sample2*sample2,axis=-1) # * x num_g x N or * x r x g x N
    H_exp=np.expand_dims(H, axis=-1) # * x numb_g x N x 1 or * x r x g x N x 1

    dist = G_exp + H_exp - 2*np.matmul(sample2,np.transpose(sample1,(0,2,1))) # * x G x N x N
    dist=dist[np.triu(np.ones(dist.shape))==1] .reshape((dist.shape[0],-1)) # g x (NN)
    
    M = dist.shape[1]
    if M % 2 == 1:
        med_ind = np.argsort(dist)[:,int((M-1)/2)]
        median_dist = np.array([dist[0,int(med_ind[0])], dist[1,int(med_ind[1])]])
    else:
        med_ind1 = np.argsort(dist)[:,int(M/2)]
        med_ind2 = np.argsort(dist)[:,int(M/2 - 1)]
        median_dist = 0.5* np.array([dist[0,int(med_ind1[0])] + dist[0,int(med_ind2[1])], dist[1,int(med_ind1[1])] + dist[1,int(med_ind2[1])]])

    #median_dist=np.median(getval(dist),axis=-1) # num_g or * x g
    return median_dist

def np_SE_kernel(sample1,sample2,**kwargs):
    '''
    Compute the square exponential kernel
    :param sample1: x
    :param sample2: y
    :param kwargs: kernel hyper-parameter: bandwidth
    :return:
    '''

    bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or * x g

    bandwidth_exp=np.expand_dims(np.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
    K = np.exp(-(sample1 - sample2) ** 2 / (bandwidth_exp ** 2+1e-9)) # g x sam1 x sam2
    return K

def np_d_SE_kernel(sample1,sample2,**kwargs):
    'The gradient of RBF kernel'
    K=kwargs['K'] # * x g x sam1 x sam2

    bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g

    bandwidth_exp=np.expand_dims(np.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
    d_K=K*(-1/(bandwidth_exp**2+1e-9)*2*(sample1-sample2)) # g x sam1 x sam2

    return d_K
def np_dd_SE_kernel(sample1,sample2,**kwargs):
    K=kwargs['K'] # * x g x sam1 x sam2

    bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g

    bandwidth_exp=np.expand_dims(np.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
    dd_K=K*(2/(bandwidth_exp**2+1e-9)-4/(bandwidth_exp**4+1e-9)*(sample1-sample2)**2)

    return dd_K # g x N x N


def np_compute_max_SKSD(samples1,samples2,score1,score2,kernel,d_kernel,dd_kernel,g,flag_U=True,bandwidth_scale=1):
    '''
    numpy version of maxSKSD with median heuristics
    :param samples1: samples from q with shape: N x dim
    :param samples2: samples from q with shape: N x dim
    :param score1: score of p for samples 1 with shape N x dim
    :param score2: score of p for samples 2 with shape N x dim
    :param kernel: kernel function (default: np_SE_kernel)
    :param d_kernel: derivative of kernel function (default: np_d_SE_kernel)
    :param dd_kernel: second derivative of kernel function (default: np_dd_SE_kernel)
    :param g: sliced direction with shape dim x dim
    :param flag_U: whether use U-statistic (True) or V-statistics (False)
    :param bandwidth_scale: coefficient for bandwidth (default:1)
    :return: KDSSD: discrepancy value; divergence:each component for KDSSD (used for debug or GOF Test)
    '''
    dim=samples1.shape[-1]
    r=np.eye(dim)

    kernel_hyper={}
    ##### Compute the median for each slice direction g
    if samples1.shape[0] > 500: # To reduce the sample number for median computation
        idx_crop = 500
    else:
        idx_crop = samples1.shape[0]

    g_cp_exp = np.expand_dims(g, 1)  # g x 1 x dim
    samples1_exp = np.expand_dims(samples1[0:idx_crop, :], 0)  # 1 x N x dim
    samples2_exp = np.expand_dims(samples2[0:idx_crop, :], 0)  # 1 x N x dim
    proj_samples1 = np.sum(samples1_exp * g_cp_exp, axis=-1, keepdims=True)  # g x N x 1
    proj_samples2 = np.sum(samples2_exp * g_cp_exp, axis=-1, keepdims=True)  # g x N x 1
    median_dist = np_median_heruistic_proj(proj_samples1, proj_samples2)  # g

    bandwidth_array = bandwidth_scale*2 * np.sqrt(0.5 * median_dist)
    kernel_hyper['bandwidth_array'] = bandwidth_array

    ##### Now compute the SKSD with slice direction g for each dimension
    # Compute Term1

    g_exp = g.reshape((g.shape[0], 1, g.shape[-1]))  # g x 1 x D
    samples1_crop_exp = np.expand_dims(samples1, axis=0)  # 1 x N x D
    samples2_crop_exp = np.expand_dims(samples2, axis=0)  # 1 x N x D
    proj_samples1_crop_exp = np.sum(samples1_crop_exp * g_exp, axis=-1)  # g x sam1
    proj_samples2_crop_exp = np.sum(samples2_crop_exp * g_exp, axis=-1)  # g x sam2

    r_exp = np.expand_dims(r, axis=1)  # r x 1 x dim
    proj_score1 = np.sum(r_exp * np.expand_dims(getval(score1), axis=0), axis=-1, keepdims=True)  # r x sam1 x 1
    proj_score2 = np.sum(r_exp * np.expand_dims(getval(score2), axis=0), axis=-1)  # r x sam2

    proj_score1_exp = proj_score1  # r x sam1 x 1
    proj_score2_exp = proj_score2.reshape((proj_score2.shape[0], 1, proj_score2.shape[-1]))  # r x 1 x sam2

    K = kernel(np.expand_dims(proj_samples1_crop_exp, axis=-1), np.expand_dims(proj_samples2_crop_exp, axis=-2),
               kernel_hyper=kernel_hyper)  # g x sam1 x sam 2
    if flag_U:
        d_K = np.diagonal(K, axis1=-1, axis2=-2).reshape((K.shape[0], K.shape[1], 1))  # g x sam1 x 1
        e_K = np.tile(np.eye(K.shape[-1]).reshape((1, K.shape[-2], K.shape[-1])),(K.shape[0],1,1))  # g x sam1 x sam2
        d_K = d_K * e_K  # g x 1 x sam1 x sam2
        Term1 = proj_score1_exp * (K - d_K) * proj_score2_exp
    else:
        Term1 = proj_score1_exp * K * proj_score2_exp  # g x sam1 x sam2

    # Compute Term2
    r_exp_exp =  np.expand_dims(r_exp, axis=1)  # r x 1 x 1 x dim
    rg = np.sum(r_exp_exp * np.expand_dims(g_exp, axis=-2), axis=-1)  # r x 1 x 1
    if flag_U:
        grad_2_K = -d_kernel(np.expand_dims(proj_samples1_crop_exp, axis=-1),
                             np.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                             K=K - d_K)  # g x N x N

    else:
        grad_2_K = -d_kernel(np.expand_dims(proj_samples1_crop_exp, axis=-1),
                             np.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                             K=K)  # g x N x N

    Term2 = rg * proj_score1_exp * grad_2_K  # g x sam1 x sam2

    # Compute Term3
    if flag_U:
        grad_1_K = d_kernel(np.expand_dims(proj_samples1_crop_exp, axis=-1),
                            np.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                            K=K - d_K)  # g x N x N
    else:
        grad_1_K = d_kernel(np.expand_dims(proj_samples1_crop_exp, axis=-1),
                            np.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                            K=K)  # g x N x N
    Term3 = rg * proj_score2_exp * grad_1_K


    # Compute Term4

    if flag_U:
        grad_21_K=dd_kernel(np.expand_dims(proj_samples1_crop_exp,axis=-1), np.expand_dims(proj_samples2_crop_exp,axis=-2), kernel_hyper=kernel_hyper,K=K-d_K) # g x N x N
    else:
        grad_21_K=dd_kernel(np.expand_dims(proj_samples1_crop_exp,axis=-1), np.expand_dims(proj_samples2_crop_exp,axis=-2), kernel_hyper=kernel_hyper,K=K) # g x N x N
    Term4=(rg**2)*grad_21_K # g x N x N

    divergence = Term1 + Term2 + Term3 + Term4 # g x sam1  x sam2
    if flag_U:

        KDSSD = np.sum(divergence) / ((samples1.shape[0] - 1) * samples2.shape[0])

    else:

        KDSSD = np.sum(divergence) / (samples1.shape[0] * samples2.shape[0])

    return KDSSD, divergence


def init_random_params(L):  
    eps = 0.01 + rs.rand(L,2) * 0.015
    log_eps = np.log(eps)
    log_v_r = np.zeros([ L, 2 ])
    mu0 = np.zeros((1,2))
    log_sigma0 = np.zeros((1,2))
    log_inflation = 0. * np.ones((1,2))
    g1 = np.array([[1.,0.]])
    g2 = np.array([[0.,1.]])
    return np.concatenate((log_eps, log_v_r, mu0, log_sigma0, log_inflation, g1, g2), 0)

def leapfrog(z, r, eps, log_v_r, dlogP):   
    for i in range(5):
        r_half = r - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
        z = z + eps * r_half / np.exp(log_v_r)
        r = r_half - eps / 2.0 * -getval(dlogP(z)) # stops the gradient computation
    return z, r

def generate_samples_HMC(params, n = 100):  
    log_eps = params[ 0 : L, : ]
    log_v_r = params[ L : (2 * L), : ]
    
    mu0 = getval(params[-5,:])
    mu0 = np.ones((n,1)) * mu0
    log_sigma0 = getval(params[-4,:])
    sigma0 = np.exp(log_sigma0)
    
    log_inflation = getval(params[-3,:][0])
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

def generate_samples_SKSD(params, n = 100):  
    log_eps = getval(params[ 0 : L, : ])
    log_v_r = getval(params[ L : (2 * L), : ])
    
    mu0 = getval(params[-5,:])
    mu0 = np.ones((n,1)) * mu0
    log_sigma0 = getval(params[-4,:])
    sigma0 = np.exp(log_sigma0)
    
    log_inflation = params[-3,:][0]
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

    samples_SKSD1 = generate_samples_SKSD(params, N)

    samples_SKSD2 = 0.+samples_SKSD1
    
    sigma0 = np.exp(params[-4,:])
    var0 = sigma0**2
    mu0 = params[-5,:]
    
    g1 = getval(np.array([params[-2,:]]))
    g2 = getval(np.array([params[-1,:]]))
    g1_normalized = g1/np.sqrt(np.sum(g1**2,axis=-1))
    g2_normalized = g2/np.sqrt(np.sum(g2**2,axis=-1))
    g_normalized = np.concatenate((g1_normalized, g2_normalized), 0)

    epsilon0 = rs.randn(N,params.shape[1])
    samples0 = np.ones((N,1)) * mu0 + epsilon0* (np.ones((N,1)) * sigma0) 

    elbo0 = np.mean(logP(samples0)) + np.log(2 * np.pi) + 1 + params[-4,0] + params[-4,1]

    max_SKSD, divergence = np_compute_max_SKSD(samples_SKSD1,samples_SKSD2,getval(dlogP(samples_SKSD1)),getval(dlogP(samples_SKSD2)),
                        np_SE_kernel,np_d_SE_kernel,np_dd_SE_kernel,g_normalized,flag_U=False,bandwidth_scale=1)
    
    
    return -np.mean(logP(samples_HMC)) - elbo0 + max_SKSD

def evaluate_sksd_for_g(params): 
    N = 100

    samples_SKSD1 = generate_samples_SKSD(getval(params), N)

    samples_SKSD2 = 0.+samples_SKSD1

    g1 = np.array([params[-2,:]])
    g2 = np.array([params[-1,:]])
    g1_normalized = g1/np.sqrt(np.sum(g1**2,axis=-1))
    g2_normalized = g2/np.sqrt(np.sum(g2**2,axis=-1))
    g_normalized = np.concatenate((g1_normalized, g2_normalized), 0)



    max_SKSD, divergence = np_compute_max_SKSD(getval(samples_SKSD1),getval(samples_SKSD2),getval(dlogP(samples_SKSD1)),getval(dlogP(samples_SKSD2)),
                        np_SE_kernel,np_d_SE_kernel,np_dd_SE_kernel,g_normalized,flag_U=False,bandwidth_scale=1)
    
    
    return -max_SKSD

def adam(evaluate_objective, params):
    print("    Step       |     objective      ")
    def print_perf(epoch, params):
        objective = evaluate_objective(params)
        print("{0:15}|{1:15}".format(epoch, -objective))
    m11 = 0
    m21 = 0
    
    m12=0
    m22=0
    
    beta11 = 0.9
    beta21 = 0.999
    
    beta12=0.9
    beta22=0.999
    
    epsilon = 1e-8
    alpha = 0.05
    t = 0
    grad_objective = grad(evaluate_objective)
    grad_sksd_for_g = grad(evaluate_sksd_for_g)
    epochs = 200    
    
    #start = time.time()
    for epoch in range(epochs):
        #if epoch + 1 == 100:
        #    end = time.time()
        #    print("time: {}".format(end-start))
        t += 1
        print_perf(epoch, params)
        grad_params= grad_objective(params)  
        m11 = beta11 * m11 + (1 - beta11) * grad_params
        m21 = beta21 * m21 + (1 - beta21) * grad_params**2
        m1_hat1 = m11 / (1 - beta11**t)
        m2_hat1 = m21 / (1 - beta21**t)
        
        params[:-2] = params[:-2] - alpha * m1_hat1[:-2] / (np.sqrt(m2_hat1[:-2]) + epsilon)   #alpha is step size of adam
        
        grad_sksd= grad_sksd_for_g(params)  
        m12 = beta12 * m12 + (1 - beta12) * grad_sksd
        m22 = beta22 * m22 + (1 - beta22) * grad_sksd**2
        m12_hat = m12 / (1 - beta12**t)
        m22_hat = m22 / (1 - beta22**t)
        
        params[-2:] = params[-2:] - alpha * m12_hat[-2:] / (np.sqrt(m22_hat[-2:]) + epsilon)
        

    return params


if __name__ == "__main__":
    rs = npr.RandomState(0)
    L = 30
    logP = logP_gauss     # can specify other targets included in targets.py
    dlogP = dlogP_gauss
    
    
    params = init_random_params(L)
    params = adam(evaluate_objective, params)     
    exp_params = np.exp(params)
    exp_params[-5,:] = np.log(exp_params[-5,:])
    exp_params[-1,:] = np.log(exp_params[-1,:])
    exp_params[-2,:] = np.log(exp_params[-2,:])
    exp_params[-3,1] = None
    print("step_sizes, mu, sigma, inflation and g: {}".format(exp_params))
    
    z = generate_samples_HMC(params, 100000)
    
    print("-Expexted Log Target Estimate: {}".format(-np.mean(logP(z))))   
    print("KSD: {}".format(KSD(z[:10000,:], dlogP(z[:10000,:]),flag_U = False)))
    
    z1=z[:,0]
    z2=z[:,1]
    plt.hist2d(z1, z2, bins=(300, 300))
    #plt.xlim(-4,4)
    #plt.ylim(-4,4)
    plt.show()