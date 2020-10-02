from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd.scipy.special import logsumexp

# correlated gaussian
def logP_gauss(x): 
    cov = np.array([[2,1.5],[1.5,1.6]])
    cov_inv = np.linalg.inv(cov)
    return -0.5* np.einsum('ij,ji->i', np.dot(x, cov_inv), x.T) - np.log(2*np.pi) -0.5 * np.log(np.linalg.det(cov))

def dlogP_gauss(x): 
    cov = np.array([[2,1.5],[1.5,1.6]])
    cov_inv = np.linalg.inv(cov)
    grad_x = -np.matmul(x,cov_inv[0,:].T)
    grad_y = -np.matmul(x,cov_inv[1,:].T)
    return np.column_stack((grad_x, grad_y))



# laplace
def logP_laplace(x): 
    x0 = x[:,0]
    x1 = x[:,1]
    return -np.abs(x0 - 5.0) - np.abs(x1 - 5.0)

def dlogP_laplace(x): 
    x0 = x[:,0]
    x1 = x[:,1]
    grad_x = (x0-5)/np.abs(x0-5)
    grad_x=np.clip(grad_x, -1, 1)
    grad_y = (x1-5)/np.abs(x1-5)
    grad_y=np.clip(grad_y, -1, 1)
    return np.column_stack((-grad_x, -grad_y))



# dual moon
def logP_dual(x): 
    x1 = x[:,0]
    x2 = x[:,1]
    term1 = 3.125 * (np.sqrt(x1**2+x2**2)-2)**2
    term2 = np.log(1e-16+np.exp(-0.5*((x1+2)/0.6)**2) + np.exp(-0.5*((x1-2)/0.6)**2))
    return -term1 + term2 

def dlogP_dual(x): 
    x1 = x[:,0]
    x2 = x[:,1]
    r = np.sqrt(x1**2+x2**2)
    grad_x = 6.25*(1-2/r)*x1 - ((-1/0.36)*x1 - 2/0.36 *np.exp(-0.5/0.36 *(x1+2)**2) + 2/0.36 *np.exp(-0.5/0.36 *(x1-2)**2))
    grad_y = 6.25*(1-2/r)*x2
    return np.column_stack((-grad_x, -grad_y))



# gaussian mixture
def logP_mixture(x):  
    num_modes = 7
    angle = np.arange(0, 2 * np.pi, 2 * np.pi / num_modes)
    mu = np.transpose(np.stack([[np.cos(angle), np.sin(angle)]], axis=0), axes=(0, 2, 1)) * 5.0
    zz = x[None,:].reshape((x.shape[0],1,2))
    return np.log(1e-300 + np.exp(logsumexp(-0.5*(np.sum((zz-mu)**2,2)),1)))

def dlogP_mixture(x):
    P = np.exp(logP_mixture(x))
    num_modes = 7
    angle = np.arange(0, 2 * np.pi, 2 * np.pi / num_modes)
    mu = np.transpose(np.stack([[np.cos(angle), np.sin(angle)]], axis=0), axes=(0, 2, 1)) * 5.0
    zz = x[None,:].reshape((x.shape[0],1,2))
    grad_x = np.sum(np.exp(-0.5*(np.sum((zz-mu)**2,2)))* (zz-mu)[:,:,0],1)
    grad_y = np.sum(np.exp(-0.5*(np.sum((zz-mu)**2,2)))* (zz-mu)[:,:,1],1)
    return np.column_stack((-grad_x/P, -grad_y/P))



# wave1
def logP_wave1(x): 
    x1 = x[:,0]
    x2 = x[:,1]
    return -0.5* ((x2 + np.sin(0.5* np.pi *x1))/0.4)**2

def dlogP_wave1(x): 
    x1 = x[:,0]
    x2 = x[:,1]
    grad_x = -np.pi/0.32 *np.cos(0.5*np.pi*x1)* (x2+np.sin(0.5*np.pi*x1))
    grad_y = -6.25* (x2+np.sin(0.5* np.pi* x1))
    return np.column_stack((grad_x, grad_y))



# wave2
def logP_wave2(x):
    x0 = x[:,0]
    x1 = x[:,1]
    term1 = np.exp(-0.5* ((x1 + np.sin(0.5*np.pi* x0))/0.35)**2)
    term2 = np.exp(-0.5* ((-x1 - np.sin(0.5*np.pi* x0) + 3*np.exp(-0.5/0.36* (x0-1)**2))/0.35)**2)
    return np.log(1e-300+ term1 + term2)
    
def dlogP_wave2(x):
    x0 = x[:,0]
    x1 = x[:,1]
    term1 = np.exp(-0.5* ((x1 + np.sin(0.5*np.pi* x0))/0.35)**2)
    term2 = np.exp(-0.5* ((-x1 - np.sin(0.5*np.pi* x0) + 3*np.exp(-0.5/0.36* (x0-1)**2))/0.35)**2)
    denom = term1+term2+1e-300
    term3 = -np.pi*(x1+np.sin(0.5*np.pi*x0))*np.cos(0.5*np.pi*x0)/0.245
    term4 = -(-x1-np.sin(0.5*np.pi*x0)+3*np.exp(-0.5*((x0-1)/0.6)**2))/0.35 * ((-1/0.7)*np.pi*np.cos(0.5*np.pi*x0) + (3/0.35)*np.exp(-0.5*((x0-1)/0.6)**2)*(-(x0-1)/0.36))             
    grad_x =  (term1*term3 + term2* term4)/denom
    grad_y =  (term1*(-(x1+np.sin(0.5*np.pi*x0))/0.35**2) - term2*(-(-x1-np.sin(0.5*np.pi*x0)+3*np.exp(-0.5/0.36* (x0-1)**2))/0.35**2))/denom
    return np.column_stack((grad_x, grad_y))



# wave3
def logP_wave3(x):
    x0 = x[:,0]
    x1 = x[:,1]
    term1 = np.exp(-0.5* ((x1 + np.sin(0.5*np.pi* x0))/0.4)**2)
    term2 = np.exp(-0.5* ((-x1 - np.sin(0.5*np.pi* x0) + 3/(1+np.exp(-(x0-1)/0.3)))/0.35)**2)
    return np.log(1e-9+term1 + term2)
    
def dlogP_wave3(x):
    x0 = x[:,0]
    x1 = x[:,1]
    term1 = np.exp(-0.5* ((x1 + np.sin(0.5*np.pi* x0))/0.4)**2)
    term2 = np.exp(-0.5* ((-x1 - np.sin(0.5*np.pi* x0) + 3/(1+np.exp(-(x0-1)/0.3)))/0.35)**2)
    denom = term1+term2+1e-9
    term3 = -np.pi*(x1-np.sin(0.5*np.pi*x0))*np.cos(0.5*np.pi*x0)/0.32
    term4 = -(-x1-np.sin(0.5*np.pi*x0)+3/(1+np.exp((1-x0)/0.3)))*(-0.5*np.pi*np.cos(0.5*np.pi*x0) + 10/(np.exp((1-x0)/0.3)+np.exp((x0-1)/0.3)))/0.35**2
    grad_x = (term1*term3 + term2* term4)/denom
    grad_y = (term1*(-(x1+np.sin(0.5*np.pi*x0))/0.4**2) - term2*(-(-x1-np.sin(0.5*np.pi*x0)+3/(1+np.exp((1-x0)/0.3)))/0.35**2))/denom
    return np.column_stack((grad_x, grad_y))

