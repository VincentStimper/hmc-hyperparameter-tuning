import numpy as np

def generate_data(n, d, tau, sigma_0):
    u = np.random.normal(0,1,(n, d))
    norm = np.sum(u**2, axis=1) **(0.5)
    norm = np.expand_dims(norm, 1)
    X = u/norm

    lamb = np.abs(np.random.standard_cauchy(size=d))
    w = np.random.randn(d) * tau * lamb

    noise = np.random.normal(0, sigma_0, (n,))

    y = np.dot(X, w) + noise

    return X, y, w, lamb