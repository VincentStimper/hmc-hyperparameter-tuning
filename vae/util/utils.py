import tensorflow as tf
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def dybinarize_mnist(x, high=1.0):
    return (x > np.random.uniform(high=high, size=np.shape(x))).astype(np.float32)


def binarise_fashion_mnist(X):
    X_cat = np.floor(X * 10)/10
    X_sum = np.sum(X_cat, axis=-1, keepdims=True)
    X_mean = ( X_sum / np.reshape(np.count_nonzero(X_cat, axis=1), newshape=np.shape(X_sum)))
    u = np.random.uniform(size=(np.shape(X_cat)[0], 1))*0.5 + 0.5
    X_norm = X_mean*u
    X_normed = X_cat/X_norm
    return (X_normed > np.random.uniform(size=np.shape(X_cat))).astype(float)



def batch_matmul(A, B, dim, einsum_expr_only=False):
    rank_A = len(A.shape)
    rank_B = len(B.shape)
    a_expr = ''
    b_expr = ''
    c_expr = ''
    dim_a = rank_A + dim[0] if dim[0] < 0 else dim[0]
    dim_b = rank_B + dim[1] if dim[1] < 0 else dim[1]

    for i in range(ord('a'), ord('a') + rank_A):
        a_expr += chr(i)
        if (i - ord('a')) is not dim_a:
            c_expr += chr(i)
    for i in range(ord('a') + rank_A, ord('a') + rank_A + rank_B):
        if (i - ord('a') - rank_A) is not dim_b:
            c_expr += chr(i)
            b_expr += chr(i)
        else:
            b_expr += a_expr[dim_a]
    ein_sum_expr = '{},{}->{}'.format(a_expr, b_expr, c_expr)
    if einsum_expr_only:
        return ein_sum_expr
    else:
        return tf.einsum(ein_sum_expr, A, B)


def generate_hist2d(samples, bin_size, xlim=None, y_lim=None):
    if xlim is None:
        x_min = np.floor(min(samples[:, 0]) * 10) / 10
        x_max = np.ceil(max(samples[:, 0]) * 10) / 10
    else:
        x_min = xlim[0]
        x_max = xlim[1]
    if y_lim is None:
        y_min = np.floor(min(samples[:, 1]) * 10) / 10
        y_max = np.ceil(max(samples[:, 1]) * 10) / 10
    else:
        y_min = y_lim[0]
        y_max = y_lim[1]

    # x_min_s = -min(abs(x_min), abs(x_max))
    # x_max_s = min(abs(x_min), abs(x_max))
    # y_min_s = -min(abs(y_min), abs(y_max))
    # y_max_s = min(abs(y_min), abs(y_max))

    xedges = np.arange(start=x_min, stop=x_max + bin_size, step=bin_size)
    yedges = np.arange(start=y_min, stop=y_max + bin_size, step=bin_size)
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=(xedges, yedges))
    H = H.T  # Let each row list bins with common y range.
    return H, xedges, yedges

def show_hist1d(samples):
    plt.hist(samples, 50, normed=1, facecolor='green', alpha=0.75)
    plt.grid(True)
    plt.show()


def show_hist2d(samples, bin_size=0.01, xlim=None, y_lim=None, title=None, save_path=None):
    H, xedges, yedges = generate_hist2d(samples, bin_size)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(X, Y, H)
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is not None:
        plt.title(title)
    if not (save_path is None):
        plt.savefig(save_path, format="pdf")
    plt.show()


def show_generated_hist2d(H, xedges, yedges, xlim=None, y_lim=None, title=None, save_path=None):
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    H = H
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pcolormesh(X, Y, H)
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if title is not None:
        plt.title(title)
    if not (save_path is None):
        plt.savefig(save_path, format="pdf")
    plt.show()


def numeric_integrator_2d(fun, x_batch, dx):
    log_part = tf.reduce_logsumexp(-fun(x_batch) + tf.log(tf.constant(dx, dtype=tf.float32)))
    return log_part


def plot_mnist(samples, size=(5, 5), title=None, save_path=None, show=False):
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(size[0], size[1])
    if title is not None:
        fig.suptitle(title)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if not (save_path is None):
        plt.savefig(save_path, format="pdf")
    if show:
        plt.show()
    return fig
