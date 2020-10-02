import tensorflow as tf
from decoder.vae_helper import xavier_init
from util.utils import batch_matmul

from decoder import VAE_ABC_GPU


class VAE_SHNN_GPU(VAE_ABC_GPU):
    def __init__(self, z_dim, h_dim, num_vis, dtype, vfun='sigmoid', trainable=True):
        self.num_vis = num_vis
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.dtype = dtype
        self.vfun = vfun
        output_dim = num_vis
        if vfun == 'trunc_logistic':
            self.logscale = tf.get_variable(name="vae_gpu_x_trunc_logistic_logscale", dtype=dtype,
                                    initializer=tf.zeros(shape=(1, output_dim), dtype=dtype), trainable=trainable)
        if vfun == 'bernoulli_poisson':
            output_dim = 2*output_dim
        self.P_W1 = tf.get_variable(name="vae_gpu_zh_weights1", dtype=dtype,
                                    initializer=xavier_init([z_dim, h_dim], dtype), trainable=trainable)
        self.P_W2 = tf.get_variable(name="vae_gpu_hx_weights2", dtype=dtype,
                                    initializer=xavier_init([h_dim, output_dim], dtype), trainable=trainable)

        self.P_b1 = tf.get_variable(name="vae_gpu_zh_bias1", dtype=dtype,
                                    initializer=tf.zeros(shape=(1, h_dim), dtype=dtype), trainable=trainable)
        self.P_b2 = tf.get_variable(name="vae_gpu_hx_bias2", dtype=dtype,
                                    initializer=tf.zeros(shape=(1, output_dim), dtype=dtype), trainable=trainable)

    def get_parameters(self):
        params = (self.P_W1, self.P_W2, self.P_b1, self.P_b2)
        if self.vfun == 'trunc_logistic':
            params += (self.logscale, )
        return params

    def get_parameters_reg(self):
        reg = tf.reduce_sum(self.P_W1**2) + tf.reduce_sum(self.P_W2**2) + \
            tf.reduce_sum(self.P_b1**2) + tf.reduce_sum(self.P_b2**2)
        if self.vfun == 'trunc_logistic':
            reg += tf.reduce_sum(self.logscale**2)
        return reg

    def z_to_logits(self, z):
        h = tf.nn.relu(batch_matmul(z, self.P_W1, dim=(-1, 0)) + self.P_b1)
        h_reg = h
        return batch_matmul(h_reg, self.P_W2, dim=(-1, 0)) + self.P_b2
