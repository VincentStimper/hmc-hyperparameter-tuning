import tensorflow as tf

from decoder.vae_conv_util import generator, get_parameters
from decoder.vae_gen_gpu import VAE_ABC_GPU

class VAE_DCNN_GPU(VAE_ABC_GPU):
    def __init__(self, dimH=500, dimZ=32, vfun='sigmoid'):
        self.z_dim = dimZ
        self.h_dim = dimH
        self.dcnn = generator(dimH=dimH, dimZ=dimZ)
        self.dtype = tf.float32
        self.vfun = vfun

    def get_parameters(self):
        return tuple(get_parameters())

    def get_parameters_reg(self):
        reg = tf.zeros(shape=())
        for var in get_parameters():
            reg += tf.reduce_sum(var ** 2)
        return reg

    def z_to_logits(self, z):
        z_shape = tuple(z.shape.as_list())
        dim_keep = z_shape[0:len(z_shape)-1]
        dim_keep_prod = 1
        for dim in dim_keep:
            dim_keep_prod *= dim
        z_reshaped = tf.reshape(z, shape=(dim_keep_prod, self.z_dim))
        print(z_reshaped)
        logits = self.dcnn(z_reshaped)
        return tf.reshape(logits, shape=dim_keep+(-1,)) 
