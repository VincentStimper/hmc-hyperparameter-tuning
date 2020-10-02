import tensorflow as tf
from util.constant import log_2pi
from core.ham import leapfrog
import numpy as np


class HamInfNetNN:
    def __init__(self, num_lfsteps,
                 num_layers,
                 sample_dim,
                 min_step_size=0.01,
                 dtype=tf.float32):
        self.num_lfsteps = num_lfsteps
        self.num_layers = num_layers
        self.num_layers_max = num_layers

        self.sample_dim = sample_dim
        self.dtype = dtype

        self.lfstep_size_raw = tf.get_variable(name="lfstep_size",
                                           initializer=tf.constant(
                                               np.random.uniform(0.02, 0.05, size=(num_layers, 1, sample_dim)),
                                               dtype=dtype),
                                           trainable=True, dtype=dtype)
        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size
        
        
        self.raw_inflation = tf.get_variable(name="raw_inflation",
                                             #shape=(),
                                             #initializer=tf.zeros_initializer,
                                             initializer=tf.constant(1.0),
                                             trainable=True, dtype=dtype)
        self.inflation=tf.abs(self.raw_inflation)
        
        # if using maxsksd, then we need to incorporate g: 
        #self.g=tf.get_variable(name="g",
        #                       initializer=tf.constant(np.eye(sample_dim),dtype=tf.float32),
        #                       trainable=True,dtype=dtype)
        #self.g_normalized=self.g/tf.sqrt(tf.reduce_sum(self.g**2,axis=-1,keepdims=True))
         
    def getParams(self):
        #return self.lfstep_size_raw,  self.raw_inflation, self.g
        return self.lfstep_size_raw,  self.raw_inflation

    def getInflation(self):
        return self.inflation
    
    """
    def get_gnorm(self):
        return self.g_normalized
    """
    
    def get_step_size(self):
        return self.lfstep_size

    def __build_LF_graph(self, pot_fun, state_init,momemtum, back_prop=False):
        cond = lambda layer_index, state: tf.less(layer_index, self.num_layers)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=momemtum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=self.lfstep_size[layer_index],
                                    r_var=1.0,
                                    numleap=self.num_lfsteps,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final
   
    def state_init_gen(self, mean, logvar, sample_batch_size, input_data_batch_size):
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * self.inflation*(tf.exp(logvar/2)) + mean
        return state_init
    
    def build_simulation_graph(self, pot_fun, mean, logvar, sample_batch_size, input_data_batch_size,training=False):
    
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * (self.inflation)*(tf.exp(logvar/2)) + mean
       
        momemtum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        
        state_init_stop_gradient = tf.stop_gradient(state_init)
        state_final = self.__build_LF_graph(pot_fun, state_init_stop_gradient,momemtum, back_prop=training)
       
        return state_final