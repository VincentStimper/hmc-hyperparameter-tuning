import tensorflow as tf

from util.constant import log_2pi
from core.ham import leapfrog
import numpy as np


class HamInfNet:
    def __init__(self, num_lfsteps,
                 num_layers,
                 sample_dim,
                 training=False,
                 min_step_size=0.01,
                 init_step_scale=1.0,
                 log_q0_std_init=0.0,
                 name_space="",
                 stop_gradient=True,
                 dtype=tf.float32):
        self.num_lfsteps = num_lfsteps
        self.num_layers = num_layers
        self.num_layers_max = num_layers
        self.stop_gradient = stop_gradient
        self.sample_dim = sample_dim
        self.dtype = dtype

        self.lfstep_size_raw = tf.get_variable(name="{}lfstep_size".format(name_space),
                                               initializer=tf.constant(
                                                   init_step_scale*np.random.uniform(0.02, 0.05, size=(num_layers, 1, sample_dim)),
                                                   dtype=dtype),
                                               trainable=True, dtype=dtype)
        self.q0_mean = tf.get_variable(name="{}q0_mean".format(name_space),
                                       shape=(1, sample_dim),
                                       initializer=tf.zeros_initializer,
                                       trainable=training, dtype=dtype)
        self.log_q0_std = tf.get_variable(name="{}log_q0_std".format(name_space),
                                          shape=(1, sample_dim),
                                          initializer=tf.constant_initializer(value=log_q0_std_init),
                                          trainable=training, dtype=dtype)
        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size

    def getParams(self):
        return self.lfstep_size_raw, self.q0_mean, self.log_q0_std

    def getlf_step(self):
        return self.lfstep_size_raw

    def getInitParams(self):
        return self.q0_mean, self.log_q0_std

    def __build_LF_graph(self, pot_fun, state_init, momentum, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=momentum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=self.lfstep_size[layer_index],
                                    r_var=1.0,
                                    numleap=self.num_lfsteps,
                                    stop_gradient_pot=self.stop_gradient,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final

    def __build_LF_scan_graph(self, pot_fun, state_init, momentum, back_prop=False):
        initializer = (state_init, tf.zeros_like(state_init, dtype=self.dtype))
        elems = tf.range(0, self.num_layers_max)

        def _loopbody(state, layer_index):
            state_new, momentum_new = leapfrog(x=state[0],
                                               r=momentum[layer_index],
                                               pot_fun=pot_fun,
                                               eps=self.lfstep_size[layer_index],
                                               r_var=1.0,
                                               numleap=self.num_lfsteps,
                                               stop_gradient_pot=self.stop_gradient,
                                               back_prop=back_prop)
            return (state_new, momentum_new)

        state_final = tf.scan(_loopbody, elems=elems, initializer=initializer)
        final_position = state_final[0][-1]
        momentum_out = state_final[1]
        return final_position, momentum_out

    def build_simulation_gauss_graph(self, pot_fun, sample_batch_size, input_data_batch_size):
        state_init, _ = self.__prepare_state_init(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph(pot_fun, state_init, momentum, back_prop=False)
        return state_samples

    def build_simulation_gauss_graph_with_length(self, pot_fun, sample_batch_size, input_data_batch_size, length):
        state_init, _ = self.__prepare_state_init(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph(pot_fun, state_init, momentum, length, back_prop=False)
        return state_samples

    def build_simulation_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size):
        state_init, _ = state_init_gen(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph(pot_fun, state_init, momentum, back_prop=False)
        return state_samples

    def __prepare_state_init(self, sample_batch_size, input_data_batch_size):
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * tf.exp(self.log_q0_std) + self.q0_mean
        log_q0_z = -0.5 * tf.reduce_sum(q0_init ** 2 + 2*self.log_q0_std, axis=-1)
        return state_init, log_q0_z  # sample_batch_size x input_data_batch_size x sample_dim

    def build_elbo_graph_gauss(self, pot_fun, sample_batch_size, input_data_batch_size, training=False):
        return self.build_elbo_graph(pot_fun, self.__prepare_state_init, sample_batch_size, input_data_batch_size,
                                     training)

    def build_elbo_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        """
        sample batch shape: sample_batch_size x input_data_batch_size x sample_dim
        potential batch shape: sample_batch_size x input_data_batch_size

        the list of shape of variables
        state_init: sample batch shape
        pot_energy_all_samples_final: potential batch shape
        log_q0_z: potential batch shape

        :param pot_fun: the potential function takes a batch of samples as input and outputs batch of potential values
        :param state_init_gen: take sample_batch_size and input_data_batch_size as input and outputs a batch of samples
        from initial distribution q_0 and their log probability log q_0
        :param sample_batch_size: the number of samples used to estimate the gradient
        :param input_data_batch_size: the batch size of input data, which must be compatible with potential function
        :param training: Boolean variable true for training / false for evaluation

        :return:
        elbo_mean: the Monte Carlo (sample average) estimation of loss function

        """
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        # state_init_stop_gradient = tf.stop_gradient(state_init)
        state_final = self.__build_LF_graph(pot_fun, state_init, momentum, back_prop=training)
        # momentum_init = momentum[0]
        # momentum_last = tf.random_normal(stddev=1.0,
        #                                  shape=(sample_batch_size, input_data_batch_size,
        #                                         self.sample_dim), dtype=self.dtype)
        # state_final2, momentum_final = leapfrog(x=state_final,
        #          r=momentum_last,
        #          pot_fun=pot_fun,
        #          eps=self.lfstep_size[self.num_layers-1],
        #          r_var=1.0,
        #          numleap=self.num_lfsteps*5,
        #          back_prop=training)
        # state_final = state_final2
        #
        # elbo_per_data_momentum = -0.5 * tf.reduce_sum(momentum_final ** 2 - momentum_init ** 2, axis=-1)

        ####################################### Compute Energy ########################################
        # pot_energy_all_sample shape: sample_batch_size x input_data_batch_size
        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood
        pot_energy_all_samples_init = pot_fun(state_init)  # potential function is the negative log likelihood
        pot_gaussian_prior = 0.5*tf.reduce_sum(state_final**2 + log_2pi, axis=-1)

        # nelbo_per_sample = -elbo_per_data_momentum
        nelbo_per_sample = pot_energy_all_samples_final
        # nelbo_per_sample = pot_energy_all_samples_final + pot_energy_all_samples_init + log_q0_z  #- elbo_per_data_momentum #+ kinetic_out - kinetic_in  #  # -logp(x) + logq(x)
        nelbo_per_sample_x = pot_energy_all_samples_final
        elbo_per_data = tf.reduce_mean(-nelbo_per_sample, axis=0)
        elbo_per_data_x = tf.reduce_mean(-nelbo_per_sample_x, axis=0)
        logD_per_data = tf.reduce_logsumexp(-nelbo_per_sample, axis=0, keepdims=True) - tf.log(
            tf.constant(sample_batch_size, dtype=tf.float32))
        elbo_mean = tf.reduce_mean(elbo_per_data)
        elbo_x_mean = (-tf.reduce_mean(elbo_per_data_x), tf.reduce_mean(log_q0_z))
        logD_mean = tf.reduce_mean(logD_per_data)
        recon_mean = tf.reduce_mean(pot_energy_all_samples_final - pot_gaussian_prior)
        return elbo_mean, recon_mean, elbo_x_mean
