import tensorflow as tf

from util.constant import log_2pi
from core.ham import leapfrog
import numpy as np


class HamInfNetHEI:
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
                                                   init_step_scale*np.random.uniform(0.02,0.05,size=(num_layers,1,sample_dim)),
                                                   dtype=dtype),
                                               trainable=True, dtype=dtype)
       
        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size
       
        self.raw_inflation = tf.get_variable(name="{}raw_inflation".format(name_space),
                                             initializer=tf.constant(np.sqrt(1.0),dtype=dtype),
                                             trainable=True, dtype=dtype)
        self.inflation=tf.abs(self.raw_inflation)
        
        self.g=tf.get_variable(name="{}g".format(name_space),
                               initializer=tf.constant(np.eye(sample_dim),dtype=tf.float32),
                               trainable=True,dtype=dtype)
        self.g_normalized=self.g/tf.sqrt(tf.reduce_sum(self.g**2,axis=-1,keepdims=True))
        
        
    def __build_LF_graph_hmc(self, pot_fun, state_init, momemtum,num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=momemtum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=self.lfstep_size[layer_index],
                                    r_var=1.0,
                                    numleap=self.num_lfsteps,
                                    stop_gradient_pot=self.stop_gradient,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final
    
    def __build_LF_graph_maxsksd(self, pot_fun, state_init,momemtum, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=momemtum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=tf.stop_gradient(self.lfstep_size[layer_index]),
                                    r_var=1.0,
                                    numleap=self.num_lfsteps,
                                    stop_gradient_pot=self.stop_gradient,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final

    
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
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size, self.inflation)
       
        momemtum=tf.random_normal(stddev=1.0,
                                  shape=(self.num_layers_max,sample_batch_size,input_data_batch_size,
                                  self.sample_dim),dtype=self.dtype)
        state_final = self.__build_LF_graph_hmc(pot_fun, state_init,momemtum, back_prop=training)

        ####################################### Compute Energy ########################################
        # pot_energy_all_sample shape: sample_batch_size x input_data_batch_size
        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood
       
        pot_gaussian_prior = 0.5*tf.reduce_sum(state_final**2 + log_2pi, axis=-1)

        nelbo_per_sample = pot_energy_all_samples_final
        
        elbo_per_data = tf.reduce_mean(-nelbo_per_sample, axis=0)
        
        elbo_mean = tf.reduce_mean(elbo_per_data)
        
        recon_mean = tf.reduce_mean(pot_energy_all_samples_final - pot_gaussian_prior)
        
        return elbo_mean, recon_mean      

    
    def build_maxsksd_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        # use V-statistics
        def tf_median_heruistic_proj(sample1,sample2):
            '''
            Median Heuristic for projected samples
            '''
            # samples 1 is * x g x N x 1
            # samples 2 is * x g x N x 1
        
            G=tf.reduce_sum(sample1*sample1,axis=-1) # * x num_g x N or r x g x N
            G_exp = tf.expand_dims(G, axis=-2)  # * x num_g x 1 x N or * x r x g x 1 x N
        
            H=tf.reduce_sum(sample2*sample2,axis=-1) # * x num_g x N or * x r x g x N
            H_exp=tf.expand_dims(H, axis=-1) # * x numb_g x N x 1 or * x r x g x N x 1
        
            dist = G_exp + H_exp - 2*tf.matmul(sample2,tf.transpose(sample1,(0,2,1))) # * x G x N x N
        
        
            dist_triu = tf.matrix_band_part(dist, 0, -1) # Upper triangular matrix of 0s and 1s
        
            def get_median(v):   #v is g * N * N
        
                length_triu = 0.5*(v.get_shape()[1].value+1)*v.get_shape()[1].value
        
                mid = int(0.5*(length_triu) + 1)
                
                return tf.nn.top_k(tf.reshape(v,(v.shape[0],-1)),mid).values[:,-1]
        
            return tf.stop_gradient(get_median(dist_triu))
        
        def tf_SE_kernel(sample1,sample2,**kwargs):
            '''
            Compute the square exponential kernel
            :param sample1: x
            :param sample2: y
            :param kwargs: kernel hyper-parameter: bandwidth
            :return:
            '''
        
            bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or * x g
        
            bandwidth_exp=tf.expand_dims(tf.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
            K = tf.exp(-(sample1 - sample2) ** 2 / (bandwidth_exp ** 2+1e-9)) # g x sam1 x sam2
            return K
        
        def tf_d_SE_kernel(sample1,sample2,**kwargs):
            'The gradient of RBF kernel'
            K=kwargs['K'] # * x g x sam1 x sam2
        
            bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g
        
            bandwidth_exp=tf.expand_dims(tf.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
            d_K=K*(-1/(bandwidth_exp**2+1e-9)*2*(sample1-sample2)) # g x sam1 x sam2
        
            return d_K
        def tf_dd_SE_kernel(sample1,sample2,**kwargs):
            K=kwargs['K'] # * x g x sam1 x sam2
        
            bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or r x g or * x g
        
            bandwidth_exp=tf.expand_dims(tf.expand_dims(bandwidth,axis=-1),axis=-1) # g x 1 x 1
            dd_K=K*(2/(bandwidth_exp**2+1e-9)-4/(bandwidth_exp**4+1e-9)*(sample1-sample2)**2)
        
            return dd_K # g x N x N
        
        
        def tf_compute_max_SKSD(samples1,samples2,score1,score2,g,kernel=tf_SE_kernel,d_kernel=tf_d_SE_kernel,dd_kernel=tf_dd_SE_kernel,bandwidth_scale=1.0):
            '''
            tensorflow version of maxSKSD with median heuristics
            :param samples1: samples from q with shape: N x dim
            :param samples2: samples from q with shape: N x dim
            :param score1: score of p for samples 1 with shape N x dim
            :param score2: score of p for samples 2 with shape N x dim
            :param kernel: kernel function (default: tf_SE_kernel)
            :param d_kernel: derivative of kernel function (default: tf_d_SE_kernel)
            :param dd_kernel: second derivative of kernel function (default: tf_dd_SE_kernel)
            :param g: sliced direction with shape dim x dim
            :param bandwidth_scale: coefficient for bandwidth (default:1)
            :return: KDSSD: discrepancy value; divergence:each component for KDSSD (used for debug or GOF Test)
            '''
            dim=samples1.shape[-1].value
            r=tf.eye(dim)
        
            kernel_hyper={}
            ##### Compute the median for each slice direction g
            if samples1.shape[0] > 500: # To reduce the sample number for median computation
                idx_crop = 500
            else:
                idx_crop = samples1.shape[0]
        
            g_cp_exp = tf.expand_dims(g, 1)  # g x 1 x dim
            samples1_exp = tf.expand_dims(samples1[0:idx_crop, :], 0)  # 1 x N x dim
            samples2_exp = tf.expand_dims(samples2[0:idx_crop, :], 0)  # 1 x N x dim
            proj_samples1 = tf.reduce_sum(samples1_exp * g_cp_exp, axis=-1, keepdims=True)  # g x N x 1
            proj_samples2 = tf.reduce_sum(samples2_exp * g_cp_exp, axis=-1, keepdims=True)  # g x N x 1
            median_dist = tf_median_heruistic_proj(proj_samples1, proj_samples2)  # g
            bandwidth_array = bandwidth_scale*2 * tf.sqrt(0.5 * median_dist)
            kernel_hyper['bandwidth_array'] = bandwidth_array
        
            ##### Now compute the SKSD with slice direction g for each dimension
            # Compute Term1
        
            g_exp = tf.reshape(g, (g.shape[0], 1, g.shape[-1]))  # g x 1 x D
            samples1_crop_exp = tf.expand_dims(samples1, axis=0)  # 1 x N x D
            samples2_crop_exp = tf.expand_dims(samples2, axis=0)  # 1 x N x D
            proj_samples1_crop_exp = tf.reduce_sum(samples1_crop_exp * g_exp, axis=-1)  # g x sam1
            proj_samples2_crop_exp = tf.reduce_sum(samples2_crop_exp * g_exp, axis=-1)  # g x sam2
        
            r_exp = tf.expand_dims(r, axis=1)  # r x 1 x dim
            proj_score1 = tf.reduce_sum(r_exp * tf.expand_dims(tf.stop_gradient(score1), axis=0), axis=-1, keepdims=True)  # r x sam1 x 1
            proj_score2 = tf.reduce_sum(r_exp * tf.expand_dims(tf.stop_gradient(score2), axis=0), axis=-1)  # r x sam2
        
            proj_score1_exp = proj_score1  # r x sam1 x 1
            proj_score2_exp = tf.reshape(proj_score2,(proj_score2.shape[0], 1, proj_score2.shape[-1])) # r x 1 x sam2
        
            K = kernel(tf.expand_dims(proj_samples1_crop_exp, axis=-1), tf.expand_dims(proj_samples2_crop_exp, axis=-2),
                       kernel_hyper=kernel_hyper)  # g x sam1 x sam 2
            
            Term1 = proj_score1_exp * K * proj_score2_exp  # g x sam1 x sam2
        
            # Compute Term2
            r_exp_exp =  tf.expand_dims(r_exp, axis=1)  # r x 1 x 1 x dim
            rg = tf.reduce_sum(r_exp_exp * tf.expand_dims(g_exp, axis=-2), axis=-1)  # r x 1 x 1
            
            grad_2_K = -d_kernel(tf.expand_dims(proj_samples1_crop_exp, axis=-1),
                                     tf.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                                     K=K)  # g x N x N
        
            Term2 = rg * proj_score1_exp * grad_2_K  # g x sam1 x sam2
        
            # Compute Term3
            
            grad_1_K = d_kernel(tf.expand_dims(proj_samples1_crop_exp, axis=-1),
                                    tf.expand_dims(proj_samples2_crop_exp, axis=-2), kernel_hyper=kernel_hyper,
                                    K=K)  # g x N x N
            Term3 = rg * proj_score2_exp * grad_1_K
        
        
            # Compute Term4
        
            
            grad_21_K=dd_kernel(tf.expand_dims(proj_samples1_crop_exp,axis=-1), tf.expand_dims(proj_samples2_crop_exp,axis=-2), kernel_hyper=kernel_hyper,K=K) # g x N x N
            Term4=(rg**2)*grad_21_K # g x N x N
        
            divergence = Term1 + Term2 + Term3 + Term4 # g x sam1  x sam2
            
        
            KDSSD = tf.reduce_sum(divergence) / (samples1.shape[0].value * samples2.shape[0].value)

            return KDSSD
        
        
        momemtum=tf.random_normal(stddev=1.0,
                                  shape=(self.num_layers_max,sample_batch_size,input_data_batch_size,
                                  self.sample_dim),dtype=self.dtype)
        
        state_init, _ = state_init_gen(sample_batch_size, input_data_batch_size, self.inflation)
        
        state_final1 = self.__build_LF_graph_maxsksd(pot_fun, state_init,momemtum, back_prop=training)
        
        #state_final2 = state_final1 + 0.
        state_final2=tf.identity(state_final1)
        pot_energy_all_samples1 = pot_fun(state_final1)  # sample_size * input_batch , neg log-lik
        grad_pot_all_samples1 = tf.gradients(ys= -pot_energy_all_samples1, xs = state_final1)[0] #sample_size * input_batch* latent_dim
        
        pot_energy_all_samples2 = pot_fun(state_final2)  # sample_size * input_batch , neg log-lik
        grad_pot_all_samples2 = tf.gradients(ys= -pot_energy_all_samples2, xs = state_final2)[0] #sample_size * input_batch* latent_dim
        
        g_direction=self.g_normalized
        
        cond = lambda batch_index, maxsksd_sum: tf.less(batch_index, input_data_batch_size)
        def _loopbody(batch_index, maxsksd_sum):
            return batch_index+1, maxsksd_sum+tf_compute_max_SKSD(state_final1[:,batch_index,:],state_final2[:,batch_index,:],tf.stop_gradient(grad_pot_all_samples1[:,batch_index,:]),tf.stop_gradient(grad_pot_all_samples2[:,batch_index,:]),g_direction)
        
        _, maxsksd_sum_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(1, tf_compute_max_SKSD(state_final1[:,0,:],state_final2[:,0,:],tf.stop_gradient(grad_pot_all_samples1[:,0,:]),tf.stop_gradient(grad_pot_all_samples2[:,0,:]),g_direction)))
        return maxsksd_sum_final/input_data_batch_size
    
  
    def getParams(self):
        return self.lfstep_size_raw,  self.raw_inflation,self.g
    
    def getInflation(self):
        return self.inflation
    def getRawInflation(self):
        return self.raw_inflation
    def getg(self):
        return self.g
    def getlfstep(self):
        return self.lfstep_size_raw
