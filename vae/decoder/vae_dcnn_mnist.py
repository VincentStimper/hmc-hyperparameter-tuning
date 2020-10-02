import tensorflow as tf

from decoder.vae_conv_decoder import generator_train,generator_not_train, get_decoder_param
from decoder.vae_conv_encoder import encoder_convnet, get_encoder_param
from decoder.vae_gen_gpu import VAE_ABC_GPU
from decoder.vae_helper import sigmoid_cross_entroy_loss
from util.constant import log_2pi

class VAE_DCNN:
    def __init__(self, h_dim=500, z_dim=32, afun='sigmoid'):
        self.num_vis = 28**2
        self.z_dim = 32
        self.h_dim = 500
        self.dcnn_train = generator_train(dimH=h_dim, dimZ=z_dim)
        self.afun = afun

    def get_parameters(self):
        return get_decoder_param()

    def get_parameters_reg(self):
        reg = tf.zeros(shape=())
        for var in self.get_parameters():
            reg += tf.reduce_sum(var ** 2)
        return reg

    def get_parameters_l2_list(self):
        l2_list = []
        for var in self.get_parameters():
            l2_list.append((tf.reduce_sum(var ** 2), tf.shape(var)))
        return l2_list

    def get_generator(self):
        return self.dcnn_train

    def log_p_z(self, z):  # normal z prior
        return -0.5 * tf.reduce_sum(z ** 2, axis=1) - 0.5 * self.z_dim * log_2pi

    def nlog_px_z(self, z):  # negative log p(X|z)
        logits = tf.reshape(self.dcnn_train(z), shape=(tf.shape(z)[0], self.num_vis))
        if self.afun == 'gaussian':
            mu = tf.nn.sigmoid(logits)
            log_var = 2*tf.log(0.1)
            return mu, log_var
        else:
            prob = tf.nn.sigmoid(logits)
            return prob, logits

    def nlog_pD_z(self, data_x, z):  # neg log p(D | z)
        if self.afun == 'gaussian':
            mu, log_var = self.nlog_px_z(z)
            return 0.5*tf.reduce_sum((data_x - mu)**2*tf.exp(-log_var) + log_2pi + log_var, axis=-1)
        else:
            _, logits = self.nlog_px_z(z)
            return sigmoid_cross_entroy_loss(labels=data_x, logits=logits)

    def pot_fun(self, data_x, sample_z):
        return tf.reduce_sum(self.nlog_pD_z(data_x, sample_z), axis=1) - self.log_p_z(sample_z)


class VAEQ_CONV:
    def __init__(self, alpha, z_dim=32, h_dim=500):
        self.num_vis = 28**2
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.encoder = encoder_convnet(input_shape=(None, 28, 28, 1), dimH=h_dim, dimZ=z_dim)
        self.alpha = alpha

    def get_parameters(self):
        return get_encoder_param()

    def get_parameters_reg(self):
        reg = tf.zeros(shape=())
        for var in self.get_parameters():
            reg += tf.reduce_sum(var ** 2)
        return reg

    def reshape_x_batch(self, X):
        return tf.reshape(X, shape=(tf.shape(X)[0], 28, 28, 1))

    def Q(self, X, reshape_X=True):
        if reshape_X:
            X_reshaped = self.reshape_x_batch(X)
        else:
            X_reshaped = X
        mu, log_var = self.encoder(X_reshaped)
        log_var = tf.log(tf.exp(log_var) + 0.0000001)
        return mu, log_var

    def sample_z(self, mu, log_var, batch_size=1):
        eps = tf.random_normal(shape=tf.shape(mu))  # input_batch * latent_dim
        if batch_size > 1:
            eps = tf.random_normal(shape=(batch_size, tf.shape(mu)[0], tf.shape(mu)[1]))  # sample_batch * input_batch* latent_dim
        return mu + tf.exp(log_var / 2) * eps   #sample_batch * input_batch* latent_dim

    def sample_z_given_x(self, X, reshape_X=True):
        if reshape_X:
            X_reshaped = self.reshape_x_batch(X)
        else:
            X_reshaped = X
        mu, log_var = self.Q(X_reshaped)
        return self.sample_z(mu, log_var)

    def create_loss_not_train(self, vae, X_batch, batch_size=1, loss_only=True):
        z_mu, z_logvar = self.Q(X=X_batch)   # input_batch * latent_dim
        sample_z_from_X_batch = self.sample_z(z_mu, z_logvar, batch_size)  #sample_batch * input_batch* latent_dim
        
        recon_loss = tf.reduce_sum(vae.nlog_pD_z_not_train(data_x=X_batch, z=sample_z_from_X_batch), axis=1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, axis=1)
        vae_loss = tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_loss)
        
        # pot is -logP*(z|D) = -logP(z,D)
        # pot_fun_not_train avoids training decoder params (decoder params are trained by maximizing E-log-target instead)
        pot = vae.pot_fun_not_train(data_x=X_batch, sample_z=sample_z_from_X_batch)  # sample_batch* input_batch
        
        # log_q_z is log(q(z|D))
        log_q_z = -0.5*tf.reduce_sum((sample_z_from_X_batch  - z_mu)**2 *tf.exp(-z_logvar) + log_2pi + z_logvar, axis=-1)  # sample_batch* input_batch
        log_w = -pot - log_q_z    # sample_batch * input_batch
        
        

        #DReG_alpha-divergence
        alpha_div_term1 = (tf.stop_gradient((tf.exp(log_w - tf.reduce_logsumexp(log_w, axis=0)))**2))* log_w
        alpha_div_term2 = (tf.stop_gradient(tf.exp(self.alpha* log_w - tf.reduce_logsumexp(self.alpha* log_w, axis =0)))) *log_w
        DReG_mean = self.alpha*(self.alpha* tf.reduce_mean(tf.reduce_sum(alpha_div_term1, axis=0)) + (1.-self.alpha)* tf.reduce_mean(tf.reduce_sum(alpha_div_term2, axis=0)))
        
        if self.alpha > 1e-6:
            return -DReG_mean
        else:
            return vae_loss   # For standard vi (alpha = 0)
        
        
    
    def create_loss_train(self, vae, X_batch, batch_size=1, loss_only=True):
        z_mu, z_logvar = self.Q(X=X_batch)   # input_batch * latent_dim
        sample_z_from_X_batch = self.sample_z(z_mu, z_logvar, batch_size)  #sample_batch * input_batch* latent_dim
        
        recon_loss = tf.reduce_sum(vae.nlog_pD_z_train(data_x=X_batch, z=sample_z_from_X_batch), axis=1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu ** 2 - 1. - z_logvar, axis=1)
        vae_loss = tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_loss)
        
        # pot is -logP*(z|D) = -logP(z,D)
        # pot_fun_train allows training of decoder params (for vanilla VAE/IWAE):
        
        pot = vae.pot_fun_train(data_x=X_batch, sample_z=sample_z_from_X_batch)  # sample_batch* input_batch
        
        #pot_mean = tf.reduce_mean(pot)
        
        # log_q_z is log(q(z|D))
        
        log_q_z = -0.5*tf.reduce_sum((sample_z_from_X_batch  - z_mu)**2 *tf.exp(-z_logvar) + log_2pi + z_logvar, axis=-1)  # sample_batch* input_batch
        log_w = -pot - log_q_z    # sample_batch * input_batch
        
        
        #DReG_alpha-divergence
        alpha_div_term1 = (tf.stop_gradient((tf.exp(log_w - tf.reduce_logsumexp(log_w, axis=0)))**2))* log_w
        alpha_div_term2 = (tf.stop_gradient(tf.exp(self.alpha* log_w - tf.reduce_logsumexp(self.alpha* log_w, axis =0)))) *log_w
        DReG_mean = self.alpha*(self.alpha* tf.reduce_mean(tf.reduce_sum(alpha_div_term1, axis=0)) + (1.-self.alpha)* tf.reduce_mean(tf.reduce_sum(alpha_div_term2, axis=0)))
        
        if self.alpha > 1e-6:
            return -DReG_mean
        else:
            return vae_loss   # For standard vi (alpha = 0)
        

class VAE_DCNN_GPU(VAE_ABC_GPU):
    def __init__(self, h_dim=500, z_dim=32, gen=None, afun='sigmoid'):
        self.z_dim = z_dim
        self.h_dim = h_dim
        if gen is None:
            self.dcnn_train = generator_train(dimH=h_dim, dimZ=z_dim)
        else:
            self.dcnn_train = gen
            
        if gen is None:
            self.dcnn_not_train = generator_not_train(dimH=h_dim, dimZ=z_dim)
        else:
            self.dcnn_not_train = gen
            
        self.dtype = tf.float32
        self.afun = afun

    def get_parameters(self):
        return get_decoder_param()

    def get_parameters_reg(self):
        reg = tf.zeros(shape=())
        for var in self.get_parameters():
            reg += tf.reduce_sum(var ** 2)
        return reg

    def z_to_logits_train(self, z):
        z_shape = tuple(z.shape.as_list())
        dim_keep = z_shape[0:len(z_shape)-1]
        dim_keep_prod = 1
        for dim in dim_keep:
            dim_keep_prod *= dim
        z_reshaped = tf.reshape(z, shape=(dim_keep_prod, self.z_dim))
        print(z_reshaped)
        logits = self.dcnn_train(z_reshaped)
        return tf.reshape(logits, shape=dim_keep+(-1,))
    
    def z_to_logits_not_train(self, z):
        z_shape = tuple(z.shape.as_list())
        dim_keep = z_shape[0:len(z_shape)-1]
        dim_keep_prod = 1
        for dim in dim_keep:
            dim_keep_prod *= dim
        z_reshaped = tf.reshape(z, shape=(dim_keep_prod, self.z_dim))
        print(z_reshaped)
        logits = self.dcnn_not_train(z_reshaped)
        return tf.reshape(logits, shape=dim_keep+(-1,))

