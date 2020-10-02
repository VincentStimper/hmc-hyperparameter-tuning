from abc import abstractmethod

import tensorflow as tf
import math

from decoder.vae_helper import sigmoid_cross_entroy_loss
from util.constant import log_2pi


class VAE_ABC_GPU:
    def __init__(self, dtype, afun='sigmoid'):
        self.dtype = dtype
        self.afun = afun

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_parameters_reg(self):
        pass

    def log_p_z(self, z):  # normal z prior   logP(z)
        return -0.5 * tf.reduce_sum(z ** 2, axis=-1, keepdims=True) - 0.5 * self.z_dim * log_2pi

    @abstractmethod
    def z_to_logits_train(self, z):  # Your favorite NN comes here
        pass
    
    @abstractmethod
    def z_to_logits_not_train(self, z):  # Your favorite NN comes here
        pass

    def nlog_px_z_train(self, z):
        logits = self.z_to_logits_train(z)
        if self.afun == 'sigmoid':
            prob = tf.nn.sigmoid(logits)
            return prob, logits
        elif self.afun == 'gaussian':
            mu = tf.nn.sigmoid(logits)
            log_var = 2*tf.log(0.1)
            return mu, log_var

    def nlog_pD_z_train(self, data_x, z):  # -log p(D | z)
        if self.afun == 'sigmoid':
            _, logits = self.nlog_px_z_train(z)
            return sigmoid_cross_entroy_loss(labels=data_x, logits=logits)
        elif self.afun == 'gaussian':
            mu, log_var = self.nlog_px_z_train(z)
            return 0.5*tf.reduce_sum((data_x - mu)**2*tf.exp(-log_var) + log_2pi + log_var, axis=-1)


    def pot_fun_train(self, data_x, sample_z):   # -log_lik
        recon_loss = tf.reduce_sum(self.nlog_pD_z_train(data_x, sample_z), axis=-1)
        prior_loss = tf.reduce_sum(-self.log_p_z(sample_z), axis=-1)
        pot = recon_loss + prior_loss
        return pot
    
    def nlog_px_z_not_train(self, z):
        logits = self.z_to_logits_not_train(z)
        if self.afun == 'sigmoid':
            prob = tf.nn.sigmoid(logits)
            return prob, logits
        elif self.afun == 'gaussian':
            mu = tf.nn.sigmoid(logits)
            log_var = 2*tf.log(0.1)
            return mu, log_var

    def nlog_pD_z_not_train(self, data_x, z):  # -log p(D | z)
        if self.afun == 'sigmoid':
            _, logits = self.nlog_px_z_not_train(z)
            return sigmoid_cross_entroy_loss(labels=data_x, logits=logits)
        elif self.afun == 'gaussian':
            mu, log_var = self.nlog_px_z_not_train(z)
            return 0.5*tf.reduce_sum((data_x - mu)**2*tf.exp(-log_var) + log_2pi + log_var, axis=-1)


    def pot_fun_not_train(self, data_x, sample_z):   # -log_lik
        recon_loss = tf.reduce_sum(self.nlog_pD_z_not_train(data_x, sample_z), axis=-1)
        prior_loss = tf.reduce_sum(-self.log_p_z(sample_z), axis=-1)
        pot = recon_loss + prior_loss
        return pot
    
    
    
    
    
    
"""
    def pot_fun_debug(self, data_x, sample_z):
        recon_loss = tf.reduce_sum(self.nlog_pD_z(data_x, sample_z), axis=-1)
        prior_loss = tf.reduce_sum(-self.log_p_z(sample_z), axis=-1)
        pot = recon_loss + prior_loss
        return pot, recon_loss, prior_loss

    def _pot_fun_fake(self, data_x, sample_z):
        recon_loss = tf.reduce_sum(self.nlog_pD_z(data_x, sample_z), axis=-1)
        prior_loss = tf.reduce_sum(-self.log_p_z(sample_z), axis=-1)
        pot = tf.zeros_like(recon_loss) + prior_loss
        return pot

    def nlog_pDz(self, data_x, z):
        return self.pot_fun(data_x, z)

    def pot_fun_gen(self, batch_size):
        z_neg = tf.random_normal(shape=[batch_size, self.z_dim])
        prob, _ = self.nlog_px_z(z_neg)
        X_mb_binary_train_neg = prob > tf.random_uniform(shape=[batch_size, self.num_vis])
        X_mb_binary_train_neg_type_adj = tf.cast(X_mb_binary_train_neg, dtype=self.dtype)
        pot_fun_gen = self.pot_fun(X_mb_binary_train_neg_type_adj, z_neg)
        return pot_fun_gen

    ############################# Sanity check version of important functions ############################

    def log_p_z2(self, z):  # normal z prior (check version)
        return tf.reduce_sum(self.log_gaussian_prob(z), axis=-1, keepdims=True)

    def nlog_pD_z2(self, data_x, z):  # log p(D | z) (check version)
        prob, _ = self.nlog_px_z(z)
        return -self.log_bernoulli_prob(data_x, p=prob)

    def log_bernoulli_prob(self, x, p=0.5):
        logprob = x * tf.log(tf.clip_by_value(p, 1e-9, 1.0)) \
                  + (1 - x) * tf.log(tf.clip_by_value(1.0 - p, 1e-9, 1.0))
        return logprob

    def log_gaussian_prob(self, x, mu=0.0, log_sig=0.0):
        logprob = -(0.5 * tf.log(2 * math.pi) + log_sig) \
                  - 0.5 * ((x - mu) / tf.exp(log_sig)) ** 2
        return logprob

    def pot_fun2(self, data_x, sample_z):
        recon_loss = tf.reduce_sum(self.nlog_pD_z2(data_x, sample_z), axis=-1)
        prior_loss = tf.reduce_sum(-self.log_p_z2(sample_z), axis=-1)
        pot = recon_loss + prior_loss
        return pot
"""
