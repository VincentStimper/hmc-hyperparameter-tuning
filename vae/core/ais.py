import tensorflow as tf
import numpy as np

from core.ham import hmc_kernel
from util.constant import log_2pi


def sigmoid_schedule(num, rad=4):
    """The sigmoid schedule defined in Section 6.2 of the paper. This is defined as:
          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),
    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    if num == 1:
        return [np.asarray(0.0), np.asarray(1.0)]
    t = np.linspace(-rad, rad, num)
    sigm = 1. / (1. + np.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())


def LogMeanExp(A, axis=None):
    """
    Compute the log of average of input array A.

    :param A: the log weights of size K
    :param axis: the axis of summation
    :return: log sum(exp(A)) - log K
    """
    A_max = tf.reduce_max(A, axis=axis, keepdims=True)
    B = (
            tf.log(tf.reduce_mean(tf.exp(A - A_max), axis=axis)) +
            A_max
    )
    return B[0]


def ais_step(pot_pre, pot_new, sample_pre, log_weight_pre, mcmc_kernel):
    """
    Compute log Z(p*) = \sum log_avg_exp(log w_T(x_i)) + log Z(p_0).
    It includes two steps:
    1) update w(x_i) by the ratio of two adjacent annealing distribution
    log w_{t}(x_i^t) = log w_{t-1}(x_i^{t-1}) + log p^*_{t}(x_i^{t-1}) - log p^*_{t-1}(x_i^{t-1})
    2) update the sample by HMC transition
    x_i^t ~ T_{HMC}(x, x_{i}^{t-1}).

    :param pot_pre: the potential function of p_{t-1}
    :param pot_new: the potential function of p_{t}(x_i^{t-1})
    :param sample_pre: the batch of samples x^{t-1}
    :param log_weight_pre: the batch of previous log weights log w_{t-1}(x^{t-1})
    :param mcmc_kernel: MCMC kernel for updating sample batch x_{t-1}
    :return: sample_new: updated sample batch x_{t}
             log_weight_new: updated log weight batch log w_{t}(x^t)
             acp_flag: boolean array for acceptance of new samples
             w_update: the log weight update log p^*_{t}(x_i^{t-1}) - log p^*_{t-1}(x_i^{t-1})
    """
    w_update = -pot_new(sample_pre) + pot_pre(sample_pre)  # log f_t(x_{t-1}) - log f_{t-1}(x_{t-1})
    log_weight_new = log_weight_pre + w_update
    sample_new, _, _, acp_flag = mcmc_kernel(pot_new, sample_pre)
    return sample_new, log_weight_new, acp_flag, w_update


def hais_gauss(pot_target, num_chains, input_batch_size, dim, num_scheduled_dists=1000, num_leaps=5, step_size=0.1, dtype=tf.float32):
    sample_init = tf.random_normal(shape=(num_chains, input_batch_size, dim), dtype=dtype)
    pot_init = lambda x: 0.5 * tf.reduce_sum(x ** 2.0, axis=-1)
    log_init_partition = 0.5 * log_2pi * dim
    schedule_np = sigmoid_schedule(num=num_scheduled_dists)
    print(schedule_np[0], schedule_np[-1])
    schedule = tf.constant(schedule_np, dtype=dtype)
    return hais(pot_target=pot_target, pot_init=pot_init, sample_init=sample_init,
                log_init_partition=log_init_partition, schedule=schedule, num_leaps=num_leaps, step_size=step_size,
                dtype=dtype)


def hais(pot_target, pot_init, sample_init, log_init_partition, schedule, num_leaps=5, step_size=0.1,
         dtype=tf.float32):
    """
    This is an implementation of the Hamiltonian annealed importance sampling (HAIS) based on the paper
    "On the Quantitative Analysis of Decoder-based Generative Models" from Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov,
    and Roger Grosse.
    It simply follows equation (4) and (5) in the paper.

    The goal of HAIS is to construct unbiased estimation of the partition constant of the target distribution p(x).
    Given the potential function h(x), the unnormalised density function is defined as
    p*(x) = exp(-h(x))
    and the log partition function is
    log Z(p*) = log \int exp(-h(x)) dx \approx log_avg_exp(-h(x_i)),
    where x_i is sampled from p(x).
    log Z(p*) is estimated by HAIS by
    log Z(p*) = \sum log_avg_exp(log w_T(x_i)) + log Z(p_0),
    where w(x_i) is updated by the ratio of two adjacent annealing distribution
    log w_{t}(x_i^t) = log w_{t-1}(x_i^{t-1}) + log p^*_{t}(x_i^{t-1}) - log p^*_{t-1}(x_i^{t-1})
    and the sample is updated by HMC transition
    x_i^t ~ T_{HMC}(x, x_{i}^{t-1}).

    :param pot_target: the negative unnormalised target density fucntion with
    input shape (sample_batch_size, sample_dim) and output shape (sample_batch_size, )
    :param pot_init: the negative unnormalised initial density fucntion with
    input shape (sample_batch_size, sample_dim) and output shape (sample_batch_size, )
    :param sample_init: the samples from initial distribution, each sample corresponds to one HMC chain
    :param log_init_partition: the log partition of initial distribution
    :param schedule: the sequence of scheduled temperature of annealing distributions
    :param num_leaps: the number of leapfrog steps in Hamiltonian Monte Carlo transition for annealing distributions
    :param step_size: the number of leapfrog step size in Hamiltonian Monte Carlo transition for annealing distributions
    :param dtype: default data type
    :return: log_partition: the estimated log partition
             log_weights: the log weights of samples
             sample: the generated samples at the end of HAIS
             acp_rate: averaged acceptance rate of HMC transition over all annealing distributions
    """
    pot_tmp = lambda beta, z: (1 - beta) * pot_init(z) + beta * pot_target(z)  #   num_chains* input_batch       output dim: num_chains x pot_dim
    mcmc_kernel = lambda pot_fun, sample_pre: hmc_kernel(pot_fun, sample_pre, num_leaps=num_leaps, step_size=step_size)
    log_weights_init = tf.zeros(shape=(tf.shape(sample_init)[0], tf.shape(sample_init)[1]), dtype=dtype)  # num_chains * input_batch      dim: num_chains x pot_dim
    loop_vars = (1, sample_init, log_weights_init, tf.cast(tf.tile(tf.constant(np.array([1.0])), (sample_init.shape[1],)),tf.float32))
    num_dists = tf.shape(schedule)[0]

    cond = lambda dist_id, sample_init, log_weights_init, acp_rate: tf.less(dist_id, num_dists)

    def _loop_body(dist_id, samples, log_weights, acp_rate):
        beta_pre = schedule[dist_id - 1]
        beta = schedule[dist_id]

        pot_fun_pre = lambda z: pot_tmp(beta_pre, z)
        pot_fun_i = lambda z: pot_tmp(beta, z)
        sample_new, log_weight_new, acp_flag, _ = ais_step(pot_fun_pre, pot_fun_i, samples, log_weights, mcmc_kernel)
        acp_rate = (acp_rate * (tf.cast(dist_id, tf.float32) - 1) + tf.reduce_mean(acp_flag,0)) / tf.cast(dist_id,
                                                                                   tf.float32)
        return dist_id + 1, sample_new, log_weight_new, acp_rate

    _, sample, log_weights, acp_rate = tf.while_loop(cond=cond, body=_loop_body, loop_vars=loop_vars)
    log_partition = LogMeanExp(log_weights, axis=0) + log_init_partition   # dim: input_batch
    return log_partition, log_weights, sample, acp_rate
