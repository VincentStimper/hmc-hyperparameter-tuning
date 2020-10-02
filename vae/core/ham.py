import tensorflow as tf

def gaussian_kinetic(r):
    with tf.name_scope("K"):
        return 0.5 * tf.reduce_sum(r ** 2, axis=-1)


def ham_energy(state, momentum, pot_fun, kin_fun):
    with tf.name_scope("ham_energy"):
        return pot_fun(state) + kin_fun(momentum)


def __leapfrog_step_pre(x, r, pot_fun, eps=0.1, stop_gradient_pot=True):
    with tf.name_scope("LF_step_pre"):
        potential = pot_fun(x)
        if stop_gradient_pot:
            r_half = r - 0.5 * eps * tf.stop_gradient(tf.gradients(ys=potential, xs=x)[0])
        else:
            r_half = r - 0.5 * eps * tf.gradients(ys=potential, xs=x)[0]
        return x, r_half


def __leapfrog_step_loop(x, r, pot_fun, eps=0.1, r_var=1.0, stop_gradient_pot=True):
    with tf.name_scope("LF_step"):
        x_new = x + eps * r / r_var
        potential = pot_fun(x_new)
        if stop_gradient_pot:
            r_new = r - eps * tf.stop_gradient(tf.gradients(ys=potential, xs=x_new)[0])
        else:
            r_new = r - eps * tf.gradients(ys=potential, xs=x_new)[0]
        return x_new, r_new


def __leapfrog_step_post(x, r, pot_fun, eps=0.1, r_var=1.0, stop_gradient_pot=True):
    with tf.name_scope("LF_step_post"):
        x_new = x + eps * r / r_var
        potential = pot_fun(x_new)
        if stop_gradient_pot:
            r_new = r - 0.5 * eps * tf.stop_gradient(tf.gradients(ys=potential, xs=x_new)[0])
        else:
            r_new = r - 0.5 * eps * tf.gradients(ys=potential, xs=x_new)[0]
        return x_new, r_new


def leapfrog_noMH(x, r, pot_fun, eps=0.1, numleap=3, r_var=1.0, back_prop=False, stop_gradient_pot=True):
    # no MH correction
    with tf.name_scope("LF"):
        x_new, r_new = __leapfrog_step_pre(x, r, pot_fun, eps, stop_gradient_pot)
        i = tf.constant(0)
        numleap_clean = tf.abs(tf.maximum(1, tf.constant(value=numleap, dtype=tf.int32)))
        condition = lambda i, phase: tf.less(i, numleap_clean)
        loopbody = lambda i, phase: (i + 1, __leapfrog_step_loop(phase[0], phase[1], pot_fun, eps, r_var=r_var,
                                                                 stop_gradient_pot=stop_gradient_pot))
        _, (x_new_interm, r_new_interm) = tf.while_loop(cond=condition,
                                                        body=loopbody,
                                                        loop_vars=(i, (x_new, r_new)),
                                                        back_prop=back_prop)
        return __leapfrog_step_post(x_new_interm, r_new_interm, pot_fun, eps, r_var=r_var,
                                    stop_gradient_pot=stop_gradient_pot)

def leapfrog(x, r, pot_fun, eps=0.1, numleap=3, r_var=1.0, back_prop=False, stop_gradient_pot=True):
    with tf.name_scope("LF"):
        x_new, r_new = __leapfrog_step_pre(x, r, pot_fun, eps, stop_gradient_pot)
        i = tf.constant(0)
        numleap_clean = tf.abs(tf.maximum(1, tf.constant(value=numleap, dtype=tf.int32)))
        condition = lambda i, phase: tf.less(i, numleap_clean)
        loopbody = lambda i, phase: (i + 1, __leapfrog_step_loop(phase[0], phase[1], pot_fun, eps, r_var=r_var,
                                                                 stop_gradient_pot=stop_gradient_pot))
        _, (x_new_interm, r_new_interm) = tf.while_loop(cond=condition,
                                                        body=loopbody,
                                                        loop_vars=(i, (x_new, r_new)),
                                                        back_prop=back_prop)
        x_new_end, r_new_end =  __leapfrog_step_post(x_new_interm, r_new_interm, pot_fun, eps, r_var=r_var,
                                    stop_gradient_pot=stop_gradient_pot)
        pot_init = pot_fun(x)
        kin = tf.reduce_sum(0.5* r**2, axis=-1)
        pot_end = pot_fun(x_new_end)
        kin_end = tf.reduce_sum(0.5*r_new_end**2,axis=-1)
        dH=pot_init+kin-(pot_end+kin_end)
        acp=tf.minimum(1.0,tf.exp(dH))
        acp_flag = acp>tf.random_uniform(shape = tf.shape(acp))
        x_accepted = tf.where(tf.tile(tf.expand_dims(acp_flag,2),(1,1,x.shape[-1])),x_new_end,x)
        return x_accepted, r_new_end

def hmc_kernel(pot_fun, x_init, num_leaps, step_size, dtype=tf.float32):
    pot_init = pot_fun(x_init)
    momentum = tf.random_normal(shape=tf.shape(x_init), dtype=dtype)
    kin = tf.reduce_sum(0.5 * momentum ** 2, axis=-1)
    x_new, momentum_new = leapfrog_noMH(x_init, momentum, pot_fun, step_size, num_leaps, r_var=1.0)
    pot_new = pot_fun(x_new)
    kin_new = tf.reduce_sum(0.5 * momentum_new ** 2, axis=-1)
    dH = pot_init + kin - (pot_new + kin_new)
    acp = tf.minimum(1.0, tf.exp(dH))
    acp_flag = acp > tf.random_uniform(shape=tf.shape(acp))  # num_chains* input_batch
    x_accpeted = tf.where(tf.tile(tf.expand_dims(acp_flag, 2), (1,1,x_init.shape[-1])), x_new, x_init)
    U_accepted = pot_fun(x_accpeted)
    return x_accpeted, U_accepted, dH, tf.cast(acp_flag, tf.float32)

"""
def hmc_sampler(pot_fun, sample_init, num_samples, burn_in=1000, num_leaps=20, step_size=0.05, dtype=tf.float32):
    initializer = (sample_init, tf.ones(shape=[1, ], dtype=tf.float32))
    elems = tf.range(0, num_samples+burn_in)

    def _loop_body(state, iter_num):
        sample_new, _, _, acp_flag = hmc_kernel(pot_fun, state[0], num_leaps=num_leaps, step_size=step_size,
                                                dtype=dtype)
        return sample_new, acp_flag

    state_final = tf.scan(_loop_body, elems=elems, initializer=initializer)
    samples = state_final[0][burn_in:]
    acp_flags = state_final[1][burn_in:]
    return samples, acp_flags
"""
def hmc_sampler(pot_fun, sample_init, num_samples, burn_in=1000, num_leaps=20, step_size=0.05, dtype=tf.float32):
    cond = lambda layer_index, state, acp: tf.less(layer_index, num_samples)

    def _loopbody(layer_index, state, acp):
        state_new, _, _, acp_new = hmc_kernel(pot_fun,state,num_leaps,step_size,dtype = tf.float32)
        return layer_index + 1, state_new, acp + acp_new

    _, state_final, acp_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, sample_init, tf.zeros(shape=[sample_init.shape[0],sample_init.shape[1]])))
    avg_acp = acp_final/num_samples
    return state_final, avg_acp
