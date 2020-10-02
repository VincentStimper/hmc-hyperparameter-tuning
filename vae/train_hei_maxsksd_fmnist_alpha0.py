import numpy as np
import tensorflow as tf
import json
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

from core.hamInfnet_hei_maxsksd import HamInfNetHEI as HamInfNet
from decoder.vae_dcnn_mnist import VAE_DCNN_GPU, VAEQ_CONV
from util.constant import log_2pi
#from util.utils import dybinarize_mnist
from util.utils import binarise_fashion_mnist


def training_setting(z_dim):
    setting = {'mb_size': 128,
               'alpha': 0.0,     # alpha=0-divergence
               'z_dim': z_dim,
               'h_dim': 500,  
               'X_mnist_dim': 28 ** 2,
               'momentum_train_batch_size': 1,
               'z_train_sample_batch_size': 30,  
               'num_layers': 30,  
               'num_lfsteps': 5,
               'momentum_std': 1.0,
               'generator': 'dcnn_relu',
               'batches': 95000,  
               'dybin': True,
               'reg': 0.000001,
               'lr': 0.0002, 
               'lr-decay': 0.97,
               }
    return setting


def train(setting, dataset, dataset_name='mnist', save_model=False, device='CPU', dtype=tf.float32):
    mb_size = setting['mb_size']
    alpha = setting['alpha']
    z_dim = setting['z_dim']  
    h_dim = setting['h_dim']  
    X_mnist_dim = setting['X_mnist_dim']  
    momentum_train_batch_size = setting['momentum_train_batch_size']  
    z_train_sample_batch_size = setting['z_train_sample_batch_size']  
    generator = setting['generator']
    num_layers = setting['num_layers']
    num_lfsteps = setting['num_lfsteps'] 
    momentum_std = setting['momentum_std'] 
    batches = setting['batches'] 
    dybin = setting['dybin']  
    reg = setting['reg']
    lr = setting['lr']
    lr_decay = setting['lr-decay']
    if 'vfun' in setting.keys():
        vfun = setting['vfun']
    else:
        vfun = 'sigmoid'

    bin_label = 'dybin'
    if not dybin:
        bin_thresh = setting['bin_thresh']
        bin_label = 'stbin'

    setting = {'mb_size': mb_size,
               'alpha': alpha,
               'z_dim': z_dim,
               'h_dim': h_dim,
               'X_mnist_dim': X_mnist_dim,
               'momentum_train_batch_size': momentum_train_batch_size,
               'z_train_sample_batch_size': z_train_sample_batch_size,
               'num_layers': num_layers,
               'num_lfsteps': num_lfsteps,
               'momentum_std': momentum_std,
               'generator': generator,
               'vfun': vfun,
               'batches': batches,
               'dybin': dybin,
               'reg': reg,
               'lr': lr,
               'lr-decay': lr_decay,
               }
    if not dybin:
        setting['bin_thresh'] = bin_thresh
    model_name = "vae_{}-hei-{}-alpha{:.0e}-zd{}-hd{}-mbs{}-mbn{}-h{}-l{}-reg{:.0e}-{}-lr{:.0e}".format(generator,
                                                                                              dataset_name,
                                                                                              alpha,
                                                                                              z_dim,
                                                                                              h_dim, mb_size,
                                                                                              batches,
                                                                                              num_layers,
                                                                                              num_lfsteps,
                                                                                              reg, bin_label, lr)
    output_dir = "model/fashion_hei_maxsksd_alpha1/"

    if save_model:
        #os.mkdir(output_dir + '{}'.format(model_name))
        ckpt_name = output_dir + '{}.ckpt'.format(model_name)
        #setting_filename = output_dir + '{}/setting.json'.format(model_name)
        setting_filename = output_dir + 'setting.json'.format(model_name)
        with open(setting_filename, 'w') as f:
            json.dump(setting, f)
        
            

    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step2 = tf.Variable(0, trainable=False, name='global_step2')
    global_step3 = tf.Variable(0, trainable=False, name='global_step3')
    global_step4 = tf.Variable(0, trainable=False, name='global_step4')


    device_config = '/device:{}:0'.format(device)
    with tf.device(device_config):
        X_batch_train = tf.placeholder(dtype, shape=[mb_size, X_mnist_dim])
        vaeq = VAEQ_CONV(alpha = alpha, h_dim=h_dim, z_dim=z_dim)
        vae = VAE_DCNN_GPU(h_dim=h_dim, z_dim=z_dim)

        def gen_fun_train_hmc(sample_batch_size, input_data_batch_size ,inflation):
            mu, log_var = vaeq.Q(X_batch_train)
            mu_nograd = tf.stop_gradient(mu)  
            log_var_nograd = tf.stop_gradient(log_var)   
            inflation = tf.stop_gradient(inflation)
            eps = tf.random_normal(shape=(sample_batch_size, input_data_batch_size, z_dim))
            logp = -0.5 * tf.reduce_sum((inflation*eps) ** 2 + log_2pi + 2*tf.log(inflation) + log_var_nograd, axis=-1)
            return eps * (inflation* tf.exp(log_var_nograd / 2)) + mu_nograd, logp

        def gen_fun_train_ksd(sample_batch_size, input_data_batch_size ,inflation):
            mu, log_var = vaeq.Q(X_batch_train)
            mu_nograd = tf.stop_gradient(mu)
            log_var_nograd = tf.stop_gradient(log_var)
            eps = tf.random_normal(shape=(sample_batch_size, input_data_batch_size, z_dim))
            logp = -0.5 * tf.reduce_sum((inflation*eps) ** 2 + log_2pi + 2*tf.log(inflation) + log_var_nograd, axis=-1)
            return eps * (inflation* tf.exp(log_var_nograd / 2)) + mu_nograd, logp

        pot_fun_train = lambda state: vae.pot_fun_train(data_x=X_batch_train, sample_z=state)
        pot_fun_not_train = lambda state: vae.pot_fun_not_train(data_x=X_batch_train, sample_z=state)

        hamInfNet_hm = HamInfNet(num_layers=num_layers, num_lfsteps=num_lfsteps,
                                 sample_dim=z_dim, dtype=dtype)

        neg_pot, recon_mean = hamInfNet_hm.build_elbo_graph(
            pot_fun=pot_fun_train,
            state_init_gen=gen_fun_train_hmc,
            input_data_batch_size=mb_size,
            sample_batch_size=1,
            training=True
        )
        pot_batch_mean = -neg_pot    # pot_batch_mean is neg-log-lik
        
        maxsksd_mean = hamInfNet_hm.build_maxsksd_graph(
            pot_fun=pot_fun_not_train,
            state_init_gen=gen_fun_train_ksd,
            input_data_batch_size=mb_size,
            sample_batch_size=z_train_sample_batch_size,
            training=True
        )
        
        q_loss1 = vaeq.create_loss_train(vae, X_batch_train, batch_size = 1, loss_only=True)
        
        
        q_loss2 = vaeq.create_loss_not_train(vae, X_batch_train, batch_size = 1,loss_only=True)
        
        infl = hamInfNet_hm.getInflation()
        g=hamInfNet_hm.getg()
        raw_inflation=hamInfNet_hm.getRawInflation()
        raw_lfstep=hamInfNet_hm.getlfstep()
        
        loss = q_loss2 + pot_batch_mean + reg * (vae.get_parameters_reg() + vaeq.get_parameters_reg())

        starter_learning_rate = lr
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                     1000, lr_decay, staircase=True)

        learning_rate2 = tf.train.exponential_decay(starter_learning_rate, global_step2,
                     1000, lr_decay, staircase=True)
        learning_rate3 = tf.train.exponential_decay(starter_learning_rate, global_step3,
                     1000, lr_decay, staircase=True)
        learning_rate4 = tf.train.exponential_decay(starter_learning_rate, global_step4,
                     1000, lr_decay, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(q_loss1, global_step=global_step)
        
        optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
        train_op2 = optimizer2.minimize(loss, global_step=global_step2)
        
        optimizer3 = tf.train.AdamOptimizer(learning_rate=learning_rate3)
        train_op3 = optimizer3.minimize(maxsksd_mean,var_list=[raw_lfstep,raw_inflation], global_step=global_step3)
        
        optimizer4 = tf.train.AdamOptimizer(learning_rate=learning_rate4)
        train_op4 = optimizer4.minimize(-maxsksd_mean,var_list=[g], global_step=global_step4)

        
        
    loss_seq = []
    loss_q_seq = []
    pot_seq = []
    recon_seq = []
    maxsksd_seq = []
    saved_variables = vae.get_parameters() + hamInfNet_hm.getParams() + vaeq.get_parameters()
    saver = tf.train.Saver(saved_variables, max_to_keep=10)
    print(saved_variables)
    
    log = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_batch = 5000 #1000
        total_time = 0
        
        for i in np.arange(0,100000):  # pretrain encoder+decoder using stadard VAE/IWAE training scheme
            X_mb_raw,_=dataset.train.next_batch(mb_size)
            #X_mb=dybinarize_mnist(X_mb_raw)
            X_mb = binarise_fashion_mnist(X_mb_raw)
            start=time.time()
            _,q_loss1_i,inflation_i=sess.run([train_op,q_loss1,infl],feed_dict={X_batch_train:X_mb})

        
            end = time.time()
            total_time += end - start
        
            loss_q_seq.append(q_loss1_i)
            
            if i % 1000 ==999:
                log_line = 'iter: {}, q_loss: {},inflation:{}, time: {}'.format(i+1, np.mean(np.asarray(loss_q_seq)),
                                                                                     inflation_i,
                                                                                     total_time)
                print(log_line)
                log.append(log_line + '\n')
                
            
                loss_q_seq.clear()
                
                
                total_time = 0
            if i % checkpoint_batch == 4999:
                #with open(output_dir + '{}/training.cklog'.format(model_name), "a+") as log_file:
                with open(output_dir + 'training.cklog', "a+") as log_file:
                    log_file.writelines(log)
                    log.clear()
        
        for i in np.arange(0, batches):
            X_mb_raw, _ = dataset.train.next_batch(mb_size)
            #X_mb = dybinarize_mnist(X_mb_raw)
            X_mb = binarise_fashion_mnist(X_mb_raw)
            

            start = time.time()
            
            _,_,_, loss_i, q_loss_i, maxsksd_mean_i, pot_mean_i, recon_mean_i,inflation_i = sess.run(
                [train_op2,train_op3,train_op4, loss, q_loss2, maxsksd_mean, pot_batch_mean, recon_mean, infl],
                feed_dict={X_batch_train: X_mb})
        
            pot_seq.append(pot_mean_i)
            end = time.time()
            total_time += end - start
            loss_seq.append(loss_i)
            loss_q_seq.append(q_loss_i)
            maxsksd_seq.append(maxsksd_mean_i)
            recon_seq.append(recon_mean_i)
            #if i % 10 ==9:
            if True:
                                                                                    
                log_line = 'iter: {}, loss: {}, q_loss: {}, maxsksd_loss: {}, pot: {}, recon: {}, inflation:{}, time: {}'.format(i + 1,
                                                                                     np.mean(np.array(loss_seq)),
                                                                                     np.mean(np.array(loss_q_seq)),
                                                                                     np.mean(np.array(maxsksd_seq)),
                                                                                     np.mean(np.array(pot_seq)),
                                                                                     np.mean(np.array(recon_seq)),
                                                                                     inflation_i,
                                                                                     total_time)
                
                print(log_line)
                log.append(log_line + '\n')
                loss_seq.clear()
                pot_seq.clear()
                loss_q_seq.clear()
                maxsksd_seq.clear()
                recon_seq.clear()
                total_time = 0
            if save_model and i % checkpoint_batch == 4999:
                print("model saved at iter: {}".format(i + 1))
                saver.save(sess, ckpt_name, global_step=global_step2)
                #with open(output_dir + '{}/training.cklog'.format(model_name), "a+") as log_file:
                with open(output_dir + 'training.cklog', "a+") as log_file:
                    log_file.writelines(log)
                    log.clear()
        if save_model:
            saver.save(sess, ckpt_name, global_step=global_step2)
            #with open(output_dir + '{}/training.cklog'.format(model_name), "a+") as log_file:
            with open(output_dir + 'training.cklog', "a+") as log_file:
                log_file.writelines(log)
                log.clear()
    #return output_dir + '{}/'.format(model_name)
    return output_dir


if __name__ == '__main__':
    mnist = input_data.read_data_sets('data/fashion_mnist', one_hot=True)
    train(setting=training_setting(32), dataset=mnist, save_model=True, device="GPU") # 32