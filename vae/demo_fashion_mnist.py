import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from core.hamInfnet_hei_nn import HamInfNetNN
from data import load_iwae_binarised_mnist_dataset
from decoder.vae_dcnn_mnist import VAE_DCNN_GPU, VAEQ_CONV, VAE_DCNN
from util.utils import plot_mnist


from data import load_fashion_mnist_binary_test



def fashion_mnist_preprocess(X):
    X_mean = tf.reduce_mean(X, axis=-1, keepdims=True)
    X_m = X - X_mean
    return X_m

def load_setting(model_path):
    setting_filename = model_path + 'setting.json'
    if setting_filename:
        with open(setting_filename, 'r') as f:
            setting = json.load(f)
            print('Restored setting from {}'.format(setting_filename))
    print(setting)
    return setting

def load_model(model_path, mb_size=36, dtype=tf.float32):
    setting = load_setting(model_path)
    z_dim = setting['z_dim']  # 50
    h_dim = setting['h_dim']

    num_layers = setting['num_layers']
    num_lfsteps = setting['num_lfsteps']
    generator = setting['generator']
    print(generator)
    if 'vfun' in setting.keys():
        vfun = setting['vfun']
    else:
        vfun = 'sigmoid'
        
    vae_decoder = VAE_DCNN(h_dim=h_dim, z_dim=z_dim)
    vae_decoder_pot = VAE_DCNN_GPU(h_dim=h_dim, z_dim=z_dim, gen=vae_decoder.get_generator(), afun=vfun)

    vae_encoder = VAEQ_CONV(alpha=0.0,z_dim=z_dim, h_dim=h_dim) # value of alpha doesn't matter here 
    hamInfNet_hm = HamInfNetNN(num_layers=num_layers,
                               num_lfsteps=num_lfsteps,
                               sample_dim=z_dim,
                               dtype=dtype)
    return vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm

def demo(model_path, dataset=None, device="CPU", dtype=tf.float32):
    mb_demo_size = 36
    demo_size = (6, 6)
    setting = load_setting(model_path)
    X_mnist_dim = setting['X_mnist_dim']
    z_dim = setting['z_dim']

    device_config = '/device:{}:0'.format(device)
    tf.reset_default_graph()

    with tf.device(device_config):
        X_batch_demo = tf.placeholder(dtype, shape=[mb_demo_size, X_mnist_dim])
        vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm = load_model(model_path, mb_size=mb_demo_size)

        q0_mean, q0_logvar = vae_encoder.Q(X_batch_demo)
        
        z_prior = tf.random_normal(shape=(mb_demo_size, z_dim), dtype=dtype)
        vae_decoder = load_model(model_path)

        PX_demo_gen = vae_decoder.nlog_px_z(z_prior)[0]
        
        pot_fun = lambda z: vae_decoder_pot.pot_fun_train(data_x=X_batch_demo, sample_z=z)
        z_recon_samples = hamInfNet_hm.build_simulation_graph(pot_fun=pot_fun, mean=q0_mean, logvar=q0_logvar,
                                                              sample_batch_size=1,
                                                              input_data_batch_size=mb_demo_size)   # sample* input* latent
        PX_demo_recon = vae_decoder.nlog_px_z(tf.reduce_mean(z_recon_samples, axis=0))[0]
        


    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored.")
        

        for _ in np.arange(0, 5):
            
            X_test, _ = dataset.test.next_batch(mb_demo_size)
            X_gen, PX_demo_recon_samples= sess.run([PX_demo_gen, PX_demo_recon], feed_dict={X_batch_demo: X_test})
            plot_mnist(X_test, size=(6, 6))
            plot_mnist(X_gen, size=(6, 6))
            plot_mnist(PX_demo_recon_samples, size=(6, 6))

            plt.show()

if __name__ == '__main__':
    fashion_mnist = load_fashion_mnist_binary_test()
    model_path = "model/fmnist_hei_maxsksd_alpha1/"
    demo(model_path, fashion_mnist)
