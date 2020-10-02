import tensorflow as tf
import tensorflow.contrib.slim as slim

from decoder.mlp import mlp_layer


def conv_layer(filter_shape, activation, strides, name):
    W = slim.variable(shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(), name=name + '_W')
    b = tf.Variable(tf.zeros([filter_shape[-1]]), name=name + '_b')

    def apply(x):
        a = slim.nn.conv2d(x, W, strides, 'SAME') + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'softplus':
            return tf.nn.softplus(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply


def construct_filter_shapes(layer_channels, filter_width=5):
    filter_shapes = []
    for n_channel in layer_channels:
        shape = (n_channel, filter_width, filter_width)
        filter_shapes.append(shape)
    return filter_shapes


def encoder_convnet(input_shape, dimH=500, dimZ=32, name='conv_encoder'):
    with tf.variable_scope("vae_encoder"):
        # encoder for z (low res)
        layer_channels = [input_shape[-1], 16, 32, 32]
        filter_width = 5
        fc_layer_sizes = [dimH]
        conv_layers = []
        N_layers = len(layer_channels) - 1
        strides = (1, 2, 2, 1)
        activation = 'relu'
        l = 0

        print_shapes = []
        for i in range(N_layers):
            name_layer = name + '_conv_l%d' % l
            filter_shape = (filter_width, filter_width, layer_channels[i], layer_channels[i + 1])
            print_shapes.append(filter_shape)
            conv_layers.append(conv_layer(filter_shape, activation, strides, name_layer))
            l += 1

        # fc_layer = [int(np.prod(filter_shape)), dimH, dimZ * 2]
        fc_layer = [512, dimH, dimZ * 2]
        enc_mlp = []
        for i in range(len(fc_layer) - 1):
            if i + 2 < len(fc_layer):
                activation = 'relu'
            else:
                activation = 'linear'
            name_layer = name + '_mlp_l%d' % l
            enc_mlp.append(mlp_layer(fc_layer[i], fc_layer[i + 1], activation, name_layer)[0])
            l += 1

        print('encoder architecture', print_shapes, 'reshape', fc_layer)

        def apply(x):
            out = x
            for layer in conv_layers:
                out = layer(out)
            out = tf.reshape(out, (out.get_shape().as_list()[0], -1))
            for layer in enc_mlp:
                out = layer(out)
            mu, log_sig = tf.split(out, 2, axis=1)
            return mu, log_sig

        return apply


def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())

def get_encoder_param():
    return tuple(tf.trainable_variables("vae_encoder"))

if __name__ == '__main__':
    input_batch_size = 64
    encoder_convnet(input_shape=(input_batch_size, 28, 28, 1))
    a = get_encoder_param()
    print(len(a))
