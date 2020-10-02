import numpy as np
import tensorflow as tf


def deconv_layer(output_shape, filter_shape, activation, strides, name):
    scale = 1.0 / np.prod(filter_shape[:3])
    seed = int(np.random.randint(0, 1000))  # 123
    with tf.name_scope('conv_mnist/conv'):
        W = tf.Variable(tf.random_uniform(filter_shape,
                                      minval=-scale, maxval=scale,
                                      dtype=tf.float32, seed=seed), name = name+ '_W')
        b = tf.Variable(tf.zeros([filter_shape[-2]]), name=name + '_b')  # use output channel

    def apply(x):
        output_shape_x = (x.get_shape().as_list()[0],) + output_shape
        a = tf.nn.conv2d_transpose(x, W, output_shape_x, strides, 'SAME') + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply


def generator(dimH=500, dimZ=32, name='generator'):
    # now construct a decoder
    input_shape = (28, 28, 1)
    filter_width = 5
    decoder_input_shape = [(4, 4, 32), (7, 7, 32), (14, 14, 16)]
    decoder_input_shape.append(input_shape)
    fc_layers = [dimZ, dimH, int(np.prod(decoder_input_shape[0]))]
    l = 0
    # first include the MLP
    mlp_layers = []
    N_layers = len(fc_layers) - 1
    for i in np.arange(0, N_layers):
        name_layer = name + '_mlp_l%d' % l
        mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i + 1], 'relu', name_layer))
        l += 1

    conv_layers = []
    N_layers = len(decoder_input_shape) - 1
    for i in np.arange(0, N_layers):
        if i < N_layers - 1:
            activation = 'relu'
        else:
            activation = 'linear'
        name_layer = name + '_conv_l%d' % l
        output_shape = decoder_input_shape[i + 1]
        input_shape = decoder_input_shape[i]
        up_height = int(np.ceil(output_shape[0] / float(input_shape[0])))
        up_width = int(np.ceil(output_shape[1] / float(input_shape[1])))
        strides = (1, up_height, up_width, 1)
        filter_shape = (filter_width, filter_width, output_shape[-1], input_shape[-1])

        conv_layers.append(deconv_layer(output_shape, filter_shape, activation, \
                                        strides, name_layer))
        l += 1

    print('decoder architecture', fc_layers, 'reshape', decoder_input_shape)

    def apply(z):
        x = z
        for layer in mlp_layers:
            x = layer(x)
        x = tf.reshape(x, (x.get_shape().as_list()[0],) + decoder_input_shape[0])
        for layer in conv_layers:
            x = layer(x)
        return x

    return apply


def init_weights(input_size, output_size, constant=1.0, seed=123):
    """ Glorot and Bengio, 2010's initialization of network weights"""
    scale = constant * np.sqrt(6.0 / (input_size + output_size))
    if output_size > 0:
        return tf.random_uniform((input_size, output_size),
                                 minval=-scale, maxval=scale,
                                 dtype=tf.float32, seed=seed)
    else:
        return tf.random_uniform([input_size],
                                 minval=-scale, maxval=scale,
                                 dtype=tf.float32, seed=seed)


def mlp_layer(d_in, d_out, activation, name):
    with tf.name_scope('conv_mnist/mlp'):
        W = tf.Variable(init_weights(d_in, d_out), name=name + '_W')
        b = tf.Variable(tf.zeros([d_out]), name=name + '_b')

    def apply_layer(x):
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply_layer

def get_parameters():
    return tf.trainable_variables('conv_mnist')

################################## Conv Encoder ##############################

def conv_layer(filter_shape, activation, strides, name):
    scale = 1.0 / np.prod(filter_shape[:3])
    seed = int(np.random.randint(0, 1000))  # 123
    W = tf.Variable(tf.random_uniform(filter_shape,
                                      minval=-scale, maxval=scale,
                                      dtype=tf.float32, seed=seed), name=name + '_W')
    b = tf.Variable(tf.zeros([filter_shape[-1]]), name=name + '_b')

    def apply(x):
        a = tf.nn.conv2d(x, W, strides, 'SAME') + b
        if activation == 'relu':
            return tf.nn.relu(a)
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
    fc_layer = [512, dimH, dimZ*2]
    print(fc_layer)
    enc_mlp = []
    for i in range(len(fc_layer) - 1):
        if i + 2 < len(fc_layer):
            activation = 'relu'
        else:
            activation = 'linear'
        name_layer = name + '_mlp_l%d' % l
        enc_mlp.append(mlp_layer2(fc_layer[i], fc_layer[i + 1], activation, name_layer))
        print(fc_layer[i], fc_layer[i + 1])
        l += 1

    print('encoder architecture', print_shapes, 'reshape', fc_layer)

    def apply(x):
        out = x
        for layer in conv_layers:
            out = layer(out)
            print(out)
        out = tf.reshape(out, (out.get_shape().as_list()[0], -1))
        print(out)
        for layer in enc_mlp:
            out = layer(out)
        mu, log_sig = tf.split(out, 2, axis=1)
        return mu, log_sig

    return apply

def mlp_layer2(d_in, d_out, activation, name):
    with tf.name_scope('conv_mnist/mlp2'):
        W = tf.Variable(init_weights(d_in, d_out), name=name + '_W')
        b = tf.Variable(tf.zeros([d_out]), name=name + '_b')

    def apply_layer(x):
        a = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply_layer


def sample_gaussian(mu, log_sig):
    return mu + tf.exp(log_sig) * tf.random_normal(mu.get_shape())
