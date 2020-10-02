import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from decoder.mlp import mlp_layer


def deconv_layer(output_shape, filter_shape, activation, strides, name):
    W = tf.get_variable(shape=filter_shape, initializer=tf.contrib.layers.xavier_initializer(), name=name + '_W')  # use output channel
    b = tf.get_variable(shape=(filter_shape[-2],),initializer=tf.zeros_initializer, name=name + '_b')  # use output channel

    def apply_train(x):
        output_shape_x = (x.get_shape().as_list()[0],) + output_shape
        a = slim.nn.conv2d_transpose(x, W, output_shape_x, strides, 'SAME') + b
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'softplus':
            return tf.nn.softplus(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a
    
    def apply_not_train(x):
        output_shape_x = (x.get_shape().as_list()[0],) + output_shape
        a = slim.nn.conv2d_transpose(x, tf.stop_gradient(W), output_shape_x, strides, 'SAME') + tf.stop_gradient(b)
        if activation == 'relu':
            return tf.nn.relu(a)
        if activation == 'softplus':
            return tf.nn.softplus(a)
        if activation == 'sigmoid':
            return tf.nn.sigmoid(a)
        if activation == 'linear':
            return a

    return apply_train, apply_not_train

def generator_train(dimH=500, dimZ=32, name='generator'):
    with tf.variable_scope("vae_decoder"):
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
        for i in range(N_layers):
            name_layer = name + '_mlp_l%d' % l
            mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i + 1], 'relu', name_layer)[0])
            l += 1

        conv_layers = []
        N_layers = len(decoder_input_shape) - 1
        for i in range(N_layers):
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
                                            strides, name_layer)[0])
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
    
def generator_not_train(dimH=500, dimZ=32, name='generator'):
    with tf.variable_scope("vae_decoder") as scope:
        scope.reuse_variables()
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
        for i in range(N_layers):
            name_layer = name + '_mlp_l%d' % l
            mlp_layers.append(mlp_layer(fc_layers[i], fc_layers[i + 1], 'relu', name_layer)[1])
            l += 1

        conv_layers = []
        N_layers = len(decoder_input_shape) - 1
        for i in range(N_layers):
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
                                            strides, name_layer)[1])
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

def get_decoder_param():
    return tuple(tf.trainable_variables("vae_decoder"))

if __name__ == '__main__':
    generator_train()
    a = get_decoder_param()
    print(len(a))
