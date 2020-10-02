import tensorflow as tf

from decoder.vae_conv_util import encoder_convnet, get_parameters


def xavier_init(size, dtype):
    in_dim = size[0]
    xavier_stddev = tf.cast(1. / tf.sqrt(in_dim / 2.), dtype=dtype)
    return tf.random_normal(shape=size, stddev=xavier_stddev, dtype=dtype)


def sigmoid_cross_entroy_loss(logits, labels):
    """Computes sigmoid cross entropy given `logits`.

      Measures the probability error in discrete classification tasks in which each
      class is independent and not mutually exclusive.  For instance, one could
      perform multilabel classification where a picture can contain both an elephant
      and a dog at the same time.

      For brevity, let `x = logits`, `z = labels`.  The logistic loss is

            z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
          = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
          = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
          = (1 - z) * x + log(1 + exp(-x))
          = x - x * z + log(1 + exp(-x))

      For x < 0, to avoid overflow in exp(-x), we reformulate the above

            x - x * z + log(1 + exp(-x))
          = log(exp(x)) - x * z + log(1 + exp(-x))
          = - x * z + log(1 + exp(x))

      Hence, to ensure stability and avoid overflow, the implementation uses this
      equivalent formulation

          max(x, 0) - x * z + log(1 + exp(-abs(x)))

      `logits` and `labels` must have the same type and shape.

      Args:
        _sentinel: Used to prevent positional parameters. Internal, do not use.
        labels: A `Tensor` of the same type and shape as `logits`.
        logits: A `Tensor` of type `float32` or `float64`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        logistic losses.

      Raises:
        ValueError: If `logits` and `labels` do not have the same shape.
      """
    # The following code are commented out for broadcasting.
    # # pylint: disable=protected-access
    # nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,
    #                          labels, logits)
    # # pylint: enable=protected-access
    #
    # with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    #     logits = ops.convert_to_tensor(logits, name="logits")
    #     labels = ops.convert_to_tensor(labels, name="labels")
    #     try:
    #         labels.get_shape().merge_with(logits.get_shape())
    #     except ValueError:
    #         raise ValueError("logits and labels must have the same shape (%s vs %s)" %
    #                          (logits.get_shape(), labels.get_shape()))
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)
    return tf.add(
        relu_logits - logits * labels,
        tf.log1p(tf.exp(neg_abs_logits)))

class VAEQ:
    def __init__(self, z_dim, h_dim, num_vis, dtype, log_var=0.0):
        self.num_vis = num_vis
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.log_var = log_var
        self.Q_W1 = tf.get_variable(name="vaeq_zh_weights1", dtype=dtype,
                                    initializer=xavier_init([num_vis, h_dim], dtype))
        self.Q_b1 = tf.get_variable(name="vaeq_zh_bias1", dtype=dtype,
                                    initializer=tf.zeros(shape=(1, h_dim), dtype=dtype))

        self.Q_W2_mu = tf.get_variable(name="vaeq_hx_weights_2_mu", dtype=dtype,
                                    initializer=xavier_init([h_dim, z_dim], dtype))
        self.Q_b2_mu = tf.get_variable(name="vaeq_hx_bias2_mu", dtype=dtype,
                                    initializer=tf.zeros(shape=(1, z_dim), dtype=dtype))
        self.Q_W2_sigma = tf.get_variable(name="vaeq_hx_weights2_sigma", dtype=dtype,
                                       initializer=xavier_init([h_dim, z_dim], dtype))
        self.Q_b2_sigma = tf.get_variable(name="vaeq_hx_bias2_sigma", dtype=dtype,
                                       initializer=tf.zeros(shape=(1, z_dim), dtype=dtype))

    def get_parameters(self):
        return self.Q_W1, self.Q_b1, self.Q_W2_mu, self.Q_b2_mu, self.Q_W2_sigma, self.Q_b2_sigma

    def Q(self, X):
        h = tf.nn.relu(tf.matmul(X, self.Q_W1) + self.Q_b1)
        z_mu = tf.matmul(h, self.Q_W2_mu) + self.Q_b2_mu
        z_logvar = tf.matmul(h, self.Q_W2_sigma) + self.Q_b2_sigma
        return z_mu, z_logvar

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def sample_z_given_x(self, X):
        mu, log_var = self.Q(X)
        return self.sample_z(mu, log_var)


class VAEQ_CONV:
    def __init__(self, input_batch_size=64, z_dim=32, h_dim=500):
        self.num_vis = 28**2
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.encoder = encoder_convnet(input_shape=(input_batch_size, 28, 28, 1), dimH=h_dim, dimZ=z_dim)

    def get_parameters(self):
        return get_parameters()

    def Q(self, X):
        return self.encoder(X)

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def sample_z_given_x(self, X):
        mu, log_var = self.Q(X)
        return self.sample_z(mu, log_var)


if __name__ == '__main__':
    logits = tf.random_normal(shape=(1, 2, 3, 4))
    labels = tf.random_normal(shape=(3, 4))
    xe_loss = sigmoid_cross_entroy_loss(logits=logits, labels=labels)
    with tf.Session() as sess:
        print(sess.run(xe_loss))
