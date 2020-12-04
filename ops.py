import tensorflow as tf
from tensorflow import keras


class Conv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding="SAME"):
        super(Conv2D, self).__init__()
        self.conv_op = keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           use_bias=False,
                                           kernel_initializer="he_normal")

    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding="SAME"):
        super(UpConv2D, self).__init__()
        self.up_conv_op = keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=False,
                                                 kernel_initializer="he_normal")

    def call(self, inputs, **kwargs):
        return self.up_conv_op(inputs)


class BatchNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.batch_norm = keras.layers.BatchNormalization(epsilon=epsilon,
                                                          axis=axis,
                                                          momentum=momentum)

    def call(self, inputs, **kwargs):
        return self.batch_norm(inputs)



class LayerNorm(keras.layers.LayerNormalization):
    def __init__(self, epsilon=1e-4, axis=-1):
        super(LayerNorm, self).__init__()
        self.layer_norm = keras.layers.LayerNormalization(epsilon=epsilon, axis=axis)

    def call(self, inputs, **kwargs):
        return self.layer_norm(inputs)


class LeakyReLU(keras.layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__()
        self.leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, **kwargs):
        return self.leaky_relu(inputs)


class AdamOptWrapper(tf.optimizers.Adam):
    def __init__(self, learning_rate=1e-4, beta_1=0., beta_2=0.9, epsilon=1e-4, amsgrad=False, **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                                             epsilon=epsilon, amsgrad=amsgrad, **kwargs)


def d_loss_fn(fake_logit, real_logit):
    fake_loss = tf.reduce_mean(fake_logit)
    real_loss = tf.reduce_mean(real_logit)
    return fake_loss - real_loss


def g_loss_fn(fake_logit):
    fake_loss = - tf.reduce_mean(fake_logit)
    return fake_loss