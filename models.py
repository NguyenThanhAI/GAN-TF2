import tempfile
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import ops

from utils import img_merge, pbar, save_image_grid

AUTOTUNE = tf.data.experimental.AUTOTUNE


class WGANGPConfig:
    def __init__(self, z_dim, epochs, batch_size, image_size, n_critic, grad_penalty_weight, total_images, g_lr, d_lr,
                 n_samples):
        self.z_dim = z_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_critic = n_critic
        self.grad_penalty_weight = grad_penalty_weight
        self.total_images = total_images
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_samples = n_samples

class WGANGP:
    def __init__(self, config: WGANGPConfig):
        self.config = config
        self.g_opt = ops.AdamOptWrapper(learning_rate=self.config.g_lr)
        self.d_opt = ops.AdamOptWrapper(learning_rate=self.config.d_lr)
        self.G = self.build_generator()
        self.D = self.build_discriminator()

        self.G.summary()
        self.D.summary()

    def train(self, dataset):
        z = tf.constant(tf.random.normal((self.config.n_samples, 1, 1, self.config.z_dim)))
        g_train_loss = keras.metrics.Mean()
        d_train_loss = keras.metrics.Mean()

        for epoch in range(self.config.epochs):
            bar = pbar(self.config.total_images, self.config.batch_size, epoch, self.config.epochs)
            for batch in dataset:
                for _ in range(self.config.n_critic):
                    self.train_d(batch)
                    d_loss = self.train_d(batch)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix["g_loss"] = f"{g_train_loss.result():6.3f}"
                bar.postfix["d_loss"] = f"{d_train_loss.result():6.3f}"
                bar.update(self.config.batch_size)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            samples = self.generate_samples(z)
            image_grid = img_merge(samples, n_rows=8).squeeze()
            save_image_grid(image_grid, epoch + 1)

    @tf.function
    def train_g(self):
        z = tf.random.normal((self.config.batch_size, 1, 1, self.config.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = ops.g_loss_fn(fake_logits)

        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    @tf.function
    def train_d(self, x_real):
        z = tf.random.normal((self.config.batch_size, 1, 1, self.config.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            real_logits = self.D(x_real, training=True)
            cost = ops.d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), x_real, x_fake)
            cost += self.config.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    def gradient_penalty(self, f, real, fake):
        alpha = tf.random.uniform([self.config.batch_size, 1, 1, 1], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def generate_samples(self, z):
        return self.G(z, training=False)

    def build_generator(self):
        dim = self.config.image_size
        mult = dim // 8

        x = inputs = keras.layers.Input((1, 1, self.config.z_dim))
        x = ops.Conv2D(1024, 1, 1, "valid")(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(512)(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(256)(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(128)(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(64)(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(32)(x)
        x = ops.BatchNorm()(x)
        x = keras.layers.Activation("relu")(x)

        x = ops.UpConv2D(3)(x)
        x = keras.layers.Activation("tanh")(x)
        return keras.models.Model(inputs, x, name="Generator")

    def build_discriminator(self):
        dim = self.config.image_size
        mult = 1
        i = dim // 2

        x = inputs = keras.layers.Input((dim, dim, 3))
        x = ops.Conv2D(dim)(x)
        x = ops.LeakyReLU()(x)

        while i > 4:
            x = ops.Conv2D(dim * (2 * mult))(x)
            x = ops.LayerNorm(axis=[1, 2, 3])(x)
            x = ops.LeakyReLU()(x)

            i //= 2
            mult *= 2

        x = ops.Conv2D(1, 4, 1, "valid")(x)
        return keras.models.Model(inputs, x, name="Discriminator")


class DatasetPipeline:
    def __init__(self, dataset, epochs, batch_size, image_size, crop):
        self.dataset_name = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.crop = crop
        self.dataset_info = {}

    def preprocess_image(self, image):
        if self.crop:
            image = tf.image.central_crop(image, 0.50)
        image = tf.image.resize(image, (self.image_size, self.image_size), antialias=True)
        image = (tf.dtypes.cast(image, tf.float32) / 127.5) - 1.0
        return image

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def load_dataset(self):
        ds, self.dataset_info = tfds.load(name=self.dataset_name,
                                          split=tfds.Split.ALL,
                                          with_info=True)
        ds = ds.map(lambda x: self.preprocess_image(x["image"]), AUTOTUNE)
        ds = self.dataset_cache(ds)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
        return ds
