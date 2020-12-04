import os
import argparse

from functools import partial

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

import module

from read_tfrecord import get_batch

import tqdm

import skimage.color as color
import skimage.transform as transform
import skimage.io as iio


rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale


def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).
    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).
    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img


def _check(images, dtypes, min_value=-np.inf, max_value=np.inf):
    # check type
    assert isinstance(images, np.ndarray), '`images` should be np.ndarray!'

    # check dtype
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
    assert images.dtype in dtypes, 'dtype of `images` shoud be one of %s!' % dtypes

    # check nan and inf
    assert np.all(np.isfinite(images)), '`images` contains NaN or Inf!'

    # check value
    if min_value not in [None, -np.inf]:
        l = '[' + str(min_value)
    else:
        l = '(-inf'
        min_value = -np.inf
    if max_value not in [None, np.inf]:
        r = str(max_value) + ']'
    else:
        r = 'inf)'
        max_value = np.inf
    assert np.min(images) >= min_value and np.max(images) <= max_value, \
        '`images` should be in the range of %s!' % (l + ',' + r)


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    _check(images, [np.float32, np.float64], -1.0, 1.0)
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, im2uint(image), quality=quality, **plugin_args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_path", type=str, default=r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--n_d', type=int, default=1)  # # d updates per g update
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    return args

def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)

    return gp


if __name__ == '__main__':
    args = get_args()

    if args.output_dir == None:
        output_dir = "output"
    else:
        output_dir = args.output_dir

    summary_dir = os.path.join(output_dir, "summaries")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)

    sample_dir = os.path.join(output_dir, "samples_training")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)

    video_dir = os.path.join(output_dir, "video")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)


    @tf.function
    def train_G():
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
            x_fake = G(z, training=True)
            x_fake_d_logit = D(x_fake, training=True)
            G_loss = g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, G.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

        return {'g_loss': G_loss}


    @tf.function
    def train_D(x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
            x_fake = G(z, training=True)

            x_real_d_logit = D(x_real, training=True)
            x_fake_d_logit = D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = gradient_penalty(partial(D, training=True), x_real, x_fake,
                                  mode="wgan-gp")

            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

        D_grad = t.gradient(D_loss, D.trainable_variables)
        D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


    @tf.function
    def sample(z):
        return G(z, training=False)

    dataset = get_batch(tfrecord_path=args.tfrecord_path, batch_size=args.batch_size)
    num_samples = sum(1 for _ in tf.data.TFRecordDataset(args.tfrecord_path))

    G = module.ConvGenerator(input_shape=(1, 1, args.z_dim), output_channels=3, n_upsamplings=4, name="G_anime")
    D = module.ConvDiscriminator(input_shape=(64, 64, 3), n_downsamplings=4, norm="layer_norm", name="D_anime")

    G.summary()
    D.summary()

    d_loss_fn, g_loss_fn = get_wgan_losses_fn()

    G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
    D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)

    summary_writer = tf.summary.create_file_writer(logdir=summary_dir)

    fps = 10.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(os.path.join(video_dir, "video.mp4"), fourcc, fps, (640, 640))

    z = tf.random.normal((100, 1, 1, args.z_dim))
    with summary_writer.as_default():
        for epoch in tqdm.tgrange(args.epochs, desc="Epoch loop"):

            print("Epoch: {}/{}".format(epoch + 1, args.epochs))

            for x_real in tqdm.tqdm(dataset, desc="Inner epoch loop", total=int(num_samples / args.batch_size)):
                d_loss_dict = train_D(x_real)
                tf.summary.scalar("d_loss", d_loss_dict["d_loss"], D_optimizer.iterations.numpy())
                tf.summary.scalar("gp", d_loss_dict["gp"], D_optimizer.iterations.numpy())

                if D_optimizer.iterations.numpy() % args.n_d == 0:
                    G_loss_dict = train_G()
                    tf.summary.scalar("g_loss", G_loss_dict["g_loss"], D_optimizer.iterations.numpy())

                if G_optimizer.iterations.numpy() % 100 == 0:
                    x_fake = sample(z)
                    tf.summary.image("real_image", x_real, D_optimizer.iterations.numpy())
                    tf.summary.image("fake_image", x_fake, D_optimizer.iterations.numpy())
                    img = immerge(x_fake, n_rows=10).squeeze()
                    imwrite(img, os.path.join(sample_dir, "iter-%09d.jpg" % G_optimizer.iterations.numpy()))
                    video_writer.write(im2uint(img[:, :, ::-1]))

    video_writer.release()
