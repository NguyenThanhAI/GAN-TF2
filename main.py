import os
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import control_flow_util

import models

from read_tfrecord import get_batch

keras.backend.clear_session()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--g_lr", type=float, default=0.0001)
    parser.add_argument("--d_lr", type=float, default=0.0001)
    parser.add_argument("--tfrecord_path", type=str, default=r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord")
    parser.add_argument("--crop", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--g_penalty", type=float, default=10.)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=64)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    print("Args: {}".format(args))
    total_images = sum(1 for _ in tf.data.TFRecordDataset(args.tfrecord_path))
    dataset = get_batch(tfrecord_path=args.tfrecord_path, batch_size=args.batch_size)

    config = models.WGANGPConfig(z_dim=args.z_dim, epochs=args.epochs, batch_size=args.batch_size,
                                 image_size=args.image_size, n_critic=args.n_critic, grad_penalty_weight=args.g_penalty,
                                 total_images=total_images, g_lr=args.g_lr, d_lr=args.d_lr, n_samples=args.n_samples)

    wgangp = models.WGANGP(config=config)
    wgangp.train(dataset=dataset)
