from tqdm import tqdm

import cv2

import tensorflow as tf


keys_to_features = {"image": tf.compat.v1.FixedLenFeature([], tf.string)}


def _parse_fn(data_record, in_size=80, out_size=64):
    features = keys_to_features
    sample = tf.compat.v1.parse_single_example(data_record, features)

    image = tf.image.decode_jpeg(sample["image"])
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape(shape=[in_size, in_size, 3])
    image = tf.image.random_crop(image, size=[out_size, out_size, 3])
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=5.)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    #image = tf.image.random_hue(image, max_delta=0.2)
    #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.divide(tf.subtract(image, 127.5 * tf.ones_like(image)), 127.5 * tf.ones_like(image))

    return image


def get_batch(tfrecord_path, batch_size, in_size=80, out_size=64):
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(lambda x: _parse_fn(x, in_size=in_size, out_size=out_size))
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset = dataset.repeat(num_epochs)

    return dataset


#datasets = get_batch(r"D:\Anime_Face_Dataset\tfrecord\anime_face_dataset.tfrecord", 1, 1)
#
#for image in tqdm(datasets):
#    image = tf.cast(tf.add(tf.multiply(image, 127.5 * tf.ones_like(image)), 127.5 * tf.ones_like(image)), dtype=tf.uint8)
#    image = image[0].numpy()[:, :, ::-1]
#    cv2.imshow("Anh", image)
#    cv2.waitKey(100)