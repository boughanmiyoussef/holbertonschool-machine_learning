#!/usr/bin/env python3
import tensorflow as tf

def crop_image(image, size):
    """
    Performs a random crop of an image
    Args:
        image is a 3D tf.Tensor containing the image to crop
        size is a tuple containing the size of the crop
    Returns:
        the cropped image
    """
    h, w, c = size
    crop = tf.image.stateless_random_crop(
        image, (h, w, c), seed=tf.constant([1, 1])
    )
    return crop