#!/usr/bin/env python3
"""
Crop
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image
    Args:
        image: 3-D tf.Tensor (H, W, C)
        size:  tuple (h, w, c) giving the desired crop size
    Returns:
        3-D tf.Tensor of shape (h, w, c)
    """
    h, w, c = size
    cropped = tf.image.random_crop(image, (h, w))
    cropped.set_shape((h, w, c))
    return cropped
