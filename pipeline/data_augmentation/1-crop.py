#!/usr/bin/env python3
"""
Performs a random crop of an image
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image
    :param image: is a 3D tf.Tensor containing the image to crop
    :param size: is a tuple containing the size of the crop
    :return: the cropped image
    """
    return tf.image.random_crop(image, size=size)
