#!/usr/bin/env python3
"""
Contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image
    Args:
        image:3D tf.Tensor representing the input image to adjust the contrast
        lower: A float representing the lower bound of the random contrast.
        upper: A float representing the upper bound of the random contrast.
    Returns:
        the contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
