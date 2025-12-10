#!/usr/bin/env python3
"""
Rotate
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise
    Args:
        image is a 3D tf.Tensor containing the image to rotate
    Returns:
        the rotated image
    """
    return tf.image.rot90(image)
