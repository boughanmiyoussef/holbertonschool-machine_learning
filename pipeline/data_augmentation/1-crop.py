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
    # Unpack size
    crop_height, crop_width, channels = size
    
    # Get image shape
    image_shape = tf.shape(image)
    img_height, img_width = image_shape[0], image_shape[1]
    
    # Calculate maximum offset
    max_offset_height = tf.maximum(0, img_height - crop_height)
    max_offset_width = tf.maximum(0, img_width - crop_width)
    
    # Generate random offsets
    offset_height = tf.random.uniform(
        shape=[], minval=0, maxval=max_offset_height + 1, dtype=tf.int32)
    offset_width = tf.random.uniform(
        shape=[], minval=0, maxval=max_offset_width + 1, dtype=tf.int32)
    
    # Perform crop
    cropped = image[
        offset_height:offset_height + crop_height,
        offset_width:offset_width + crop_width,
        :
    ]
    
    # If crop dimensions don't match requested (edge case), resize
    cropped_shape = tf.shape(cropped)
    if crop_height != cropped_shape[0] or crop_width != cropped_shape[1]:
        cropped = tf.image.resize(cropped, (crop_height, crop_width))
    
    return cropped
