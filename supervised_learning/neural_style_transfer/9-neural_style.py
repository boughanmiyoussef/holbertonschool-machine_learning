#!/usr/bin/env python3
"""
Neural Style Transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs Neural Style Transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        self.style_image = self.scale_image(style_image)

        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.beta = beta

        self.model = None
        self.load_model()
        self.gram_style_features, self.content_feature = self.generate_features()

    @staticmethod
    def scale_image(image):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        resized = tf.image.resize(image, [256, 512], method='bicubic')
        normalized = resized / 255.0
        clipped = tf.clip_by_value(normalized, 0, 1)
        return tf.expand_dims(clipped, axis=0)

    def load_model(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers + [self.content_layer]]
        model = tf.keras.Model(vgg.input, outputs)

        config = model.get_config()
        for layer in config['layers']:
            if layer['class_name'] == 'MaxPooling2D':
                layer['class_name'] = 'AveragePooling2D'
        self.model = tf.keras.Model.from_config(config)

    @staticmethod
    def gram_matrix(input_layer):
        _, h, w, c = input_layer.shape
        flattened = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(tf.transpose(flattened), flattened)
        return tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)

    def generate_features(self):
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(preprocessed_style)
        content_output = style_outputs[-1]
        style_features = [NST.gram_matrix(layer) for layer in style_outputs[:-1]]

        return style_features, content_output

    def style_cost(self, style_outputs):
        weight = 1.0 / len(self.style_layers)
        return sum(weight * tf.reduce_mean(tf.square(s - t)) for s, t in zip(style_outputs, self.gram_style_features))

    def content_cost(self, content_output):
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        inputs = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(inputs)
        style_outputs, content_output = outputs[:-1], outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J_total = self.alpha * J_content + self.beta * J_style

        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)
        grad = tape.gradient(J_total, generated_image)
        return grad, J_total, J_content, J_style

    def generate_image(self, iterations=2000, step=None, lr=0.002, beta1=0.9, beta2=0.99):
        # Input validation
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float) or not (0 <= beta1 <= 1):
            raise TypeError("beta1 must be a float in [0, 1]")
        if not isinstance(beta2, float) or not (0 <= beta2 <= 1):
            raise TypeError("beta2 must be a float in [0, 1]")

        generated_image = tf.Variable(self.content_image)
        best_cost = float('inf')
        best_image = self.content_image.numpy().copy()

        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(generated_image)
            optimizer.apply_gradients([(grads, generated_image)])
            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image.numpy()
            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}")

        best_image = best_image[0]
        best_image = np.clip(best_image, 0, 1).astype(np.float32)

        return best_image, best_cost