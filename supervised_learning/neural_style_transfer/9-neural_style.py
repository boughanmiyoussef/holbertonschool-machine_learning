#!/usr/bin/env python3
"""Neural Style Transfer module"""
import tensorflow as tf
import numpy as np


class NST:
    """Neural Style Transfer class

    Attributes:
        style_layers (list): The pretrained selected layers for the style.
        content_layer (string): Represents the pretrained selected layer for
            the content.

    Raises:
        TypeError: If style_image is not np.ndarray with shape (h, w, 3)
        TypeError: If content_image is not np.ndarray with shape (h, w, 3)
        TypeError: If beta is a negative value.
        TypeError: If alpha is a negative value.

    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializer"""
        if not isinstance(style_image, np.ndarray)\
                or len(style_image.shape) != 3\
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(content_image, np.ndarray)\
                or len(content_image.shape) != 3\
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        # No need for tf.enable_eager_execution() in TF2.x

        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales the image such that pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Image of shape (h, w, 3)

        Returns:
            tf.Tensor: The scaled image with shape (1, new_h, new_w, 3)

        """
        if not isinstance(image, np.ndarray)\
                or len(image.shape) != 3\
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )
        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * new_h / h)
        else:
            new_w = 512
            new_h = int(h * new_w / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)  # Add batch dim

        image = tf.image.resize(image, [new_h, new_w], method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def load_model(self):
        """Creates the model used to calculate the cost"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = []
        for layer_name in self.style_layers:
            outputs.append(vgg.get_layer(layer_name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of the input layer.

        Args:
            input_layer (tf.Tensor): shape (1, h, w, c)

        Returns:
            tf.Tensor: Gram matrix shape (1, c, c)
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (1, h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = gram / tf.cast(h * w, tf.float32)
        return gram

    def generate_features(self):
        """Extract features of style and content images"""
        nb_layers = len(self.style_layers)

        style_img = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        content_img = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_img)
        content_outputs = self.model(content_img)

        self.gram_style_features = [NST.gram_matrix(style_outputs[i]) for i in range(nb_layers)]
        self.content_feature = content_outputs[nb_layers]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate style cost for one layer"""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(style_output.shape) != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, h, w, c = style_output.shape
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or gram_target.shape != (1, c, c):
            raise TypeError(f'gram_target must be a tensor of shape [1, {c}, {c}]')

        gram_style = NST.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculate style cost for generated image"""
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(f'style_outputs must be a list with length {length}')

        style_score = 0
        weight = 1 / length
        for target, output in zip(self.gram_style_features, style_outputs):
            style_score += weight * self.layer_style_cost(output, target)
        return style_score

    def content_cost(self, content_output):
        """Calculate content cost for generated image"""
        c_feature = self.content_feature
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or content_output.shape != c_feature.shape:
            raise TypeError(f'content_output must be a tensor of shape {c_feature.shape}')
        return tf.reduce_mean(tf.square(content_output - c_feature))

    def total_cost(self, generated_image):
        """Calculate total cost"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be a tensor of shape {self.content_image.shape}')

        processed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(processed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = self.alpha * J_content + self.beta * J_style
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Compute gradients for generated image"""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be a tensor of shape {self.content_image.shape}')

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        grads = tape.gradient(J_total, generated_image)
        return grads, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the styled image"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be positive')
        if step is not None and not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step is not None and (step < 0 or step > iterations):
            raise ValueError('step must be positive and less than iterations')
        if not isinstance(lr, (float, int)):
            raise TypeError('lr must be a number')
        if lr < 0:
            raise ValueError('lr must be positive')
        if not isinstance(beta1, float) or not 0 < beta1 < 1:
            raise ValueError('beta1 must be a float in (0, 1)')
        if not isinstance(beta2, float) or not 0 < beta2 < 1:
            raise ValueError('beta2 must be a float in (0, 1)')

        generated_image = tf.Variable(self.content_image, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             beta_1=beta1,
                                             beta_2=beta2)

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(generated_image)
            optimizer.apply_gradients([(grads, generated_image)])

            # Clip values between 0 and 1
            generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

            if step and i % step == 0:
                print(f"Iteration {i}")
                print(f"Total cost: {J_total.numpy()}")
                print(f"Content cost: {J_content.numpy()}")
                print(f"Style cost: {J_style.numpy()}")

        return generated_image.numpy()
