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

        # Enable eager execution (TF2.x has it enabled by default, so no need)
        # tf.compat.v1.enable_eager_execution()

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
            image (np.ndarray): Image to be scaled (h, w, 3).

        Returns:
            tf.Tensor: The scaled image tensor with shape (1, h_new, w_new, 3).

        """
        if not isinstance(image, np.ndarray)\
                or len(image.shape) != 3\
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize(image, [h_new, w_new], method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """Creates the model used to calculate the cost"""
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )

        # Replace MaxPooling with AveragePooling
        # Need to rebuild the model with AveragePooling
        # So save and reload with custom_objects is one way
        # Or simpler: rebuild manually

        # Build new model with AveragePooling2D layers replacing MaxPooling2D
        new_input = tf.keras.Input(shape=(None, None, 3))
        x = new_input
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding)(x)
            else:
                x = layer(x)
        custom_model = tf.keras.Model(new_input, x)
        custom_model.trainable = False

        # Extract outputs for style and content layers
        style_outputs = [custom_model.get_layer(name).output for name in self.style_layers]
        content_output = custom_model.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(custom_model.input, outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrices

        Args:
            input_layer (tf.Tensor): layer output, shape (1, h, w, c).

        Returns:
            tf.Tensor: Gram matrix, shape (1, c, c).
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable))\
                or tf.rank(input_layer).numpy() != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        shape = tf.shape(input_layer)
        _, h, w, c = shape[0], shape[1], shape[2], shape[3]

        features = tf.reshape(input_layer, (h * w, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram /= tf.cast(h * w, tf.float32)
        gram = tf.expand_dims(gram, axis=0)  # shape (1, c, c)
        return gram

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        nb_layers = len(self.style_layers)

        style_img = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        content_img = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_img)
        content_outputs = self.model(content_img)

        self.style_features = [style_outputs[i] for i in range(nb_layers)]
        self.content_feature = content_outputs[nb_layers]

        self.gram_style_features = [NST.gram_matrix(style_feature) for style_feature in self.style_features]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer

        Args:
            style_output (tf.Tensor): output from generated image layer.
            gram_target (tf.Tensor): gram matrix of target style.

        Returns:
            float: style cost for the layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable))\
                or tf.rank(style_output).numpy() != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, h, w, c = tf.shape(style_output)
        if not isinstance(gram_target, (tf.Tensor, tf.Variable))\
                or gram_target.shape != (1, c, c):
            raise TypeError(
                f'gram_target must be a tensor of shape [1, {c}, {c}]'
            )

        gram_style = NST.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image

        Args:
            style_outputs (list of tf.Tensor): style outputs for generated image.

        Returns:
            float: total style cost
        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(f'style_outputs must be a list with length {length}')

        style_score = 0
        weight = 1 / length
        for output, target in zip(style_outputs, self.gram_style_features):
            style_score += weight * self.layer_style_cost(output, target)
        return style_score

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image

        Args:
            content_output (tf.Tensor): content output for generated image.

        Returns:
            float: content cost
        """
        c_feature = self.content_feature
        if not isinstance(content_output, (tf.Tensor, tf.Variable))\
                or content_output.shape != c_feature.shape:
            raise TypeError(f'content_output must be tensor of shape {c_feature.shape}')
        return tf.reduce_mean(tf.square(content_output - c_feature))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image

        Args:
            generated_image (tf.Tensor): generated image tensor.

        Returns:
            tuple: total cost, content cost, style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable))\
                or generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be tensor of shape {self.content_image.shape}')

        processed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(processed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = self.alpha * J_content + self.beta * J_style
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image

        Args:
            generated_image (tf.Tensor): generated image tensor.

        Returns:
            tuple: gradients, total cost, content cost, style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable))\
                or generated_image.shape != self.content_image.shape:
            raise TypeError(f'generated_image must be tensor of shape {self.content_image.shape}')

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)
        gradient = tape.gradient(J_total, generated_image)
        return gradient, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the stylized image

        Args:
            iterations (int): number of iterations for gradient descent.
            step (int or None): step interval to print cost.
            lr (float): learning rate.
            beta1 (float): beta1 parameter for Adam optimizer.
            beta2 (float): beta2 parameter for Adam optimizer.

        Returns:
            tuple: best generated image (numpy array), best loss (float).
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be positive')
        if step is not None:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and less than iterations')
        if not isinstance(lr, (float, int)):
            raise TypeError('lr must be a number')
        if lr < 0:
            raise ValueError('lr must be positive')
        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')
        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 must be in the range [0, 1]')
        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')
        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        init_img = tf.Variable(self.content_image)
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1,
                                       beta_2=beta2, epsilon=1e-1)

        best_loss = float('inf')
        best_img = None

        for iteration in range(iterations):
            gradient, J_total, J_content, J_style = self.compute_grads(init_img)
            optimizer.apply_gradients([(gradient, init_img)])

            if J_total < best_loss:
                best_loss = J_total
                best_img = tf.clip_by_value(init_img[0], 0.0, 1.0)

            if step is not None and iteration % step == 0:
                print(f"Cost at iteration {iteration}: {J_total.numpy()}, "
                      f"content {J_content.numpy()}, style {J_style.numpy()}")

        return best_img.numpy(), best_loss.numpy()
