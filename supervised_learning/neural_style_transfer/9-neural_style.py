#!/usr/bin/env python3
"""
    A script that performs tasks for neural style transfer.
"""
import tensorflow as tf
import numpy as np


class NST:
    """
        A class NST that performs tasks for neural style transfer.
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
            Constructor for NST.

            Args:
                style_image (np.ndarray): the image used with a style
                    reference, stored in a np.ndarray.
                content_image (np.ndarray): the image used with a content
                    reference, stored in a np.ndarray.
                alpha (float): the weight for content cost.
                beta (float): the weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
            Rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 pixels.

            Args:
                image (np.ndarray): A np.ndarray of shape (h, w, 3)
                containing the image to be scaled.

            Returns:
                The scaled image.
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int((w / h) * 512)
        else:
            new_w = 512
            new_h = int((h / w) * 512)
        image_resized = tf.image.resize(image,
                                        (new_h, new_w),
                                        method=tf.image.ResizeMethod.BICUBIC)

        image_resized = tf.clip_by_value(image_resized, 0.0, 255.0)
        image_resized = image_resized / 255.0
        image_resized = tf.cast(image_resized, tf.float32)
        image_batched = tf.expand_dims(image_resized, axis=0)

        return image_batched

    def load_model(self):
        """
            Loads the model for neural style transfer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        outputs = {}
        x = vgg.input

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size,
                                                     name=layer.name)(x)
            else:
                x = layer(x)
            outputs[layer.name] = x

        style_outputs = []
        for name in self.style_layers:
            output = outputs[name]
            style_outputs.append(output)

        content_layer_name = self.content_layer
        content_output = outputs[content_layer_name]

        self.model = tf.keras.Model(inputs=vgg.input,
                                    outputs=style_outputs + [content_output])

        return self.model

    @staticmethod
    def gram_matrix(input_layer):
        """
            Calculates the Gram matrix of a given layer's output.

            Args:
                input_layer (tf.Tensor ot tf.Variable): An instance of
                tf.Tensor or tf.Variable of shape (1, h, w, c) containing
                the layer output whose gram matrix should be calculated.

            Returns:
                A tf.Tensor of shape (1, c, c) containing the gram matrix of
                input_layer.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)
                          ) or input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        clipped = tf.clip_by_value(input_layer, -69.0, 69.0)

        features = tf.reshape(clipped, shape=[-1, c])
        gram = tf.matmul(features, features, transpose_a=True)

        denom = tf.cast(h * w, tf.float32)
        denom = tf.maximum(denom, 1.0)

        gram = gram / denom
        gram = tf.expand_dims(gram, axis=0)

        return gram

    def generate_features(self):
        """
            Extracts the style and content features.
        """
        style_bgr = tf.reverse(self.style_image, axis=[-1])
        content_bgr = tf.reverse(self.content_image, axis=[-1])

        imagenet_mean = tf.constant(
            [103.939, 116.779, 123.68], dtype=tf.float32)
        imagenet_mean = tf.reshape(imagenet_mean, (1, 1, 1, 3))

        style_preprocessed = style_bgr - imagenet_mean
        content_preprocessed = content_bgr - imagenet_mean

        outputs_style = self.model(style_preprocessed)
        style_outputs = outputs_style[:len(self.style_layers)]

        self.style_features = list(style_outputs)
        self.gram_style_features = \
            [self.gram_matrix(feat) for feat in style_outputs]

        outputs_content = self.model(content_preprocessed)
        self.content_feature = outputs_content[len(self.style_layers)]

    def layer_style_cost(self, style_output, gram_target):
        """
            Calculates the style cost for a single layer.

            Args:
                style_output (tf.Tensor): a tf.Tensor of shape (1, h, w, c)
                    containing the layer style output of the generated image.
                gram_target (tf.Tensor): a tf.Tensor of shape (1, c, c)
                the gram matrix of the target style output for that layer.

            Returns:
                The layer's style cost.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)
                          ) or style_output.shape.rank != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = tf.shape(style_output)[-1]
        expected_shape = tf.stack([1, c, c])
        if tf.reduce_any(tf.not_equal(tf.shape(gram_target), expected_shape)):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}] \
                    where {c} is the number of channels in style_output")

        gram_style = self.gram_matrix(style_output)

        layer_style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return layer_style_cost

    def style_cost(self, style_outputs):
        """
            Calculates the style cost.

            Args:
                style_output (tf.Tensor): a list of tf.Tensor style outputs for
                the generated image.

            Returns:
                The style cost.
        """
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len(self.style_layers)):
            raise TypeError(f"style_outputs must be a list with a length of \
                            {len(self.style_layers)}")

        weight = 1.0 / len(self.style_layers)

        total_style_cost = 0.0
        for i in range(len(self.style_layers)):
            cost = self.layer_style_cost(style_outputs[i],
                                         self.gram_style_features[i])
            total_style_cost += weight * cost

        return total_style_cost

    def content_cost(self, content_output):
        """
            Calculates the content cost for the generated image.

            Args:
                content_output (tf.Tensor): a tf.Tensor of the
                content layer output for the generated image.

            Returns:
                The content cost.
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(f"content_output must be a tensor of shape \
                            {self.content_feature.shape}")

        if content_output.shape != self.content_feature.shape:
            raise TypeError(f"content_output must be a tensor of shape \
                            {self.content_feature.shape}")

        cost = tf.reduce_mean(tf.square(content_output - self.content_feature))

        return cost

    def total_cost(self, generated_image):
        """
            Calculates the total cost for the generated image.

            Args:
                generated_image (tf.Tensor): a tf.Tensor of shape
                (1, nh, nw, 3) containing the generated image.

            Returns:
                (J, J_content, J_style):
                    J is the total cost
                    J_content is the content cost
                    J_style is the style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor of shape \
                            {self.content_image.shape}")

        if generated_image.shape != self.content_image.shape:
            raise TypeError(f"generated_image must be a tensor of shape \
                            {self.content_image.shape}")

        outputs = self.model(generated_image)

        style_outputs = outputs[:len(self.style_layers)]
        content_output = outputs[len(self.style_layers):][0]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
            Calculates the gradients for the tf.Tensor generated image of shape
            (1, nh, nw, 3).

            Args:
                generated_image (tf.Tensor): a tf.Tensor of shape
                (1, nh, nw, 3) containing the generated image.

            Returns:
                gradients, J_total, J_content, J_style:
                    gradients is a tf.Tensor containing the gradients for the
                    generated image.
                    J_total is the total cost for the generated image.
                    J_content is the content cost for the generated image.
                    J_style is the style cost for the generated image.
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(f"generated_image must be a tensor of shape \
                            {self.content_image.shape}")

        if generated_image.shape != self.content_image.shape:
            raise TypeError(f"generated_image must be a tensor of shape \
                            {self.content_image.shape}")

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000,
                       step=None,
                       lr=0.01, beta1=0.9,
                       beta2=0.99):
        """
            Generates the neural style transferred image using
            gradient descent.

            Args:
                iterations (int): number of iterations to perform.
                step (int): interval at which to print progress info.
                lr (float): learning rate.
                beta1 (float): beta1 parameter for Adam optimizer.
                beta2 (float): beta2 parameter for Adam optimizer.

            Returns:
                generated_image, cost:
                    generated_image is the best generated image.
                    cost is the best cost.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")

        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             beta_1=beta1,
                                             beta_2=beta2)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image)
            optimizer.apply_gradients([(grads, generated_image)])
            clipped = tf.clip_by_value(generated_image, 0.0, 255.0)
            generated_image.assign(clipped)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: {J_total}, "
                    f"content {J_content}, style {J_style}"
                )

        return best_image, best_cost